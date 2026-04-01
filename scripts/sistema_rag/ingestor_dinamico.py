# RECUERDA QUE ESTE SISTEMA ESTÁ BASADO EN QUE TÚ LE PONDRÁS EL NOMBRE A CADA ARCHIVO QUE SE SUBA
# TENDREMOS QUE GUARDAR EN UNA VARIABLE GLOBAL EL NÚMERO DE DOCUMENTOS QUE TENEMOS EN LA 4
# BASE DE DATOS VECTORIAL INICIAL Y AUMENTAR EN 1 CADA VEZ QUE SE SUBA UNO NUEVO
# ADEMÁS DE QUE PARA RENOMBRAR LOS ARCHIVOS DEBEREMOS USAR ESTA VARIABLE GLOBAL
# ASÍ TAMBIÉN MEJORAMOS LA SEGURIDAD
import json
import fitz
from langchain_core.documents import Document

from scripts.preprocesamiento.extraer_metadatos import procesar_documento
from scripts.preprocesamiento.chunking_embeddings import(
    configurar_chunkeador, generar_id_hash, limpiar_texto, extraer_id)

def carga_diccionario_archivo_unico(archivo_json):
    
    if not (archivo_json) :
        print(f"No existe el json ({archivo_json}). Checkea eso")
        return {}

    with open(archivo_json, "r", encoding="utf-8") as jsonl_metadatos:
        diccionario = {}
        for linea in jsonl_metadatos:
            linea_limpia = linea.strip()
            if linea_limpia:
                item = json.loads(linea)
                nombre_archivo = item.get("archivo")
                if nombre_archivo:
                    id = extraer_id(nombre_archivo)
                    if id:
                        datos = item.copy()
                        diccionario[id] = datos
                    else:
                        print(f"Aviso: No se pudo extraer ID de {nombre_archivo}")
                
    return diccionario, nombre_archivo, id

def ingestar_pdf_individual(ruta_pdf, vector_db, llm):
    """
    Procesa un solo PDF y lo añade a la base de datos vectorial existente.
    """
    print(f"\n[INGESTA INCREMENTAL] Iniciando proceso para: {ruta_pdf}")
    
    # 1. Extraccion de metadatos con tu Pydantic + LLM
    print("[1/4] Extrayendo metadatos con IA...")
    dicc_metadatos = procesar_documento(ruta_pdf, llm)
    json_metadatos = json.dumps(dicc_metadatos, ensure_ascii=False)
    
    # nombre_archivo = os.path.basename(ruta_pdf)
    diccionario_json, archivo, id_actual = carga_diccionario_archivo_unico(json_metadatos)
    # 2. Lectura del texto del PDF
    print("[2/4] Leyendo contenido del PDF...")
    try:
        with fitz.open(ruta_pdf) as pdf_doc:
            for pagina in pdf_doc:
                texto_limpio = limpiar_texto(pagina.get_text("text"))

                if not texto_limpio:
                    continue   

                metadatos_final = {
                    "fuente": archivo,
                    "doc_id": id_actual if id_actual else "unknown",
                    "tipo_documento": "Archivo Externo",
                    "pagina": pagina.number+1 
                }
                tipos_permitidos = (str, int, float, bool)
                for clave, valor in diccionario_json.items():
                    if clave != "archivo" and valor is not None:
                        if isinstance(valor, list):
                            metadatos_final[clave] = ", ".join(map(str, valor))
                        elif isinstance(valor, tipos_permitidos):
                            metadatos_final[clave] = valor
                        else:
                            metadatos_final[clave] = str(valor)
            
                doc = Document(page_content=texto_limpio, metadata=metadatos_final)
            
            print(f"-> {archivo} procesado ({len(pdf_doc)} paginas).")
    except Exception as e:
        print(f"ERROR procesando {archivo}: {e}")

    # 3. Chunking inteligente
    print("[3/4] Troceando el documento...")
    chunky = configurar_chunkeador(800, 100)
    chunks = chunky.split_documents(doc)
    print(f" -> Generados {len(chunks)} fragmentos.")

    # 4. Guardado en ChromaDB
    print("[4/4] Generando embeddings y guardando en BD...")
    try:
        ids = [generar_id_hash(chunk.page_content, chunk.metadata) for chunk in chunks]
        vector_db.add_documents(documents=chunks, ids=ids)
        return True, f"Exito: Se han añadido {len(chunks)} fragmentos a la base de datos."
    except Exception as e:
        for chunk_individual in chunks:
                try:
                    chunk_id = generar_id_hash(chunk_individual.page_content, chunk_individual.metadata)
                    vector_db.add_documents(documents=[chunk_individual], ids=[chunk_id])
                    
                except Exception as ex:
                    print(f"\n[DESCARTADO] Exceso de tokens en ID {chunk_individual.metadata.get('doc_id')}")
                    continue