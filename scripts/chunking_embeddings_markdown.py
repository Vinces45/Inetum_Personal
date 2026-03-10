import os
import json
import re
import hashlib
from pathlib import Path

import pymupdf4llm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from tqdm import tqdm

# --- CONFIGURACION DE RUTAS ---  
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 

DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON = PROJECT_ROOT / "datos" / "jsons" / "metadatos.jsonl"
DIR_DB = PROJECT_ROOT / "datos" / "base_datos_vectorial"

def extraer_id(nombre_archivo):
    match = re.match(r"^(\d+)_", nombre_archivo)
    if match:
        return match.group(1)
    return None

def generar_id_hash(texto, metadatos):
    id_doc = metadatos.get("doc_id", "sin_id")
    pagina = metadatos.get("pagina", 0)
    texto_base = f"{id_doc}_{pagina}_{texto}"
    return hashlib.sha256(texto_base.encode("utf-8")).hexdigest()

def normalizar_cabeceras_markdown(texto_md):
    patron_titulos_universales = r"^(?:\*\*)?(\d{1,2}[\.\-]\s+[A-Z].*?)(?:\*\*)?$"
    texto_corregido = re.sub(patron_titulos_universales, r"# \1", texto_md, flags=re.MULTILINE)
    return texto_corregido

def cargar_diccionario_metadatos(ruta_jsonl):
    if not os.path.exists(ruta_jsonl):
        print(f"No existe el archivo de metadatos en ({ruta_jsonl}).")
        return {}

    diccionario = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            linea_limpia = linea.strip()
            if linea_limpia:
                item = json.loads(linea_limpia)
                nombre_archivo = item.get("archivo")
                if nombre_archivo:
                    id_doc = extraer_id(nombre_archivo)
                    if id_doc:
                        diccionario[id_doc] = item.copy()
                    else:
                        print(f"Aviso: No se pudo extraer ID de {nombre_archivo}")
    return diccionario

# def procesar_documentos_y_capa_semantica(dir_pcap, dir_ppt, dicc_metadatos, markdown_splitter):
#     docs_lista = []

#     archivos_pcap = [f for f in os.listdir(dir_pcap) if f.endswith(".pdf")]
#     archivos_ppt = [f for f in os.listdir(dir_ppt) if f.endswith(".pdf")]
#     archivos = archivos_pcap + archivos_ppt

#     for archivo in archivos:
#         es_pcap = "PCAP" in archivo.upper()
#         ruta_pdf = dir_pcap / archivo if es_pcap else dir_ppt / archivo
#         tipo_archivo = "PCAP" if es_pcap else "PPT"
        
#         try:
#             id_actual = extraer_id(archivo) 
#             datos_jsonl = dicc_metadatos.get(id_actual, {})
            
#             # Extraemos todas las paginas
#             paginas_md = pymupdf4llm.to_markdown(str(ruta_pdf), page_chunks=True)
            
#             texto_completo_md = ""
            
#             # 1. ENSAMBLAJE DEL DOCUMENTO COMPLETO
#             for pag_dict in paginas_md:
#                 texto_md = pag_dict.get("text", "").strip()
#                 if not texto_md:
#                     continue
#                 num_pag = pag_dict.get("metadata", {}).get("page", 0) + 1
#                 # Inyectamos un marcador sutil por si necesitamos saber la pagina luego
#                 texto_completo_md += f"\n\n\n\n" + texto_md

#             if not texto_completo_md.strip():
#                 continue

#             # 2. NORMALIZACION
#             texto_md_limpio = normalizar_cabeceras_markdown(texto_completo_md)
            
#             # 3. METADATOS A NIVEL DE DOCUMENTO
#             meta_final = {
#                 "fuente": archivo,
#                 "doc_id": id_actual if id_actual else "unknown",
#                 "tipo_documento": tipo_archivo
#             }

#             tipos_permitidos = (str, int, float, bool)  
#             for clave, valor in datos_jsonl.items():
#                 if clave != "archivo" and valor is not None:
#                     if isinstance(valor, list):
#                         meta_final[clave] = ", ".join(map(str, valor))
#                     elif isinstance(valor, tipos_permitidos):
#                         meta_final[clave] = valor
#                     else:
#                         meta_final[clave] = str(valor)
            
#             # 4. SPLIT SEMANTICO AL DOCUMENTO ENTERO
#             md_chunks = markdown_splitter.split_text(texto_md_limpio)
            
#             if not md_chunks:
#                 md_chunks = [Document(page_content=texto_md_limpio, metadata={})]
            
#             # 5. INYECCION DE CONTEXTO Y METADATOS EN CADA CHUNK
#             for chunk in md_chunks:
#                 niveles = []
#                 if "Clausula" in chunk.metadata: 
#                     niveles.append(chunk.metadata["Clausula"])
#                 if "Apartado" in chunk.metadata: 
#                     niveles.append(chunk.metadata["Apartado"])
#                 if "Subapartado" in chunk.metadata: 
#                     niveles.append(chunk.metadata["Subapartado"])
                
#                 seccion = " > ".join(niveles) if niveles else "General"
#                 id_exp = meta_final.get("doc_id", "N/A")
                
#                 contexto = f"[EXP {id_exp} | {seccion}]\n"
#                 chunk.page_content = contexto + chunk.page_content
                
#                 chunk.metadata.update(meta_final)
#                 docs_lista.append(chunk)
                
#             print(f"-> {archivo} procesado (Estructura semantica preservada).")
            
#         except Exception as e:
#             print(f"ERROR procesando {archivo}: {e}")
            
#     return docs_lista

def procesar_documentos_y_capa_semantica(dir_pcap, dir_ppt, dicc_metadatos, markdown_splitter):
    docs_lista = []

    archivos_pcap = [f for f in os.listdir(dir_pcap) if f.endswith(".pdf")]
    archivos_ppt = [f for f in os.listdir(dir_ppt) if f.endswith(".pdf")]
    archivos = archivos_pcap + archivos_ppt

    for archivo in archivos:
        es_pcap = "PCAP" in archivo.upper()
        ruta_pdf = dir_pcap / archivo if es_pcap else dir_ppt / archivo
        tipo_archivo = "PCAP" if es_pcap else "PPT"
        
        try:
            id_actual = extraer_id(archivo) 
            datos_jsonl = dicc_metadatos.get(id_actual, {})
            
            # Extraemos todas las paginas
            paginas_md = pymupdf4llm.to_markdown(str(ruta_pdf), page_chunks=True)
            
            texto_completo_md = ""
            
            # 1. ENSAMBLAJE DEL DOCUMENTO COMPLETO CON MARCADORES
            for pag_dict in paginas_md:
                texto_md = pag_dict.get("text", "").strip()
                if not texto_md:
                    continue
                
                num_pag = pag_dict.get("metadata", {}).get("page", 0) + 1
                # INYECCION DEL TOKEN DE PAGINA
                texto_completo_md += f"\n\n[MARCADOR_PAGINA:{num_pag}]\n\n{texto_md}"

            if not texto_completo_md.strip():
                continue

            # 2. NORMALIZACION
            texto_md_limpio = normalizar_cabeceras_markdown(texto_completo_md)
            
            # 3. METADATOS A NIVEL DE DOCUMENTO
            meta_final = {
                "fuente": archivo,
                "doc_id": id_actual if id_actual else "unknown",
                "tipo_documento": tipo_archivo
            }

            tipos_permitidos = (str, int, float, bool)  
            for clave, valor in datos_jsonl.items():
                if clave != "archivo" and valor is not None:
                    if isinstance(valor, list):
                        meta_final[clave] = ", ".join(map(str, valor))
                    elif isinstance(valor, tipos_permitidos):
                        meta_final[clave] = valor
                    else:
                        meta_final[clave] = str(valor)
            
            # 4. SPLIT SEMANTICO AL DOCUMENTO ENTERO
            md_chunks = markdown_splitter.split_text(texto_md_limpio)
            
            if not md_chunks:
                md_chunks = [Document(page_content=texto_md_limpio, metadata={})]
            
            # 5. EXTRACCION DE TOKENS E INYECCION DE CONTEXTO
            pagina_actual = 1 # Variable de estado para arrastrar la pagina
            
            for chunk in md_chunks:
                # Buscamos si el chunk contiene nuestro token de pagina
                marcadores = re.findall(r"\[MARCADOR_PAGINA:(\d+)\]", chunk.page_content)
                
                if marcadores:
                    # Si hay varios, cogemos el primero para indicar donde empieza la seccion
                    pagina_actual = int(marcadores[0])
                
                # Limpiamos el texto borrando los marcadores para no ensuciar los embeddings
                texto_limpio_chunk = re.sub(r"\[MARCADOR_PAGINA:\d+\]\s*", "", chunk.page_content)
                chunk.page_content = texto_limpio_chunk.strip()
                
                # GUARDAMOS LA PAGINA RECUPERADA EN LOS METADATOS
                chunk.metadata["pagina"] = pagina_actual
                
                niveles = []
                if "Clausula" in chunk.metadata: 
                    niveles.append(chunk.metadata["Clausula"])
                if "Apartado" in chunk.metadata: 
                    niveles.append(chunk.metadata["Apartado"])
                if "Subapartado" in chunk.metadata: 
                    niveles.append(chunk.metadata["Subapartado"])
                
                seccion = " > ".join(niveles) if niveles else "General"
                id_exp = meta_final.get("doc_id", "N/A")
                
                contexto = f"[EXP {id_exp} | {seccion}]\n"
                chunk.page_content = contexto + chunk.page_content
                
                chunk.metadata.update(meta_final)
                docs_lista.append(chunk)
                
            print(f"-> {archivo} procesado (Estructura semantica y paginacion preservadas).")
            
        except Exception as e:
            print(f"ERROR procesando {archivo}: {e}")
            
    return docs_lista

def configurar_splitters():
    # 1. Cortador Semantico (Markdown - Capa 1)
    cabeceras_markdown = [
        ("#", "Clausula"),
        ("##", "Apartado"),
        ("###", "Subapartado")
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=cabeceras_markdown, 
        strip_headers=False 
    )
    
    # 2. Cortador de Seguridad Inteligente (Regex Legal - Capa 2)
    rec_splitter = RecursiveCharacterTextSplitter(
        separators=[
            # 1. Numeracion principal
            r"\n\d+[\.\-]+\s+[A-Z]", 
            # 2. Numeracion de subapartados
            r"\n\d+\.\d+\.\s+[A-Z]",
            # 3. Letras mayusculas
            r"\n[A-Z]\.\s+[A-Z]",
            # 4. Palabras clave clasicas
            r"\nART.CULO", r"\nArt.culo", 
            r"\nCL.USULA", r"\nCl.usula", 
            r"\nANEXO", r"\nAnexo",
            # 5. Saltos de parrafo estandar (Vital para no romper tablas de Markdown)
            r"\n\n", 
            # 6. Listas con guiones o letras
            r"\n-\s",
            r"\n[a-z]\)\s",
            # 7. Red de seguridad final: frases y espacios
            r"\.\s", 
            r" "
        ],
        chunk_size=600, 
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=True
    )
    
    return md_splitter, rec_splitter

if __name__ == "__main__":
    print("--- INICIO DE INGESTA AVANZADA (TWO-STAGE CHUNKING) ---")
    
    print("\n1. Cargando metadatos (Lazy Loading JSONL)...")
    mapa_metadatos = cargar_diccionario_metadatos(DIR_JSON)
    
    print("\n2. Configurando motores de chunking...")
    markdown_splitter, recursive_splitter = configurar_splitters()
    
    print("\n3. Leyendo PDFs, extrayendo Markdown y aplicando Capa 1 (Semantica)...")
    docs_semanticos = procesar_documentos_y_capa_semantica(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_metadatos, markdown_splitter)
    
    print("\n4. Aplicando Capa 2 (Seguridad por tamaño y solapamiento)...")
    chunks_finales = recursive_splitter.split_documents(docs_semanticos)
    
    print(f"   -> Resultado: Generados {len(chunks_finales)} chunks listos para vectorizar.")
    
    print("\n5. Inicializando Ollama y ChromaDB...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DIR_DB),
        collection_name="pliegos_oficiales"
    )

    TAMANO_LOTE = 50 
    print(f"\n6. Guardando en ChromaDB por lotes de {TAMANO_LOTE} (con prevencion de duplicados)...")
    
    for i in tqdm(range(0, len(chunks_finales), TAMANO_LOTE), desc="Progreso Embeddings"):
        lote_chunks = chunks_finales[i : i + TAMANO_LOTE]
        try:
            ids = [generar_id_hash(chunk.page_content, chunk.metadata) for chunk in lote_chunks]
            vector_db.add_documents(documents=lote_chunks, ids=ids)
            
        except Exception as e:
            for chunk_individual in lote_chunks:
                try:
                    chunk_id = generar_id_hash(chunk_individual.page_content, chunk_individual.metadata)
                    vector_db.add_documents(documents=[chunk_individual], ids=[chunk_id])
                except Exception as ex:
                    print(f"\n[DESCARTADO] Fallo en ID documento: {chunk_individual.metadata.get('doc_id')}")
                    continue
            
    print(f"\n--- PROCESO COMPLETADO ---")
    print(f"Base de datos optimizada y guardada en: {DIR_DB}")