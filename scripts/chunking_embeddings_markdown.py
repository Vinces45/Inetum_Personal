import os
import json
import re
import hashlib
from pathlib import Path

# Librerias avanzadas de extraccion y RAG
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

def generar_id_hash(texto):
    texto_bytes = texto.encode("utf-8")
    hash_obj = hashlib.sha256(texto_bytes)
    return hash_obj.hexdigest()

def cargar_diccionario_metadatos(ruta_jsonl):
    if not os.path.exists(ruta_jsonl):
        print(f"No existe el archivo de metadatos en ({ruta_jsonl}).")
        return {}

    diccionario = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            linea_limpia = linea.strip()
            if linea_limpia: #
                item = json.loads(linea_limpia)
                nombre_archivo = item.get("archivo")
                if nombre_archivo:
                    id_doc = extraer_id(nombre_archivo)
                    if id_doc:
                        diccionario[id_doc] = item.copy()
                    else:
                        print(f"Aviso: No se pudo extraer ID de {nombre_archivo}")
    return diccionario

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
            
            # MAGIA: Extraemos Markdown respetando la paginacion
            paginas_md = pymupdf4llm.to_markdown(str(ruta_pdf), page_chunks=True)
            
            for pag_dict in paginas_md:
                texto_md = pag_dict.get("text", "").strip()
                if not texto_md:
                    continue
                
                # Rescatamos el numero de pagina (pymupdf4llm suele guardar metadatos internos)
                num_pag = pag_dict.get("metadata", {}).get("page", 0)
                
                # Preparamos el esqueleto de metadatos hibridos
                meta_final = {
                    "fuente": archivo,
                    "doc_id": id_actual if id_actual else "unknown",
                    "tipo_documento": tipo_archivo,
                    "pagina": num_pag + 1 #A DIFERENCIA QUE LA VERSION ANTERIOR NO SE SI HACE FALTA SUMAR O NO
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
                
                # CAPA 1: Aplicamos el corte semantico a esta pagina en concreto
                md_chunks = markdown_splitter.split_text(texto_md)
                
                # Red de seguridad: si no hay cabeceras, guardamos la pagina entera
                if not md_chunks:
                    md_chunks = [Document(page_content=texto_md, metadata={})]
                
                # Fusionamos los metadatos estructurales de Langchain con los nuestros
                for chunk in md_chunks:
                    # 1. Buscamos el titulo de la seccion que ha detectado el MarkdownSplitter
                    # Usamos Nivel_1 porque asi lo definiste en configurar_splitters
                    seccion = chunk.metadata.get("Nivel_1", "General")
                    id_exp = meta_final.get("doc_id", "N/A")
                    
                    # 2. Modificamos el contenido real: Inyectamos el "DNI" del chunk al principio
                    contexto = f"[EXP {id_exp} | {seccion}] "
                    chunk.page_content = contexto + chunk.page_content
                    
                    # 3. Guardamos metadatos y a la lista
                    chunk.metadata.update(meta_final)
                    docs_lista.append(chunk)
                    
            print(f"-> {archivo} procesado ({len(paginas_md)} paginas mapeadas).")
            
        except Exception as e:
            print(f"ERROR procesando {archivo}: {e}")
            
    return docs_lista

def configurar_splitters():
    # 1. Cortador Semantico (Identifica la jerarquia legal)
    cabeceras_markdown = [
        ("#", "Nivel_1"),
        ("##", "Nivel_2"),
        ("###", "Nivel_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=cabeceras_markdown, 
        strip_headers=False 
    )
    
    # 2. Cortador de Seguridad (Evita trozos masivos y solapa para no perder contexto)
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200, 
        length_function=len
    )

    # rec_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     encoding_name="cl100k_base",
    #     chunk_size=400, # 400 tokens es una medida optima
    #     chunk_overlap=50
    # )
    
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
            ids = [generar_id_hash(chunk.page_content) for chunk in lote_chunks]
            vector_db.add_documents(documents=lote_chunks, ids=ids)
            
        except Exception as e:
            # Estrategia Fallback
            for chunk_individual in lote_chunks:
                try:
                    chunk_id = generar_id_hash(chunk_individual.page_content)
                    vector_db.add_documents(documents=[chunk_individual], ids=[chunk_id])
                except Exception as ex:
                    print(f"\n[DESCARTADO] Fallo en ID documento: {chunk_individual.metadata.get('doc_id')}")
                    continue
            
    print(f"\n--- PROCESO COMPLETADO ---")
    print(f"Base de datos optimizada y guardada en: {DIR_DB}")