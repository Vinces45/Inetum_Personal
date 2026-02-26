import os
import json
import re
from pathlib import Path

# Librerias de LangChain y Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from tqdm import tqdm

# Se guarda el ids de las páginas que se hagan chunking así nos puede ayudar a futuro (doc_id + pagina + numero de chunk) para la reingesta de documentos
# así cuando se ejecuta de nuevo este script en lugar de duplicar los vectores lo que hace es actualizarse.
# 

# --- CONFIGURACION DE RUTAS ---  
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 

DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON = PROJECT_ROOT / "datos" / "metadatos" / "metadatos_finales.json"
DIR_DB = PROJECT_ROOT / "datos" / "base_datos_vectorial"

def extraer_id(nombre_archivo):
    match = re.match(r"^(\d+)_", nombre_archivo)
    if match:
        return match.group(1)
    return None

def limpiar_texto(texto):
    if not texto:
        return ""
    texto = texto.replace('\r\n', '\n')
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r'[ \t]+', ' ', texto)
    return texto.strip()

def cargar_diccionario_metadatos(ruta_json):
    if not os.path.exists(ruta_json):
        print(f"No existe el json en la ruta administrada ({ruta_json}). Checkea eso")
        return {}

    with open(ruta_json, "r", encoding="utf-8") as f:
        lista_datos = json.load(f)
    
    diccionario = {}
    for item in lista_datos:
        nombre_archivo = item.get("archivo")
        if nombre_archivo:
            id_doc = extraer_id(nombre_archivo)
            if id_doc:
                diccionario[id_doc] = item.copy()
    return diccionario

def cargar_y_procesar_documentos(dir_pcap, dir_ppt, mapa_meta):
    docs_lista = []
    
    if not os.path.exists(dir_ppt) or not os.path.exists(dir_pcap):
        return []

    archivos_pcap = [f for f in os.listdir(dir_pcap) if f.endswith(".pdf")]
    archivos_ppt = [f for f in os.listdir(dir_ppt) if f.endswith(".pdf")]
    archivos = archivos_pcap + archivos_ppt

    for archivo in archivos:
        es_pcap = "PCAP" in archivo.upper()
        ruta_pdf = os.path.join(dir_pcap if es_pcap else dir_ppt, archivo)
        tipo_archivo = "PCAP" if es_pcap else "PPT"
        
        try:
            id_actual = extraer_id(archivo) 
            
            # Buscamos los metadatos globales del expediente
            datos_json = mapa_meta.get(id_actual, {})
            
            loader = PyPDFLoader(ruta_pdf)
            paginas = loader.load()
            
            # IMPORTANTE: Procesamos pagina por pagina para conservar el numero de pagina original
            for pagina in paginas:
                texto_limpio = limpiar_texto(pagina.page_content)
                if not texto_limpio:
                    continue
                    
                # Creamos metadatos especificos para esta pagina
                meta_final = {
                    "source": archivo,
                    "doc_id": id_actual if id_actual else "unknown",
                    "tipo_documento": tipo_archivo,
                    "pagina": pagina.metadata.get("page", 0) # Conservamos la pagina
                }
                
                # Inyectar datos (Presupuesto, CPV, etc.) del JSON
                for clave, valor in datos_json.items():
                    # Evitamos sobreescribir el 'archivo' original por el de la metadata (ej. PPT tomando nombre de PCAP)
                    if clave != "archivo": 
                        meta_final[clave] = str(valor)
                
                doc = Document(page_content=texto_limpio, metadata=meta_final)
                docs_lista.append(doc)
                
            print(f"-> {archivo} procesado ({len(paginas)} paginas).")
            
        except Exception as e:
            print(f"ERROR procesando {archivo}: {e}")
            
    return docs_lista

def configurar_chunking_legal():
    return RecursiveCharacterTextSplitter(
        separators=[
            r"\n\d+[\.\-]+\s+[A-Z]", 
            r"\n\d+\.\d+\.\s+[A-Z]",
            r"\n[A-Z]\.\s+[A-Z]",
            r"\nART.CULO", r"\nArt.culo", 
            r"\nCL.USULA", r"\nCl.usula", 
            r"\nANEXO", r"\nAnexo",
            r"\n\n", 
            r"\n-\s",
            r"\n[a-z]\)\s",
            r"\.\s", 
            r" "
        ],
        chunk_size=1500, 
        chunk_overlap=200, 
        length_function=len,
        is_separator_regex=True
    )

if __name__ == "__main__":
    print("--- INICIO DE INGESTA Y CHUNKING ---")
    
    print("1. Cargando metadatos...")
    mapa_metadatos = cargar_diccionario_metadatos(DIR_JSON)
    
    print("2. Leyendo PDFs y conservando paginas...")
    documentos_base = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_metadatos)
    
    print("3. Ejecutando Chunking Inteligente...")
    splitter = configurar_chunking_legal()
    chunks = splitter.split_documents(documentos_base)
    
    print(f"   Originales: {len(documentos_base)} paginas -> Generados: {len(chunks)} chunks.")
    
    print("4. Usando Ollama (mxbai-embed-large) para embeddings...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # IMPORTANTE: Si vas a reingestar, asegurate de limpiar la coleccion previa o usar IDs unicos
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DIR_DB),
        collection_name="pliegos_oficiales"
    )

    TAMANO_LOTE = 50 
    print(f"\nIniciando guardado en ChromaDB por lotes de {TAMANO_LOTE}...")
    
    for i in tqdm(range(0, len(chunks), TAMANO_LOTE), desc="Progreso Embeddings"):
        lote_chunks = chunks[i : i + TAMANO_LOTE]
        try:
            # Asignar IDs unicos para evitar duplicados en reingestas futuras
            ids = [f"{chunk.metadata['doc_id']}_p{chunk.metadata['pagina']}_c{idx}" for idx, chunk in enumerate(lote_chunks)]
            vector_db.add_documents(documents=lote_chunks, ids=ids)
        except Exception as e:
            for idx, chunk_individual in enumerate(lote_chunks):
                try:
                    chunk_id = f"{chunk_individual.metadata['doc_id']}_p{chunk_individual.metadata['pagina']}_c{idx}_fallback"
                    vector_db.add_documents(documents=[chunk_individual], ids=[chunk_id])
                except Exception as ex:
                    print(f"\n[DESCARTADO] Exceso de tokens en ID {chunk_individual.metadata.get('doc_id')}, Pag {chunk_individual.metadata.get('pagina')}")
                    continue
            
    print(f"\n--- PROCESO COMPLETADO ---")
    print(f"Base de datos guardada correctamente en: {DIR_DB}")