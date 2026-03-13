#https://www.ibm.com/es-es/think/topics/llamaindex

import os
import json
import re
import hashlib
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from tqdm import tqdm

# --- CONFIGURACION DE RUTAS ---  
BASE_DIR = Path(__file__).resolve().parent.parent     
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
    texto_bytes = texto_base.encode("utf-8")
    hash_obj = hashlib.sha256(texto_bytes)
    return hash_obj.hexdigest()

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

    with open(ruta_json, "r", encoding="utf-8") as jsonl_metadatos:
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
                
    return diccionario

def cargar_y_procesar_documentos(dir_pcap, dir_ppt, dicc_metadatos):
    docs_lista = []
    
    archivos_pcap = [f for f in os.listdir(dir_pcap) if f.endswith(".pdf")]
    archivos_ppt = [f for f in os.listdir(dir_ppt) if f.endswith(".pdf")]
    archivos = archivos_pcap + archivos_ppt

    for archivo in archivos:
        es_pcap = "PCAP" in archivo.upper()
        if es_pcap:
            ruta_pdf = dir_pcap / archivo 
            tipo_archivo = "PCAP"
        else:
            ruta_pdf = dir_ppt / archivo
            tipo_archivo = "PPT"
        
        try:
            id_actual = extraer_id(archivo) 

            datos_jsonl = dicc_metadatos.get(id_actual, {})
            
            with fitz.open(ruta_pdf) as pdf_doc:
                for pagina in pdf_doc:
                    texto_limpio = limpiar_texto(pagina.get_text("text"))

                    if not texto_limpio:
                        continue   

                    metadatos_final = {
                        "fuente": archivo,
                        "doc_id": id_actual if id_actual else "unknown",
                        "tipo_documento": tipo_archivo,
                        "pagina": pagina.number+1 
                    }
                    tipos_permitidos = (str, int, float, bool)
                    for clave, valor in datos_jsonl.items():
                        if clave != "archivo" and valor is not None:
                            if isinstance(valor, list):
                                metadatos_final[clave] = ", ".join(map(str, valor))
                            elif isinstance(valor, tipos_permitidos):
                                metadatos_final[clave] = valor
                            else:
                                metadatos_final[clave] = str(valor)
                
                    doc = Document(page_content=texto_limpio, metadata=metadatos_final)
                    docs_lista.append(doc)
                
                print(f"-> {archivo} procesado ({len(pdf_doc)} paginas).")
            
        except Exception as e:
            print(f"ERROR procesando {archivo}: {e}")
            
    return docs_lista


def configurar_chunkeador(chunk_size_par = 1500, chunk_overlap_par = 200):
    return RecursiveCharacterTextSplitter(
        separators=[
            # 1. Numeracion principal (Ej: "1. OBJETO" o "1.- CARACTER")
            # \d+ (numeros), [\.\-]+ (punto o guion), \s+ (espacios), [A-Z] (mayuscula)
            r"\n\d+[\.\-]+\s+[A-Z]", 
            
            # 2. Numeracion de subapartados (Ej: "1.1. Presupuesto")
            r"\n\d+\.\d+\.\s+[A-Z]",
            
            # 3. Letras mayusculas (Ej: "A. CONDICIONES MINIMAS")
            r"\n[A-Z]\.\s+[A-Z]",
            
            # 4. Palabras clave clasicas (el comodin '.' evita tildes en el codigo)
            r"\nART.CULO", r"\nArt.culo", 
            r"\nCL.USULA", r"\nCl.usula", 
            r"\nANEXO", r"\nAnexo",
            
            # 5. Saltos de parrafo estandar
            r"\n\n", 
            
            # 6. Listas con guiones o letras (Ej: "- Tarea 1" o "a) Tarea 2")
            r"\n-\s",
            r"\n[a-z]\)\s",
            
            # 7. Red de seguridad: Frases y palabras
            r"\.\s", 
            r" "
        ],
        chunk_size=chunk_size_par, #1500 // 1000 // 800 // 600 
        chunk_overlap=chunk_overlap_par, #200 // 150 // 120 // 100
        length_function=len,
        is_separator_regex=True
    )

if __name__ == "__main__":
    print("--- INICIO DE INGESTA Y CHUNKING ---")
    
    print("1. Cargando metadatos...")
    diccionario_metadatos = cargar_diccionario_metadatos(DIR_JSON)
    
    print("2. Leyendo PDFs...")
    lista_documentos_base = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, diccionario_metadatos)
   

    print("3. Ejecutando Chunking Inteligente...")
    chunky = configurar_chunkeador()
    chunks = chunky.split_documents(lista_documentos_base)
    
    print(f"   Originales: {len(lista_documentos_base)} docs -> Generados: {len(chunks)} chunks.")

    print("4. Usando Ollama (mxbai-embed-large) para embeddings...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=DIR_DB,
        collection_name="pliegos_oficiales"
    )

    TAMAÑO_LOTE = 50 
    print(f"\nIniciando guardado en ChromaDB por lotes de {TAMAÑO_LOTE}...")
    
    for i in tqdm(range(0, len(chunks), TAMAÑO_LOTE), desc="Progreso Embeddings"):
        lote_chunks = chunks[i : i + TAMAÑO_LOTE]
        
        try:
            ids = [generar_id_hash(chunk.page_content, chunk.metadata) for chunk in lote_chunks] 
            vector_db.add_documents(documents=lote_chunks, ids=ids)
            
        except Exception as e:
            for chunk_individual in lote_chunks:
                try:
                    chunk_id = generar_id_hash(chunk_individual.page_content, chunk_individual.metadata)
                    vector_db.add_documents(documents=[chunk_individual], ids=[chunk_id])
                    
                except Exception as ex:
                    print(f"\n[DESCARTADO] Exceso de tokens en ID {chunk_individual.metadata.get('doc_id')}")
                    continue
            
    print(f"\n--- PROCESO COMPLETADO ---")
    print(f"Base de datos guardada correctamente en: {DIR_DB}")