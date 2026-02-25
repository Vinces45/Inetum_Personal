#https://www.ibm.com/es-es/think/topics/llamaindex

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
import time

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
    
    # 1. Normalizar saltos de linea (Windows \r\n a Unix \n)
    texto = texto.replace('\r\n', '\n')
    
    # 2. Reducir multiples saltos de linea a maximo 2 (para marcar parrafos)
    # Esto evita tener 10 lineas en blanco, pero deja separacion entre clausulas
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    
    # 3. Eliminar espacios multiples DENTRO de las lineas (pero no tocar los \n)
    # [ \t]+ significa "espacios o tabuladores", pero NO nueva linea
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
            id = extraer_id(nombre_archivo)
            if id:
                datos = item.copy()
                diccionario[id] = datos
            else:
                print(f"Aviso: No se pudo extraer ID de {nombre_archivo}")
                
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

# def cargar_y_procesar_documentos(dir_pcap, dir_ppt, dicc_mapeo):
#     docs_lista = []
    
#     if not os.path.exists(dir_ppt) or not os.path.exists(dir_pcap):
#         return []

#     archivos_pcap = [f for f in os.listdir(dir_pcap) if f.endswith(".pdf")]
#     archivos_ppt = [f for f in os.listdir(dir_ppt) if f.endswith(".pdf")]
#     archivos = archivos_pcap + archivos_ppt

#     for archivo in archivos:
#         if "PCAP" in archivo.upper():
#             ruta_pdf = os.path.join(dir_pcap, archivo)
#             tipo_archivo = "PCAP"
#         else:
#             ruta_pdf =os.path.join(dir_ppt, archivo)
#             tipo_archivo = "PPT"
        
#         try:
#             # 1. Extraer ID del archivo actual
#             id_actual = extraer_id(archivo) 
            
#             # 2. Preparar Metadatos Base
#             meta_final = {
#                 "source": archivo,
#                 "doc_id": id_actual if id_actual else "unknown"
#             }
            
#             # 3. Detectar si es PCAP o PPT (para el tipo)
#             meta_final["tipo_documento"] = tipo_archivo
            
#             # 4. BUSCAR EN EL DICCIONARIO POR ID
#             # Aqui ocurre la magia: Si el mapa tiene la clave "01",
#             # se la aplicara tanto al 01_PCAP como al 01_PPT automaticamente.
#             if id_actual and id_actual in dicc_mapeo:
#                 datos_json = dicc_mapeo[id_actual]
                
#                 # Inyectar datos (Presupuesto, CPV, etc.)
#                 for clave, valor in datos_json.items():
#                     meta_final[clave] = str(valor)
                
#                 print(f"-> {archivo} (ID: {id_actual}) enriquecido con metadatos.")
#             else:
#                 print(f"-> {archivo} (ID: {id_actual}) NO tiene metadatos en el JSON.")

#             # 5. Cargar y Crear Documento
#             loader = PyPDFLoader(ruta_pdf)
#             paginas = loader.load()
#             texto_completo = "\n".join([p.page_content for p in paginas])
#             texto_limpio = limpiar_texto(texto_completo)
            
#             doc = Document(page_content=texto_limpio, metadata=meta_final)
#             docs_lista.append(doc)
            
#         except Exception as e:
#             print(f"ERROR procesando {archivo}: {e}")
            
#     return docs_lista


def configurar_chunking_legal():
    
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
        chunk_size=1500, #1500 // 1000 // 800 // 600 
        chunk_overlap=200, #200 // 150 // 120 // 100
        length_function=len,
        is_separator_regex=True
    )

if __name__ == "__main__":
    print("--- INICIO DE INGESTA Y CHUNKING ---")
    
    # 1. Cargar metadatos del JSON
    print("1. Cargando metadatos...")
    dicc_mapeodatos = cargar_diccionario_metadatos(DIR_JSON)
    
    # 2. Cargar PDFs y fusionar con metadatos
    print("2. Leyendo PDFs...")
    documentos_base = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, dicc_mapeodatos)
   

    # 3. CHUNKING (El paso crucial que pedias)
    print("3. Ejecutando Chunking Inteligente...")
    splitter = configurar_chunking_legal()
    chunks = splitter.split_documents(documentos_base)
    
    print(f"   Originales: {len(documentos_base)} docs -> Generados: {len(chunks)} chunks.")
    
    # Verificacion: Imprimir un chunk al azar para ver si tiene metadatos
    if chunks:
        print("\n[INSPECCION DE CHUNK]")
        print(f"Texto: {chunks[0].page_content[:100]}...")
        print(f"Metadatos Heredados: {chunks[0].metadata}\n")

    # 4. EMBEDDINGS Y GUARDADO (ChromaDB)
    print("4. Usando Ollama (mxbai-embed-large) para embeddings...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=DIR_DB,
        collection_name="pliegos_oficiales"
    )

    TAMANO_LOTE = 50 
    print(f"\nIniciando guardado en ChromaDB por lotes de {TAMANO_LOTE}...")
    
    for i in tqdm(range(0, len(chunks), TAMANO_LOTE), desc="Progreso Embeddings"):
        lote_chunks = chunks[i : i + TAMANO_LOTE]
        try:
            # Asignar IDs unicos para evitar duplicados en reingestas futuras
            ids = [f"{chunk.metadata['doc_id']}_p{chunk.metadata['pagina']}_c{idx}" for idx, chunk in enumerate(lote_chunks)] # Checkear esto porque no entiendo que hace buscar alguna forma que si entienda
            vector_db.add_documents(documents=lote_chunks, ids=ids)
        # Esta excepcion no se si es necesario lo es¿? y como funciona ¿?
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
