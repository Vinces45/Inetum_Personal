#https://www.ibm.com/es-es/think/topics/llamaindex

import os
import json
import re
from typing import List, Dict, Any

# Librerias de LangChain y Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# --- CONFIGURACION DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PDFS_PCAP = os.path.join(BASE_DIR, "datos", "pdfs", "pcap")
DIR_PDFS_PPT = os.path.join(BASE_DIR, "datos", "pdfs", "ppt")
DIR_JSON = os.path.join(BASE_DIR, "datos", "metadatos", "metadatos_finales.json")
DIR_DB = os.path.join(BASE_DIR, "datos", "base_datos_vectorial")

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
    archivos = archivos_pcap+archivos_ppt

    for archivo in archivos:
        if "PCAP" in archivo.upper():
            ruta_pdf = os.path.join(dir_pcap, archivo)
            tipo_archivo = "PCAP"
        else:
            ruta_pdf =os.path.join(dir_ppt, archivo)
            tipo_archivo = "PPT"
        
        try:
            # 1. Extraer ID del archivo actual
            id_actual = extraer_id(archivo) 
            
            # 2. Preparar Metadatos Base
            meta_final = {
                "source": archivo,
                "doc_id": id_actual if id_actual else "unknown"
            }
            
            # 3. Detectar si es PCAP o PPT (para el tipo)
            meta_final["tipo_documento"] = tipo_archivo
            
            # 4. BUSCAR EN EL DICCIONARIO POR ID
            # Aqui ocurre la magia: Si el mapa tiene la clave "01",
            # se la aplicara tanto al 01_PCAP como al 01_PPT automaticamente.
            if id_actual and id_actual in mapa_meta:
                datos_json = mapa_meta[id_actual]
                
                # Inyectar datos (Presupuesto, CPV, etc.)
                for clave, valor in datos_json.items():
                    meta_final[clave] = str(valor)
                
                print(f"-> {archivo} (ID: {id_actual}) enriquecido con metadatos.")
            else:
                print(f"-> {archivo} (ID: {id_actual}) NO tiene metadatos en el JSON.")

            # 5. Cargar y Crear Documento
            loader = PyPDFLoader(ruta_pdf)
            paginas = loader.load()
            texto_completo = "\n".join([p.page_content for p in paginas])
            texto_limpio = limpiar_texto(texto_completo)
            
            doc = Document(page_content=texto_limpio, metadata=meta_final)
            docs_lista.append(doc)
            
        except Exception as e:
            print(f"ERROR procesando {archivo}: {e}")
            
    return docs_lista


def configurar_chunking_legal():
    """
    Configuracion CLAVE para documentos legales.
    Intenta no cortar a mitad de un articulo.
    """
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nARTÍCULO", "\nArtículo", 
            "\nCLÁUSULA", "\nCláusula", 
            "\nANEXO", 
            "\n\n", # Parrafos
            ". ",   # Frases
            " "     # Palabras
        ],
        chunk_size=1500,  # Tamaño del bloque (ajustable)
        chunk_overlap=200, # Solapamiento para mantener contexto
        length_function=len
    )

def main():
    print("--- INICIO DE INGESTA Y CHUNKING ---")
    
    # 1. Cargar metadatos del JSON
    print("1. Cargando metadatos...")
    mapa_metadatos = cargar_diccionario_metadatos(DIR_JSON)
    
    # 2. Cargar PDFs y fusionar con metadatos
    print("2. Leyendo PDFs...")
    documentos_base = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_metadatos)
    
    if not documentos_base:
        print("No hay documentos. Saliendo.")
        return

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
    print("4. Usando Ollama (nomic-embed-text) para embeddings...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Si la carpeta ya existe, Chroma intentara añadir, no sobrescribir a lo bruto
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DIR_DB,
        collection_name="pliegos_oficiales"
    )
    
    print(f"--- PROCESO COMPLETADO ---")
    print(f"Base de datos guardada en: {DIR_DB}")

if __name__ == "__main__":
    main()
