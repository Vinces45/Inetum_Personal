import random
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DIR_DB = PROJECT_ROOT / "datos" / "base_datos_vectorial"

def generar_muestras_para_json(cantidad_muestras=5):
    print("--- GENERADOR DE GROUND TRUTH PARA RAG ---")
    
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DIR_DB),
        collection_name="pliegos_oficiales" 
    )
    
    # Extraemos todos los documentos de la BD
    datos_db = db.get()
    ids = datos_db['ids']
    documentos = datos_db['documents']
    metadatos = datos_db['metadatas']
    
    if not ids:
        print("La base de datos esta vacia. Ejecuta primero la ingesta.")
        return
        
    # Seleccionamos indices aleatorios
    indices_aleatorios = random.sample(range(len(ids)), min(cantidad_muestras, len(ids)))
    
    print("\nLee estos fragmentos y crea una pregunta especifica para cada uno:\n")
    
    for i, idx in enumerate(indices_aleatorios):
        texto = documentos[idx]
        meta = metadatos[idx]
        
        doc_id = meta.get('doc_id', 'Desconocido')
        pagina = meta.get('pagina', 'Desconocida')
        
        print("="*60)
        print(f"MUESTRA {i+1}")
        print(f"-> DOC ID Esperado: {doc_id}")
        print(f"-> Pagina Esperada: {pagina}")
        print("-" * 60)
        print(f"TEXTO DEL CHUNK:\n{texto[:500]}... [TEXTO CORTADO PARA LECTURA RAPIDA]")
        print("="*60)
        print("Escribe tu pregunta en el JSON basandote SOLO en este texto.\n")

if __name__ == "__main__":
    generar_muestras_para_json(16)