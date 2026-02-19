import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACION ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 
DIR_DB = os.path.join(PROJECT_ROOT, "datos", "base_datos_vectorial")

def realizar_test():
    # 1. Cargamos el mismo modelo de embeddings que usamos en la ingesta
    print("Conectando con Ollama (mxbai-embed-large)...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # 2. Conectamos a la base de datos existente (SOLO LECTURA)
    if not os.path.exists(DIR_DB):
        print(f"Error: No se encuentra la base de datos en {DIR_DB}")
        return

    vector_db = Chroma(
        persist_directory=DIR_DB,
        embedding_function=embeddings,
        collection_name="pliegos_oficiales"
    )
    
    print("\n" + "="*50)
    print("SISTEMA DE RECUPERACION DE PLIEGOS (TEST)")
    print("="*50)
    print("Escribe 'salir' para cerrar el programa.\n")
    
    while True:
        consulta = input("Haz una pregunta sobre los pliegos: ")
        
        if consulta.lower() == "salir":
            break
            
        # 3. Busqueda por similitud
        # k=3 pide los 3 fragmentos mas relevantes
        # similarity_search_with_score nos da la 'distancia' (cuanto mas baja, mejor)
        resultados = vector_db.similarity_search_with_score(consulta, k=5)
        
        print(f"\nAnalizando {len(resultados)} fragmentos mas cercanos...")
        print("-" * 50)
        
        for i, (doc, score) in enumerate(resultados):
            print(f"RESULTADO #{i+1} | SCORE: {score:.4f} (Menor es mejor)")
            print(f"ARCHIVO: {doc.metadata.get('source', 'Desconocido')}")
            print(f"TIPO: {doc.metadata.get('tipo_documento', 'N/A')} | ID: {doc.metadata.get('doc_id', 'N/A')}")
            
            # Mostramos un resumen del contenido encontrado
            contenido = doc.page_content.replace('\n', ' ').strip()
            print(f"TEXTO: {contenido[:400]}...")
            print("-" * 50)
        print("\n")

if __name__ == "__main__":
    realizar_test()