import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACION ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 
DIR_DB = os.path.join(PROJECT_ROOT, "datos", "base_datos_vectorial")

def realizar_test():
    print("Conectando con Ollama (mxbai-embed-large)...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    if not os.path.exists(DIR_DB):
        print(f"Error: No se encuentra la base de datos en {DIR_DB}")
        return

    vector_db = Chroma(
        persist_directory=DIR_DB,
        embedding_function=embeddings,
        collection_name="pliegos_oficiales"
    )
    
    print("\n" + "="*50)
    print("SISTEMA DE RECUPERACION CON FILTRO HIBRIDO")
    print("="*50)
    
    while True:
        consulta = input("\nHaz una pregunta (o 'salir'): ")
        if consulta.lower() == "salir": break
            
        # Preguntamos al usuario en que tipo de documento quiere buscar
        tipo_doc = input("¿Filtrar por tipo? (Escribe PCAP, PPT, o pulsa Enter para buscar en todos): ").strip().upper()
        
        # Construimos el filtro dinamicamente
        filtro_metadatos = None
        if tipo_doc in ["PCAP", "PPT"]:
            filtro_metadatos = {"tipo_documento": tipo_doc}
            print(f"-> Aplicando filtro estricto: {filtro_metadatos}")
        else:
            print("-> Buscando en toda la base de datos...")

        try:
            # AQUI ESTA LA MAGIA: Pasamos el filtro a ChromaDB
            if filtro_metadatos:
                resultados = vector_db.similarity_search_with_score(consulta, k=3, filter=filtro_metadatos)
            else:
                resultados = vector_db.similarity_search_with_score(consulta, k=3)
            
            print(f"\nAnalizando {len(resultados)} fragmentos mas cercanos...")
            print("-" * 50)
            
            for i, (doc, score) in enumerate(resultados):
                print(f"RESULTADO #{i+1} | SCORE: {score:.4f} (Menor es mejor)")
                print(f"ARCHIVO: {doc.metadata.get('source', 'Desconocido')}")
                print(f"METADATOS: {doc.metadata}")
                contenido = doc.page_content.replace('\n', ' ').strip()
                print(f"TEXTO: {contenido[:400]}...")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error en la busqueda: {e}")

if __name__ == "__main__":
    realizar_test()