import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Este a diferencia del otro que solo filtraba por tipo de documento (PCAP o PPT) busca con más campos de filtros.
# Se usa ahora el operador $and de ChromaDB para afinar la búsqueda del RAG.


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
        persist_directory=str(DIR_DB),
        embedding_function=embeddings,
        collection_name="pliegos_oficiales"
    )
    
    print("\n" + "="*50)
    print("SISTEMA DE RECUPERACION AVANZADO (TESTING)")
    print("="*50)
    
    while True:
        consulta = input("\nHaz una pregunta (o 'salir'): ")
        if consulta.lower() == "salir": break
            
        tipo_doc = input("¿Filtrar por tipo? (Escribe PCAP, PPT, o pulsa Enter): ").strip().upper()
        tipo_contrato = input("¿Filtrar por contrato? (Obras, Servicios, Suministros, o Enter): ").strip().capitalize()
        
        # Construimos un filtro complejo dinamicamente usando diccionarios de Chroma
        filtros_lista = []
        
        if tipo_doc in ["PCAP", "PPT"]:
            filtros_lista.append({"tipo_documento": tipo_doc})
            
        if tipo_contrato in ["Obras", "Servicios", "Suministros"]:
            filtros_lista.append({"tipo_contrato": tipo_contrato})

        # Aplicamos la logica de filtrado de LangChain/Chroma ($and si hay varios)
        filtro_metadatos = None
        if len(filtros_lista) == 1:
            filtro_metadatos = filtros_lista[0]
        elif len(filtros_lista) > 1:
            filtro_metadatos = {"$and": filtros_lista}

        if filtro_metadatos:
            print(f"-> Aplicando filtro: {filtro_metadatos}")
        else:
            print("-> Buscando en toda la base de datos...")

        try:
            # MAGIA RAG: Recuperacion hibrida (Semantica + Metadatos)
            if filtro_metadatos:
                resultados = vector_db.similarity_search_with_score(consulta, k=3, filter=filtro_metadatos)
            else:
                resultados = vector_db.similarity_search_with_score(consulta, k=3)
            
            print(f"\nAnalizando {len(resultados)} fragmentos mas cercanos...")
            print("-" * 50)
            
            for i, (doc, score) in enumerate(resultados):
                print(f"RESULTADO #{i+1} | DISTANCIA L2: {score:.4f}")
                print(f"ARCHIVO: {doc.metadata.get('source', 'Desconocido')} (Pag. {doc.metadata.get('pagina', 'N/A')})")
                
                # Mostramos solo metadatos clave para no saturar la pantalla
                meta_resumen = {
                    "exp": doc.metadata.get("expediente"),
                    "tipo": doc.metadata.get("tipo_contrato"),
                    "presupuesto": doc.metadata.get("presupuesto")
                }
                print(f"METADATOS: {meta_resumen}")
                
                contenido = doc.page_content.replace('\n', ' ').strip()
                print(f"TEXTO: {contenido[:300]}...\n")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error en la busqueda: {e}")

if __name__ == "__main__":
    realizar_test()