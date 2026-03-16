from pathlib import Path
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURACION ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 
DIR_DB = PROJECT_ROOT / "datos" /"base_datos_vectorial_alpha"

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
    print("SISTEMA DE RECUPERACION AVANZADO (TESTING)")
    print("="*50)
    
    while True:
        consulta = input("\nHaz una pregunta (o 'salir'): ")
        if consulta.lower() == "salir": break
            
        print("PONER -1 SI NO QUIERES FILTRO")
        tipo_doc = input("¿Filtrar por tipo? (Escribe PCAP, PPT, o pulsa Enter): ").strip().upper()
        presupuesto_base = int(input("Inserte presupuesto base (Sin puntos ni comas, a excepción de los puntos para marcar los céntimos)"))
        valor_estimado = int(input("Inserte valor estimado total (Sin puntos ni comas, a excepción de los puntos para marcar los céntimos)"))
        iva = int(input("Inserte valor de IVA (sin símbolo de porcentaje)"))
        plazo_meses = int(input("Inserte número de meses de plazo ordinario"))
        prorroga = int(input("Inserte número de meses de prorroga"))
        tramitacion = input("Inserte tipo de tramitación (Abierto Simplificado, Negociado sin publicidad)")
        lotes = input("Inserte si tiene lotes o no (S/N)").upper()
        if lotes == "SI":
            lotes_bool = True
        else:
            lotes_bool = False
        financiacion_europea = input("Inserte si tiene financiación europea a o no (S/N)").upper()
        if financiacion_europea == "SI":
            financiacion_europea_bool = True
        else:
            financiacion_europea_bool = False
        tipo_contrato = input("¿Filtrar por contrato? (Obras, Servicios, Suministros, o Enter): ").strip().capitalize()
        
        # Construimos un filtro complejo dinamicamente usando diccionarios de Chroma
        filtros_lista = []
        
        if tipo_doc in ["PCAP", "PPT"]:
            filtros_lista.append({"tipo_documento": tipo_doc})
            
        if tipo_contrato in ["Obras", "Servicios", "Suministros"]:
            filtros_lista.append({"tipo_contrato": tipo_contrato})

        if presupuesto_base != -1:
            filtros_lista.append({"presupuesto_base": presupuesto_base})
            
        if valor_estimado != -1:
            filtros_lista.append({"valor_estimado": valor_estimado})

        if iva != -1:
            filtros_lista.append({"iva": iva})
        
        if plazo_meses != -1:
            filtros_lista.append({"plazo_meses": plazo_meses})

        if prorroga != -1:
            filtros_lista.append({"prorroga": prorroga})

        if tramitacion != "-1" in ["Abierto Simplificado, Negociado sin publicidad"]:
            filtros_lista.append({"tramitacion": tramitacion})

        if lotes != "-1":
            filtros_lista.append({"lotes": lotes_bool})

        if financiacion_europea != "-1":
            filtros_lista.append({"financiacion_europea": financiacion_europea_bool})

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
            # Recuperacion hibrida (Semantica + Metadatos)
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