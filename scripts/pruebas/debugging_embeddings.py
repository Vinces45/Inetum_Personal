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
            
        print("PONER -1 SI NO QUIERES FILTRO")
        tipo_doc = input("¿Filtrar por tipo? (Escribe PCAP, PPT, o -1): ").strip().upper()
        
        # ERROR CORREGIDO 1: Si pulsas Enter vacio al pedir un int(), el programa crashea. 
        # Hay que capturar el string, comprobar si esta vacio, y convertirlo.
        def pedir_entero(mensaje):
            val = input(mensaje).strip()
            if not val or val == "-1": return -1
            try: return int(val)
            except: return -1

        presupuesto_base = pedir_entero("Inserte presupuesto base: ")
        valor_estimado = pedir_entero("Inserte valor estimado total: ")
        iva = pedir_entero("Inserte valor de IVA: ")
        plazo_meses = pedir_entero("Inserte numero de meses de plazo ordinario: ")
        prorroga = pedir_entero("Inserte numero de meses de prorroga: ")
        
        tramitacion = input("Inserte tipo de tramitacion (Abierto Simplificado, Negociado sin publicidad, o -1): ").strip()
        
        lotes_input = input("Inserte si tiene lotes o no (S/N o -1): ").strip().upper()
        # ERROR CORREGIDO 2: Logica condicional segura para booleanos
        lotes_bool = None
        if lotes_input == "S": lotes_bool = True
        elif lotes_input == "N": lotes_bool = False

        financiacion_europea_input = input("Inserte si tiene financiacion europea o no (S/N o -1): ").strip().upper()
        financiacion_europea_bool = None
        if financiacion_europea_input == "S": financiacion_europea_bool = True
        elif financiacion_europea_input == "N": financiacion_europea_bool = False
            
        tipo_contrato = input("¿Filtrar por contrato? (Obras, Servicios, Suministros, o -1): ").strip().capitalize()
        
        # Construimos un filtro complejo dinamicamente
        filtros_lista = []
        
        if tipo_doc in ["PCAP", "PPT"]:
            filtros_lista.append({"tipo_documento": tipo_doc})
            
        if tipo_contrato in ["Obras", "Servicios", "Suministros"]:
            filtros_lista.append({"tipo_contrato": tipo_contrato})

        # CUIDADO AQUI: En tu script de metadatos la clave es "presupuesto_base_licitacion", no "presupuesto_base".
        # Debes usar los nombres exactos que usaste en preprocesamiento.
        if presupuesto_base != -1:
            filtros_lista.append({"presupuesto_base_licitacion": presupuesto_base})
            
        if valor_estimado != -1:
            filtros_lista.append({"valor_estimado_contrato": valor_estimado})

        if iva != -1:
            filtros_lista.append({"iva_porcentaje": iva})
        
        if plazo_meses != -1:
            filtros_lista.append({"plazo_ejecucion_meses": plazo_meses})

        if prorroga != -1:
            filtros_lista.append({"tiempo_prorroga_meses": prorroga})

        if tramitacion != "-1" and tramitacion != "":
            filtros_lista.append({"tramitacion": tramitacion})

        if lotes_bool is not None:
            filtros_lista.append({"lotes": lotes_bool})

        if financiacion_europea_bool is not None:
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
                print(f"ARCHIVO: {doc.metadata.get('fuente', 'Desconocido')} (Pag. {doc.metadata.get('pagina', 'N/A')})")
                
                # CUIDADO: Ajustado para que coincida con las claves de tu extraccion
                meta_resumen = {
                    "exp": doc.metadata.get("expediente"),
                    "tipo": doc.metadata.get("tipo_contrato"),
                    "presupuesto": doc.metadata.get("presupuesto_base_licitacion")
                }
                print(f"METADATOS: {meta_resumen}")
                
                contenido = doc.page_content.replace('\n', ' ').strip()
                print(f"TEXTO: {contenido[:300]}...\n")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error en la busqueda: {e}")

if __name__ == "__main__":
    realizar_test()