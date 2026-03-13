import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from scripts.preprocesamiento.chunking_embeddings import cargar_y_procesar_documentos as procesar_v1
from scripts.preprocesamiento.chunking_embeddings import configurar_chunkeador as chunkeador_v1

from chunking_embeddings_markdown import procesar_documentos_y_capa_semantica as procesar_v2
from chunking_embeddings_markdown import configurar_splitters as chunkeador_v2

from chunking_embeddings_markdown import cargar_diccionario_metadatos

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON_META = PROJECT_ROOT / "datos" / "jsons" / "metadatos.jsonl"
RUTA_JSON_VALIDACION = PROJECT_ROOT / "datos" / "jsons" / "MRR" / "MRR_validacion_nuevo.json"

def cargar_preguntas(ruta):
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)
#AQUI NO SE EVALUA SI VIENE DE UN PCAP PPT EN GENERAL ESTA FUNCION DA MEJOR RESULTADO QUE LA DE ABAJO
def evaluar_mrr(vector_db, ground_truth, k_eval=5):
    suma_rr = 0.0
    
    for item in ground_truth:
        consulta = item["pregunta"]
        id_esperado = str(item["doc_id_esperado"])
        
        resultados = vector_db.similarity_search_with_score(consulta, k=k_eval)
        
        rr_actual = 0.0
        for posicion, (doc, score) in enumerate(resultados):
            doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
            
            # Comprobamos si el documento recuperado es el correcto
            if doc_id_obtenido == id_esperado:
                rr_actual = 1.0 / (posicion + 1)
                break 
                
        suma_rr += rr_actual
        
    return suma_rr / len(ground_truth)
# LA SEGUNDA FUNCIÓN ES LA DE ABAJO
# def evaluar_mrr(vector_db, ground_truth, k_eval=5):
#     suma_rr = 0.0
    
#     for item in ground_truth:
#         consulta = item["pregunta"]
#         id_esperado = str(item["doc_id_esperado"])
#         tipo_esperado = str(item.get("tipo_esperado", ""))
        
#         # Omitimos la comprobacion estricta de pagina para no penalizar 
#         # a los chunks semanticos que abarcan multiples paginas
        
#         resultados = vector_db.similarity_search_with_score(consulta, k=k_eval)
        
#         rr_actual = 0.0
#         for posicion, (doc, score) in enumerate(resultados):
#             doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
#             tipo_documento_obtenido = str(doc.metadata.get("tipo_documento", ""))
            
#             # Evaluamos si ha recuperado el documento y el pliego correctos
#             if (doc_id_obtenido == id_esperado and 
#                 tipo_documento_obtenido == tipo_esperado):
                
#                 rr_actual = 1.0 / (posicion + 1)
#                 break 
                
#         suma_rr += rr_actual
        
#     return suma_rr / len(ground_truth)

def main_ab_testing():
    print("=== INICIANDO A/B TESTING DE ESTRATEGIAS DE CHUNKING ===")
    
    preguntas_test = cargar_preguntas(RUTA_JSON_VALIDACION)
    modelo_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    mapa_meta = cargar_diccionario_metadatos(DIR_JSON_META)
    
    # ---------------------------------------------------------
    # ESTRATEGIA A: Script 1 (Chunking Basico / Regex)
    # ---------------------------------------------------------
    print("\n[Construyendo DB - Estrategia A: Chunking Basico]")
    docs_base_v1 = procesar_v1(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta)
    splitter_v1 = chunkeador_v1()
    chunks_v1 = splitter_v1.split_documents(docs_base_v1)
    
    # Usamos DB en memoria RAM para que sea ultra rapido
    db_v1 = Chroma(
        embedding_function=modelo_embeddings,
        collection_name="coleccion_v1"
    )
    # Añadimos en lotes por si son muchos
    for i in tqdm(range(0, len(chunks_v1), 50), desc="Ingestando V1"):
        db_v1.add_documents(chunks_v1[i:i+50])
        
    mrr_v1 = evaluar_mrr(db_v1, preguntas_test)
    print(f"-> MRR Estrategia A: {mrr_v1:.4f}")

    # ---------------------------------------------------------
    # ESTRATEGIA B: Script 2 (Markdown + Inyeccion de Contexto)
    # ---------------------------------------------------------
    print("\n[Construyendo DB - Estrategia B: Markdown Semantico]")
    md_splitter, rec_splitter = chunkeador_v2()
    # Aqui se aplica la conversion a markdown y el corte de Capa 1
    docs_semanticos_v2 = procesar_v2(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta, md_splitter)
    # Corte de Capa 2 (Respaldo)
    chunks_v2 = rec_splitter.split_documents(docs_semanticos_v2)
    
    db_v2 = Chroma(
        embedding_function=modelo_embeddings,
        collection_name="coleccion_v2"
    )
    for i in tqdm(range(0, len(chunks_v2), 50), desc="Ingestando V2"):
        db_v2.add_documents(chunks_v2[i:i+50])
        # Añade esto justo despues de: db_v2.add_documents(chunks_v2[i:i+50])
        print("\n--- DEPURACION ESTRATEGIA B ---")
        muestra_chunk = chunks_v2[0]
        print(f"Contenido (primeros 100 chars): {muestra_chunk.page_content[:100]}...")
        print(f"Metadatos guardados en V2: {muestra_chunk.metadata}")
        print("-------------------------------\n")
        
    mrr_v2 = evaluar_mrr(db_v2, preguntas_test)
    print(f"-> MRR Estrategia B: {mrr_v2:.4f}")

    # ---------------------------------------------------------
    # RESULTADOS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("RESULTADO FINAL DEL A/B TESTING")
    print("="*50)
    print(f"Estrategia A (Basica):    MRR = {mrr_v1:.4f}")
    print(f"Estrategia B (Markdown):  MRR = {mrr_v2:.4f}")
    
    mejora = ((mrr_v2 - mrr_v1) / mrr_v1) * 100 if mrr_v1 > 0 else 0
    print(f"\nLa Estrategia B mejora a la A en un {mejora:.2f}%")

if __name__ == "__main__":
    main_ab_testing()