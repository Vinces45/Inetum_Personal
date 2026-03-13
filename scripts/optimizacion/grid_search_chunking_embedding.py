import json
import os
import shutil
import itertools
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from scripts.preprocesamiento.chunking_embeddings import cargar_diccionario_metadatos, cargar_y_procesar_documentos

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DIR_DB_TEST = PROJECT_ROOT / "datos" / "bd_test_temp"

# --- MODIFICACION 1: Rutas separadas para validacion y test ---
RUTA_JSON_VAL = PROJECT_ROOT / "datos" / "jsons" / "MRR_validacion.json"
RUTA_JSON_TEST = PROJECT_ROOT / "datos" / "jsons" / "MRR_test.json"

# --- MODIFICACION 2: Corregimos la ruta al JSON de metadatos real ---
DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON_META = PROJECT_ROOT / "datos" / "jsons" / "metadatos.json"

# --- HIPERPARAMETROS A PROBAR (GRID) ---
PARAMETROS = {
  "chunk_size": [800, 1500, 2000],
  "chunk_overlap": [100, 200, 400],
  "hnsw_space": ["l2", "cosine"]
}

def cargar_preguntas(ruta):
  with open(ruta, 'r', encoding='utf-8') as f:
    return json.load(f)

def limpiar_directorio_db(directorio):
  if os.path.exists(directorio):
    shutil.rmtree(directorio)
  os.makedirs(directorio)

def evaluar_configuracion(documentos_base, size, overlap, space, ground_truth, modelo_embeddings, nombre_coleccion="test_grid"):
  print(f"\nProbando configuracion -> Size: {size}, Overlap: {overlap}, Space: {space}")
  limpiar_directorio_db(DIR_DB_TEST)
  
  # 1. Chunking
  chunky = RecursiveCharacterTextSplitter(
    separators=[
      r"\n\d+[\.\-]+\s+[A-Z]", r"\n\d+\.\d+\.\s+[A-Z]", r"\n[A-Z]\.\s+[A-Z]",
      r"\nART.CULO", r"\nArt.culo", r"\nCL.USULA", r"\nCl.usula", 
      r"\nANEXO", r"\nAnexo", r"\n\n", r"\n-\s", r"\n[a-z]\)\s", r"\.\s", r" "
    ],
    chunk_size=size,
    chunk_overlap=overlap,
    length_function=len,
    is_separator_regex=True
  )
  
  chunks = chunky.split_documents(documentos_base)
  
  # 2. Vector DB
  vector_db = Chroma(
    embedding_function=modelo_embeddings,
    persist_directory=str(DIR_DB_TEST),
    collection_name=nombre_coleccion,
    collection_metadata={"hnsw:space": space}
  )
  
  # Ingesta por lotes
  lote = 50
  for i in range(0, len(chunks), lote):
    vector_db.add_documents(chunks[i:i+lote])
    
  # 3. Evaluacion MRR
  suma_rr = 0.0
  k_eval = 5 # Miramos en el top 5
  
  for item in ground_truth:
    consulta = item["pregunta"]
    id_esperado = str(item["doc_id_esperado"])
    pag_esperada = int(item["pagina_esperada"])
    
    resultados = vector_db.similarity_search_with_score(consulta, k=k_eval)
    
    rr_actual = 0.0
    for posicion, (doc, score) in enumerate(resultados):
      doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
      pag_obtenida = doc.metadata.get("pagina", -1)
      
      if doc_id_obtenido == id_esperado and (pag_obtenida == pag_esperada):
        rr_actual = 1.0 / (posicion + 1)
        break 
        
    suma_rr += rr_actual
    
  mrr_final = suma_rr / len(ground_truth)
  return mrr_final

def main_grid_search():
  print("=== INICIANDO GRID SEARCH Y EVALUACION FINAL ===")
  
  if not os.path.exists(RUTA_JSON_VAL) or not os.path.exists(RUTA_JSON_TEST):
      print("Error: Faltan los archivos JSON de validacion o test. Revisa las rutas.")
      return

  preguntas_val = cargar_preguntas(RUTA_JSON_VAL)
  preguntas_test = cargar_preguntas(RUTA_JSON_TEST)
  modelo_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
  
  print("Cargando metadatos y documentos base...")
  mapa_meta = cargar_diccionario_metadatos(DIR_JSON_META)
  documentos_base = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta) 
  
  if not documentos_base:
    print("No se encontraron PDFs.")
    return 
  
  keys, values = zip(*PARAMETROS.items())
  combinaciones = [dict(zip(keys, v)) for v in itertools.product(*values)]
  
  resultados_mrr = []
  
  print("\n--- FASE 1: GRID SEARCH (VALIDACION) ---")
  for config in tqdm(combinaciones, desc="Evaluando combinaciones"):
    try:
      mrr = evaluar_configuracion(
        documentos_base, 
        config["chunk_size"], 
        config["chunk_overlap"], 
        config["hnsw_space"], 
        preguntas_val, 
        modelo_embeddings,
        nombre_coleccion="val_grid"
      )
      config["MRR_val"] = mrr
      resultados_mrr.append(config)
    except Exception as e:
      print(f"Error evaluando la configuracion {config}: {e}")
      
  # Ordenar por el mejor MRR en validacion
  resultados_mrr.sort(key=lambda x: x["MRR_val"], reverse=True)
  
  print("\n" + "="*50)
  print("RESULTADOS FASE 1: VALIDACION")
  print("="*50)
  for i, res in enumerate(resultados_mrr):
    print(f"#{i+1} | MRR Val: {res['MRR_val']:.4f} | Size: {res['chunk_size']}, Overlap: {res['chunk_overlap']}, Space: {res['hnsw_space']}")

  # --- FASE 2: EVALUACION EN TEST ---
  mejor_config = resultados_mrr[0]
  
  print("\n" + "="*50)
  print("--- FASE 2: EVALUACION FINAL (TEST CIEGO) ---")
  print(f"Evaluando la mejor configuracion encontrada:")
  print(f"Size: {mejor_config['chunk_size']} | Overlap: {mejor_config['chunk_overlap']} | Space: {mejor_config['hnsw_space']}")
  print("="*50)
  
  mrr_test = evaluar_configuracion(
        documentos_base, 
        mejor_config["chunk_size"], 
        mejor_config["chunk_overlap"], 
        mejor_config["hnsw_space"], 
        preguntas_test, 
        modelo_embeddings,
        nombre_coleccion="test_final"
      )
      
  print(f"\n>>>> MRR FINAL EN TEST: {mrr_test:.4f} <<<<\n")

if __name__ == "__main__":
  main_grid_search()