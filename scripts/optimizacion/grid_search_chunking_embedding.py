import json
import os
import shutil
import itertools
from pathlib import Path
from tqdm import tqdm
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from scripts.preprocesamiento.chunking_embeddings import cargar_diccionario_metadatos, cargar_y_procesar_documentos, configurar_chunkeador

from scripts.pruebas.chunking_embeddings_markdown import procesar_documentos_y_capa_semantica as procesar_v2
from scripts.pruebas.chunking_embeddings_markdown import configurar_splitters as chunkeador_v2

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DIR_DB_TEST = PROJECT_ROOT / "datos" / "bd_test_temp"

# --- MODIFICACION 1: Rutas separadas para validacion y test ---
RUTA_JSON_VAL = PROJECT_ROOT / "datos" / "jsons" / "MRR" / "MRR_validacion_nuevo.json"
RUTA_JSON_TEST = PROJECT_ROOT / "datos" / "jsons" / "MRR" / "MRR_test_nuevo.json"

# --- MODIFICACION 2: Corregimos la ruta al JSON de metadatos real ---
DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON_META = PROJECT_ROOT / "datos" / "jsons" / "metadatos.jsonl"

# --- HIPERPARAMETROS A PROBAR (GRID) ---
PARAMETROS = {
  "chunk_size": [800, 1500, 2000],
  "chunk_overlap": [100, 200, 400],
  "hnsw_space": ["l2", "cosine"],
  "estrategia": ["regex", "markdown"]
}

def cargar_preguntas(ruta):
  with open(ruta, 'r', encoding='utf-8') as f:
    return json.load(f)

def evaluar_configuracion(docs_regex, docs_md, size, overlap, space, estrategia, ground_truth, modelo_embeddings, nombre_coleccion="test_grid"):
  print(f"\nProbando -> Estrategia: {estrategia} | Size: {size} | Overlap: {overlap} | Space: {space}")
  
  if estrategia == "regex":
    chunky = configurar_chunkeador(size, overlap)
    chunks = chunky.split_documents(docs_regex) # Usa los documentos planos

  elif estrategia == "markdown":
    _, rec_splitter = chunkeador_v2(size, overlap)
    # Ya no llamamos a procesar_v2. Simplemente aplicamos el corte de tamaño a los docs que ya estan en Markdown
    chunks = rec_splitter.split_documents(docs_md) 

  else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")
    
  # 2. Vector DB (EN MEMORIA RAM)
  # Generamos un sufijo aleatorio para evitar conflictos de nombres de colecciones entre iteraciones
  nombre_unico = f"{nombre_coleccion}_{uuid.uuid4().hex[:8]}"
  
  vector_db = Chroma(
    embedding_function=modelo_embeddings,
    collection_name=nombre_unico,
    collection_metadata={"hnsw:space": space}
  )
  
  # --- LA MALLA DE SEGURIDAD ---
  # Ingesta por lotes con respaldo individual
  lote = 50
  for i in range(0, len(chunks), lote):
    lote_actual = chunks[i:i+lote]
    try:
      # Intento 1: Todo el lote de golpe (mas rapido)
      vector_db.add_documents(lote_actual)
          
    except Exception as e:
      # Intento 2: Si el lote falla, insertamos uno a uno
      for chunk_ind in lote_actual:
        try:
          vector_db.add_documents([chunk_ind])
        except Exception as ex:
        # Intento 3: Si un chunk falla individualmente (suele ser por error 400 de Ollama), lo descartamos
          doc_id = chunk_ind.metadata.get("doc_id", "Desconocido")
          print(f"    [DESCARTADO] Exceso de tokens en chunk del doc_id: {doc_id}")
          continue
  
  # # 3. Evaluacion MRR MIRA EL ID Y LA PAGINA ES MAS REESTRICTIVA QUE 
  # suma_rr = 0.0
  # k_eval = 5 # Miramos en el top 5
  
  # for item in ground_truth:
  #   consulta = item["pregunta"]
  #   id_esperado = str(item["doc_id_esperado"])
  #   pag_esperada = int(item["pagina_esperada"])
    
  #   resultados = vector_db.similarity_search_with_score(consulta, k=k_eval)
    
  #   rr_actual = 0.0
  #   for posicion, (doc, score) in enumerate(resultados):
  #     doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
  #     pag_obtenida = doc.metadata.get("pagina", -1)
      
  #     if doc_id_obtenido == id_esperado and (pag_obtenida == pag_esperada):
  #       rr_actual = 1.0 / (posicion + 1)
  #       break 
        
  #   suma_rr += rr_actual
    
  # mrr_final = suma_rr / len(ground_truth)
  # return mrr_final
  #3. Evaluacion MRR
  suma_rr = 0.0
  k_eval = 5 # Miramos en el top 5
  
  for item in ground_truth:
    consulta = item["pregunta"]
    id_esperado = str(item["doc_id_esperado"])
    
    resultados = vector_db.similarity_search_with_score(consulta, k=k_eval)
    
    rr_actual = 0.0
    for posicion, (doc, score) in enumerate(resultados):
      doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
      
      # Evaluamos solo por doc_id, igual que en el script de A/B testing
      if doc_id_obtenido == id_esperado:
        rr_actual = 1.0 / (posicion + 1)
        break 
        
    suma_rr += rr_actual
    
  mrr_final = suma_rr / len(ground_truth)

  vector_db.delete_collection()
  return mrr_final

def main_grid_search():
  print("=== INICIANDO GRID SEARCH Y EVALUACION FINAL ===")
  
  if not os.path.exists(RUTA_JSON_VAL) or not os.path.exists(RUTA_JSON_TEST):
      print("Error: Faltan los archivos JSON de validacion o test. Revisa las rutas.")
      return

  preguntas_val = cargar_preguntas(RUTA_JSON_VAL)
  preguntas_test = cargar_preguntas(RUTA_JSON_TEST)
  modelo_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
  
  print("Cargando metadatos...")
  mapa_meta = cargar_diccionario_metadatos(DIR_JSON_META)
  
  print("Cargando documentos base (Texto Plano para Regex)...")
  documentos_regex = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta) 
  
  print("Cargando documentos base (Semanticos para Markdown)...")
  # Configuramos un md_splitter generico solo para la primera fase de conversion
  cabeceras_markdown = [("#", "Clausula"), ("##", "Apartado"), ("###", "Subapartado")]
  md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=cabeceras_markdown, strip_headers=False)
  
  # Esta funcion se toma su tiempo, pero solo la ejecutamos UNA VEZ
  documentos_markdown = procesar_v2(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta, md_splitter)

  if not documentos_regex or not documentos_markdown:
    print("Error: No se pudieron cargar los PDFs en alguno de los formatos.")
    return
  
  keys, values = zip(*PARAMETROS.items())
  combinaciones = [dict(zip(keys, v)) for v in itertools.product(*values)]
  
  resultados_mrr = []
  
  print("\n--- FASE 1: GRID SEARCH (VALIDACION) ---")
  for config in tqdm(combinaciones, desc="Evaluando combinaciones"):
    try:
      mrr = evaluar_configuracion(
        docs_regex=documentos_regex,  
        docs_md=documentos_markdown,       
        size=config["chunk_size"], 
        overlap=config["chunk_overlap"], 
        space=config["hnsw_space"], 
        estrategia=config["estrategia"],
        ground_truth=preguntas_val, 
        modelo_embeddings=modelo_embeddings,
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
  print("Evaluando la mejor configuracion encontrada:")
  print(f"Estrategia: {mejor_config['estrategia']} | Size: {mejor_config['chunk_size']} | Overlap: {mejor_config['chunk_overlap']} | Space: {mejor_config['hnsw_space']}")
  print("="*50)
  
  mrr_test = evaluar_configuracion(
        docs_regex=documentos_regex, 
        docs_md=documentos_markdown,        
        size=mejor_config["chunk_size"], 
        overlap=mejor_config["chunk_overlap"], 
        space=mejor_config["hnsw_space"], 
        estrategia=mejor_config["estrategia"], 
        ground_truth=preguntas_test,         
        modelo_embeddings=modelo_embeddings,
        nombre_coleccion="test_final"
      )
      
  print(f"\n>>>> MRR FINAL EN TEST: {mrr_test:.4f} <<<<\n")
if __name__ == "__main__":
  main_grid_search()