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

from sentence_transformers import CrossEncoder

from scripts.preprocesamiento.chunking_embeddings import cargar_diccionario_metadatos, cargar_y_procesar_documentos, configurar_chunkeador
from scripts.pruebas.chunking_embeddings_markdown import procesar_documentos_y_capa_semantica as procesar_v2
from scripts.pruebas.chunking_embeddings_markdown import configurar_splitters as chunkeador_v2

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DIR_DB_TEST = PROJECT_ROOT / "datos" / "bd_test_temp"

RUTA_JSON_VAL = PROJECT_ROOT / "datos" / "jsons" / "MRR" / "MRR_validacion_nuevo.json"
RUTA_JSON_TEST = PROJECT_ROOT / "datos" / "jsons" / "MRR" / "MRR_test_nuevo.json"

DIR_PDFS_PCAP = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
DIR_PDFS_PPT = PROJECT_ROOT / "datos" / "pdfs" / "ppt"
DIR_JSON_META = PROJECT_ROOT / "datos" / "jsons" / "metadatos.jsonl"

PARAMETROS = {
    "chunk_size": [800, 1500, 2000],
    "chunk_overlap": [100, 200, 400],
    "hnsw_space": ["l2", "cosine"],
    "estrategia": ["regex", "markdown"]
}

def cargar_preguntas(ruta):
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluar_configuracion(docs_regex, docs_md, size, overlap, space, estrategia, ground_truth, modelo_embeddings, modelo_reranker, nombre_coleccion="test_grid"):
    print(f"\nProbando -> Estrategia: {estrategia} | Size: {size} | Overlap: {overlap} | Space: {space}")
    
    if estrategia == "regex":
        chunky = configurar_chunkeador(size, overlap)
        chunks = chunky.split_documents(docs_regex)

    elif estrategia == "markdown":
        _, rec_splitter = chunkeador_v2(size, overlap)
        chunks = rec_splitter.split_documents(docs_md) 

    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")
        
    nombre_unico = f"{nombre_coleccion}_{uuid.uuid4().hex[:8]}"
    
    vector_db = Chroma(
        embedding_function=modelo_embeddings,
        collection_name=nombre_unico,
        collection_metadata={"hnsw:space": space}
    )
    
    lote = 50
    for i in range(0, len(chunks), lote):
        lote_actual = chunks[i:i+lote]
        try:
            vector_db.add_documents(lote_actual)
        except Exception as e:
            for chunk_ind in lote_actual:
                try:
                    vector_db.add_documents([chunk_ind])
                except Exception as ex:
                    doc_id = chunk_ind.metadata.get("doc_id", "Desconocido")
                    print(f"    [DESCARTADO] Exceso de tokens en chunk del doc_id: {doc_id}")
                    continue
    
    
    suma_rr = 0.0
    k_recuperacion_inicial = 15 
    k_eval_final = 5            
    
    for item in ground_truth:
        consulta = item["pregunta"]
        id_esperado = str(item["doc_id_esperado"])
        
        resultados_chroma = vector_db.similarity_search_with_score(consulta, k=k_recuperacion_inicial)
        
        pares_evaluacion = []
        for doc, score in resultados_chroma:
            pares_evaluacion.append([consulta, doc.page_content])
            
        puntuaciones_reranker = modelo_reranker.predict(pares_evaluacion)
        
        resultados_reordenados = list(zip(resultados_chroma, puntuaciones_reranker))
        resultados_reordenados.sort(key=lambda x: x[1], reverse=True)
        
        top_final = resultados_reordenados[:k_eval_final]
        
        rr_actual = 0.0
        for posicion, (datos_originales, nueva_puntuacion) in enumerate(top_final):
            doc = datos_originales[0] 
            doc_id_obtenido = str(doc.metadata.get("doc_id", ""))
            
            if doc_id_obtenido == id_esperado:
                rr_actual = 1.0 / (posicion + 1)
                break 
                
        suma_rr += rr_actual
        
    mrr_final = suma_rr / len(ground_truth)

    vector_db.delete_collection()
    return mrr_final

def main_grid_search():
    print("=== INICIANDO GRID SEARCH Y EVALUACION EXHAUSTIVA ===")
    
    if not os.path.exists(RUTA_JSON_VAL) or not os.path.exists(RUTA_JSON_TEST):
        print("Error: Faltan los archivos JSON de validacion o test. Revisa las rutas.")
        return

    preguntas_val = cargar_preguntas(RUTA_JSON_VAL)
    preguntas_test = cargar_preguntas(RUTA_JSON_TEST)
    
    print("Cargando modelo de Embeddings (Ollama)...")
    modelo_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    print("Cargando modelo Cross-Encoder (Re-Ranker)...")
    modelo_reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("Cargando metadatos...")
    mapa_meta = cargar_diccionario_metadatos(DIR_JSON_META)
    
    print("Cargando documentos base (Texto Plano para Regex)...")
    documentos_regex = cargar_y_procesar_documentos(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta) 
    
    print("Cargando documentos base (Semanticos para Markdown)...")
    cabeceras_markdown = [("#", "Clausula"), ("##", "Apartado"), ("###", "Subapartado")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=cabeceras_markdown, strip_headers=False)
    documentos_markdown = procesar_v2(DIR_PDFS_PCAP, DIR_PDFS_PPT, mapa_meta, md_splitter)

    if not documentos_regex or not documentos_markdown:
        print("Error: No se pudieron cargar los PDFs en alguno de los formatos.")
        return
    
    keys, values = zip(*PARAMETROS.items())
    combinaciones = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    resultados_totales = []
    
    print("\n--- FASE 1: EVALUACION EN VALIDACION ---")
    for config in tqdm(combinaciones, desc="Validacion"):
        try:
            mrr_val = evaluar_configuracion(
                docs_regex=documentos_regex,  
                docs_md=documentos_markdown,       
                size=config["chunk_size"], 
                overlap=config["chunk_overlap"], 
                space=config["hnsw_space"], 
                estrategia=config["estrategia"],
                ground_truth=preguntas_val, 
                modelo_embeddings=modelo_embeddings,
                modelo_reranker=modelo_reranker,
                nombre_coleccion="val_grid"
            )
            config["MRR_val"] = mrr_val
            resultados_totales.append(config)
        except Exception as e:
            print(f"Error evaluando la configuracion {config} en validacion: {e}")
            
    print("\n--- FASE 2: EVALUACION EN TEST DE TODAS LAS COMBINACIONES ---")

    for config in tqdm(resultados_totales, desc="Test Ciego"):
        try:
            mrr_test = evaluar_configuracion(
                docs_regex=documentos_regex, 
                docs_md=documentos_markdown,        
                size=config["chunk_size"], 
                overlap=config["chunk_overlap"], 
                space=config["hnsw_space"], 
                estrategia=config["estrategia"], 
                ground_truth=preguntas_test,        
                modelo_embeddings=modelo_embeddings,
                modelo_reranker=modelo_reranker,
                nombre_coleccion="test_grid"
            )
            config["MRR_test"] = mrr_test
        except Exception as e:
            print(f"Error evaluando la configuracion {config} en test: {e}")
            config["MRR_test"] = 0.0 
            
    resultados_totales.sort(key=lambda x: x["MRR_val"], reverse=True)
    
    print("\n" + "="*95)
    print(f"{'ESTRATEGIA':<12} | {'SIZE':<6} | {'OVERLAP':<7} | {'SPACE':<8} || {'MRR VAL':<10} | {'MRR TEST':<10} | {'DIFERENCIA':<10}")
    print("="*95)
    
    for res in resultados_totales:
        dif = res['MRR_val'] - res['MRR_test']
        print(f"{res['estrategia']:<12} | {res['chunk_size']:<6} | {res['chunk_overlap']:<7} | {res['hnsw_space']:<8} || "
              f"{res['MRR_val']:.4f}     | {res['MRR_test']:.4f}     | {'+' if dif < 0 else '-'}{abs(dif):.4f}")
    
    print("="*95)
    
    mejor = resultados_totales[0]
    print(f"\n> GANADOR (Segun Validacion): {mejor['estrategia']}, Size {mejor['chunk_size']}, Overlap {mejor['chunk_overlap']}, {mejor['hnsw_space']}")
    print(f"> Rendimiento esperado (Test): {mejor['MRR_test']:.4f}\n")

if __name__ == "__main__":
    main_grid_search()