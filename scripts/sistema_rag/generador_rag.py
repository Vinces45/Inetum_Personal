import json
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- NUEVO IMPORT PARA EL RE-RANKER ---
from sentence_transformers import CrossEncoder

from scripts.modelo.pliego import BorradorPliego

# --- CONFIGURACION DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent.parent
DIR_DB = PROJECT_ROOT / "datos" / "base_datos_vectorial"
DIR_RESPALDO = PROJECT_ROOT / "datos" / "borradores_pliego" / "borrador_actual.json"

def inicializar_bd():
    print("[SISTEMA] Conectando a ChromaDB...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma(
        persist_directory=str(DIR_DB),
        embedding_function=embeddings,
        collection_name="pliegos_oficiales"
    )

def inicializar_llm():
    print("[SISTEMA] Conectando a Llama 3...")
    return ChatOllama(model="llama3.1", temperature=0.1)

def inicializar_reranker():
    print("[SISTEMA] Cargando modelo Cross-Encoder (Re-Ranker)...")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def construir_filtros_chroma(filtros, margen_tolerancia=0.0):
    if not filtros:
        return None
        
    condiciones = []
    for clave, valor in filtros.items():
        if isinstance(valor, dict):
            condiciones.append({clave: valor})
            
        elif clave in ["presupuesto_base_licitacion", "valor_estimado_contrato"] and isinstance(valor, (int, float)) and margen_tolerancia > 0:
            margen_inferior = valor * (1.0 - margen_tolerancia)
            margen_superior = valor * (1.0 + margen_tolerancia)
            condiciones.append({
                clave: {"$gte": margen_inferior, "$lte": margen_superior}
            })
            
        else:
            condiciones.append({clave: {"$eq": valor}})
            
    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}

def recuperar_con_reranker(vector_db, modelo_reranker, query, k_inicial=15, k_final=3, filtros=None, tolerancia=0.0):
    search_kwargs = {"k": k_inicial}
    if filtros:
        filtros_procesados = construir_filtros_chroma(filtros, margen_tolerancia=tolerancia)
        if filtros_procesados:
            search_kwargs["filter"] = filtros_procesados

    # Fase 1: Retrieval con Chroma
    docs_brutos = vector_db.similarity_search(query, **search_kwargs)
    
    if not docs_brutos:
        return "Sin contexto."

    # Fase 2: Re-Ranking
    pares_evaluacion = [[query, doc.page_content] for doc in docs_brutos]
    puntuaciones = modelo_reranker.predict(pares_evaluacion)
    
    resultados_reordenados = list(zip(docs_brutos, puntuaciones))
    resultados_reordenados.sort(key=lambda x: x[1], reverse=True)
    
    mejores_docs = [item[0] for item in resultados_reordenados[:k_final]]
    
    # Formateamos directamente el string aqui
    return "\n\n---\n\n".join(doc.page_content for doc in mejores_docs)


def generar_seccion_nueva(vector_db, llm, modelo_reranker, peticion_usuario, filtros=None):
    """Genera una seccion desde cero usando RAG de 2 fases."""
    
    # Obtenemos el texto ya masticado por el Re-Ranker
    contexto_texto = recuperar_con_reranker(
        vector_db=vector_db, 
        modelo_reranker=modelo_reranker, 
        query=peticion_usuario, 
        k_inicial=15, 
        k_final=3, 
        filtros=filtros, 
        tolerancia=0.0
    )

    template = """
    Eres un Letrado experto en Contratacion Publica del Gobierno de La Rioja.
    Redacta la siguiente seccion para un pliego basandote UNICAMENTE en el contexto proporcionado.
    
    CONTEXTO RECUPERADO:
    {contexto}

    SECCION SOLICITADA:
    {pregunta}

    REDACCION DEL BORRADOR:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # La cadena ahora es mucho mas simple de leer y depurar
    cadena_rag = prompt | llm | StrOutputParser()
    
    return cadena_rag.invoke({"contexto": contexto_texto, "pregunta": peticion_usuario})

def corregir_seccion_existente(vector_db, llm, modelo_reranker, texto_actual, feedback_usuario, filtros=None):
    """Reescribe una seccion existente aplicando el feedback del usuario y consultando la BD."""
    
    # Para corregir, la query ideal es la mezcla de lo que hay y lo que se pide
    query_busqueda = f"{texto_actual} {feedback_usuario}"
    
    contexto_texto = recuperar_con_reranker(
        vector_db=vector_db, 
        modelo_reranker=modelo_reranker, 
        query=query_busqueda, 
        k_inicial=15, 
        k_final=2, 
        filtros=filtros, 
        tolerancia=0.2
    )

    template_correccion = """
    Eres un Letrado experto en Contratacion Publica.
    Tienes el siguiente borrador de una seccion de un pliego:
    
    BORRADOR ACTUAL:
    {texto_actual}
    
    El usuario ha solicitado el siguiente CAMBIO o CORRECCION:
    {pregunta}
    
    CONTEXTO RECUPERADO (Por si necesitas consultar normativa para el cambio):
    {contexto}
    
    INSTRUCCIONES:
    1. Reescribe el BORRADOR ACTUAL aplicando el CAMBIO solicitado.
    2. Manten el mismo tono formal.
    3. Devuelve UNICAMENTE la nueva redaccion de la seccion modificada, sin comentarios.
    """
    prompt = ChatPromptTemplate.from_template(template_correccion)

    cadena_correccion = prompt | llm | StrOutputParser()
    
    return cadena_correccion.invoke({
        "contexto": contexto_texto,
        "pregunta": feedback_usuario,
        "texto_actual": texto_actual
    })