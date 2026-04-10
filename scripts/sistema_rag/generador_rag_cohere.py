import json
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from scripts.modelo.pliego import BorradorPliego

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

# def inicializar_llm():
#     print("[SISTEMA] Conectando a Llama 3...")
#     return ChatOllama(model="llama3.1", temperature=0.1)

def inicializar_llm():
    
    # 1. Cargamos las variables del .env
    load_dotenv()
    
    # 2. Extraemos las credenciales
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    
    # 3. Validacion de seguridad
    if not all([api_key, endpoint, api_version]):
        raise ValueError("Faltan credenciales de Azure en el archivo .env. Revisa la configuracion.")
        
    # 4. Devolvemos la instancia configurada
    return AzureChatOpenAI(
        azure_deployment="gpt-5.1",
        model_name="gpt-5.1",
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=0,
        max_tokens=16384
    )

def inicializar_reranker():
    print("[SISTEMA] Conectando a la API de Cohere Rerank...")
    api_key = os.environ.get("COHERE_API_KEY")
    
    if not api_key:
        raise ValueError("Falta la clave COHERE_API_KEY en el archivo .env")
        
    return CohereRerank(
        cohere_api_key=api_key, 
        model="rerank-multilingual-v3.0", 
        top_n=3 
    )

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

# def recuperar_con_reranker(vector_db, modelo_reranker, query, k_inicial=15, k_final=3, filtros=None, tolerancia=0.0, castigo_generado=2.0):
#     search_kwargs = {"k": k_inicial}
#     if filtros:
#         filtros_procesados = construir_filtros_chroma(filtros, margen_tolerancia=tolerancia)
#         if filtros_procesados:
#             search_kwargs["filter"] = filtros_procesados

#     # Fase 1: Retrieval con Chroma
#     docs_brutos = vector_db.similarity_search(query, **search_kwargs)
    
#     if not docs_brutos:
#         return "Sin contexto."

#     # Fase 2: Re-Ranking
#     pares_evaluacion = [[query, doc.page_content] for doc in docs_brutos]
#     puntuaciones = modelo_reranker.predict(pares_evaluacion)

#     # Modificación a documentos generados por IA
#     resultados_evaluados = []
#     for doc, nota in zip(docs_brutos, puntuaciones):
#         # Obtenemos el origen (si no lo tiene por ser un doc antiguo, asumimos 'original')
#         origen = doc.metadata.get("origen", "original")
        
#         nota_final = nota
#         # Si fue generado por IA, le restamos los puntos
#         if origen == "generado":
#             nota_final = nota - castigo_generado
            
#         resultados_evaluados.append((doc, nota_final))

    
#     resultados_reordenados = list(zip(docs_brutos, puntuaciones))
#     resultados_reordenados.sort(key=lambda x: x[1], reverse=True)
    
#     mejores_docs = [item[0] for item in resultados_reordenados[:k_final]]
    
#     return "\n\n---\n\n".join(doc.page_content for doc in mejores_docs)

def recuperar_con_reranker(vector_db, modelo_reranker, query, k_inicial=15, filtros=None, tolerancia=0.0):
    
    # 1. Preparamos los parametros para la busqueda en Chroma (Fase 1)
    search_kwargs = {"k": k_inicial}
    if filtros:
        filtros_procesados = construir_filtros_chroma(filtros, margen_tolerancia=tolerancia)
        if filtros_procesados:
            search_kwargs["filter"] = filtros_procesados

    # 2. Convertimos Chroma en un "Retriever" de Langchain
    retriever_base = vector_db.as_retriever(search_kwargs=search_kwargs)
    
    # 3. Ensamblamos el pipeline: Recuperacion + Compresion (Re-Ranking)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=modelo_reranker, 
        base_retriever=retriever_base
    )
    
    # 4. Ejecutamos todo el flujo en una sola llamada
    try:
        mejores_docs = compression_retriever.invoke(query)
    except Exception as e:
        print(f"[ERROR DE RED o API] Fallo al conectar con Cohere: {e}")
        return "Sin contexto."
    
    if not mejores_docs:
        return "Sin contexto."
        
    # 5. Formateamos el texto para incluir los metadatos y que el LLM pueda citar
    textos_formateados = []
    for i, doc in enumerate(mejores_docs):
        # Asegurate de que "source" y "page" coinciden con las claves reales de tu metadata en ChromaDB
        fuente = doc.metadata.get("source", "Documento desconocido")
        pagina = doc.metadata.get("page", "N/A")
        
        texto_doc = f"[FUENTE {i+1}: Archivo '{fuente}', Pagina {pagina}]\n{doc.page_content}"
        textos_formateados.append(texto_doc)
        
    return "\n\n---\n\n".join(textos_formateados)


def generar_seccion_nueva(vector_db, llm, modelo_reranker, peticion_usuario, filtros=None):
    
    contexto_texto = recuperar_con_reranker(
        vector_db=vector_db, 
        modelo_reranker=modelo_reranker, 
        query=peticion_usuario, 
        k_inicial=15,
        filtros=filtros, 
        tolerancia=0.0
    )

    template = """
    Eres un Letrado experto en Contratacion Publica del Gobierno de La Rioja.
    Redacta la siguiente seccion para un pliego basandote UNICAMENTE en el contexto proporcionado.
    
    REGLAS ESTRICTAS DE FORMATO:
    1. NO incluyas saludos, despedidas, ni frases introductorias (ej. "Aqui tienes la seccion solicitada:").
    2. Comienza a escribir directamente el contenido juridico.
    3. Si el CONTEXTO RECUPERADO dice "Sin contexto.", responde UNICAMENTE con la frase: "Falta contexto legal para generar esta seccion."
    
    CONTEXTO RECUPERADO:
    {contexto}

    SECCION SOLICITADA:
    {pregunta}

    REDACCION DEL BORRADOR:
    """
    prompt = ChatPromptTemplate.from_template(template)

    cadena_rag = prompt | llm | StrOutputParser()
    
    return cadena_rag.invoke({"contexto": contexto_texto, "pregunta": peticion_usuario})

def corregir_seccion_existente(vector_db, llm, modelo_reranker, titulo_seccion, texto_actual, feedback_usuario, filtros=None):
    """Reescribe una seccion existente aplicando el feedback del usuario y consultando la BD."""
    
    query_busqueda = f"Seccion {titulo_seccion}: {feedback_usuario}"
    
    contexto_texto = recuperar_con_reranker(
        vector_db=vector_db, 
        modelo_reranker=modelo_reranker, 
        query=query_busqueda, 
        k_inicial=10,
        filtros=filtros, 
        tolerancia=0.2
    )

    template_correccion = """
    Eres un Letrado experto en Contratacion Publica del Gobierno de La Rioja.
    Tu tarea es MODIFICAR la seccion titulada '{titulo}' basandote en las instrucciones del usuario.
    
    [TEXTO ACTUAL DE LA SECCION]
    {texto_actual}
    
    [INSTRUCCION DE CAMBIO DEL USUARIO]
    {pregunta}
    
    [NUEVO CONTEXTO LEGAL RECUPERADO]
    {contexto}
    
    REGLAS DE ACTUACION ESTRICTAS:
    1. Aplica el cambio solicitado al TEXTO ACTUAL.
    2. Si el NUEVO CONTEXTO LEGAL aplica al cambio, integralo con lenguaje juridico formal.
    3. Si la instruccion contradice el texto actual, reemplaza esa parte especifica.
    4. NO agregues introducciones, saludos, ni frases como "Aqui tienes la redaccion" o "Entendido".
    5. Devuelve EXCLUSIVAMENTE el texto final resultante de la seccion modificada.

    NUEVA REDACCION:
    """
    prompt = ChatPromptTemplate.from_template(template_correccion)

    cadena_correccion = prompt | llm | StrOutputParser()
    
    return cadena_correccion.invoke({
        "titulo": titulo_seccion,
        "contexto": contexto_texto,
        "pregunta": feedback_usuario,
        "texto_actual": texto_actual
    })

def recolectar_texto_rama(nodo):
    """Recolecta recursivamente el texto de un nodo y todos sus hijos."""
    texto = nodo.contenido + "\n" if nodo.contenido else ""
    for sub_nodo in nodo.subsecciones.values():
        texto += recolectar_texto_rama(sub_nodo)
    return texto

def resumir_seccion(llm, titulo, texto):
    """Genera un resumen ejecutivo de un fragmento de texto usando LLM."""
    if not texto.strip(): 
        return "Seccion sin contenido redactado."
    
    template = """
    Eres un Letrado Supervisor. Haz un resumen ejecutivo de la siguiente seccion.
    
    REGLAS:
    1. Se muy conciso y usa viñetas.
    2. Destaca solo datos clave (plazos, presupuestos, objetos, penalizaciones).
    3. NO uses frases introductorias (ej. "Aqui tienes el resumen"). Empieza directo.
    
    SECCION: {titulo}
    TEXTO ORIGINAL:
    {texto}
    
    RESUMEN:
    """
    prompt = ChatPromptTemplate.from_template(template)
    cadena = prompt | llm | StrOutputParser()
    
    return cadena.invoke({"titulo": titulo, "texto": texto})

# def consultar_duda_legal(vector_db, llm, modelo_reranker, pregunta, filtros=None):
#     """Responde a una pregunta legal usando la BD vectorial sin modificar el pliego."""
    
#     contexto_texto = recuperar_con_reranker(
#         vector_db=vector_db, 
#         modelo_reranker=modelo_reranker, 
#         query=pregunta, 
#         k_inicial=10, 
#         k_final=3, 
#         filtros=filtros, 
#         tolerancia=0.0
#     )

#     template = """
#     Eres un Letrado Consultor del Gobierno de La Rioja.
#     Responde a la duda legal del usuario basandote UNICAMENTE en el contexto proporcionado.
    
#     CONTEXTO NORMATIVO:
#     {contexto}

#     PREGUNTA DEL USUARIO:
#     {pregunta}

#     REGLAS:
#     1. Responde de forma clara, didactica y directa.
#     2. Si el contexto dice "Sin contexto", responde: "No he encontrado informacion sobre esto en la normativa base."
#     3. Cita el articulo o la ley si aparece en el contexto.
    
#     RESPUESTA LEGAL:
#     """
    
#     prompt = ChatPromptTemplate.from_template(template)
#     cadena = prompt | llm | StrOutputParser()
    
#     return cadena.invoke({"contexto": contexto_texto, "pregunta": pregunta})

def consultar_duda_legal(vector_db, llm, modelo_reranker, pregunta, filtros=None):
    """Responde a una pregunta legal usando la BD vectorial sin modificar el pliego."""
    
    contexto_texto = recuperar_con_reranker(
        vector_db=vector_db, 
        modelo_reranker=modelo_reranker, 
        query=pregunta, 
        k_inicial=10,
        filtros=filtros, 
        tolerancia=0.0
    )

    template = """
    Eres un Letrado Consultor del Gobierno de La Rioja.
    Responde a la duda legal del usuario basandote UNICAMENTE en el contexto proporcionado.
    
    CONTEXTO NORMATIVO:
    {contexto}

    PREGUNTA DEL USUARIO:
    {pregunta}

    REGLAS:
    1. Responde de forma clara, didactica y directa.
    2. Si el contexto dice "Sin contexto", responde: "No he encontrado informacion sobre esto en la normativa base."
    3. CITA OBLIGATORIA: Al final de tu explicacion, o entre parentesis, debes citar la fuente exacta usando las etiquetas [FUENTE X] que aparecen en el texto.
       Ejemplo: "El plazo es de 15 dias (Fuente 1: Archivo 'pliego_condiciones.pdf', Pagina 4)."
    
    RESPUESTA LEGAL:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    cadena = prompt | llm | StrOutputParser()
    
    return cadena.invoke({"contexto": contexto_texto, "pregunta": pregunta})