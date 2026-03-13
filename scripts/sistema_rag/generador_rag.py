import json
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from scripts.modelo.pliego import BorradorPliego

# --- CONFIGURACION DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 
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

def construir_filtros_chroma(filtros, margen_tolerancia=0.0):
    
    if not filtros:
        return None
        
    condiciones = []
    for clave, valor in filtros.items():
        if isinstance(valor, dict):
            condiciones.append({clave: valor})
            
        # Aqui usamos la variable parametrizada en lugar de numeros magicos
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



def generar_seccion_nueva(vector_db, llm, peticion_usuario, filtros=None):
    """Genera una seccion desde cero usando RAG estandar."""
    search_kwargs = {"k": 3}

    if filtros:
        filtros_procesados = construir_filtros_chroma(filtros, margen_tolerancia=0.0)
        if filtros_procesados:
            search_kwargs["filter"] = filtros_procesados

    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

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

    def formatear_documentos(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs) if docs else "Sin contexto."

    cadena_rag = (
        {"contexto": retriever | formatear_documentos, "pregunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena_rag.invoke(peticion_usuario)

def corregir_seccion_existente(vector_db, llm, texto_actual, feedback_usuario, filtros=None):
    """Reescribe una seccion existente aplicando el feedback del usuario y consultando la BD."""
    
    search_kwargs = {"k": 2}
    if filtros:
        filtros_procesados = construir_filtros_chroma(filtros, margen_tolerancia=0.2)
        if filtros_procesados:
            search_kwargs["filter"] = filtros_procesados
        
    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

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

    def formatear_documentos(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs) if docs else "Sin contexto extra."

    cadena_correccion = (
        {
            # AHORA SI: Busca usando la combinacion del texto base y lo que se pide
            "contexto": lambda x: formatear_documentos(retriever.invoke(f"{x['texto_actual']} {x['pregunta']}")), 
            
            # La pregunta y el texto siguen yendo a sus huecos del prompt igual que antes
            "pregunta": lambda x: x["pregunta"],
            "texto_actual": lambda x: x["texto_actual"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return cadena_correccion.invoke({
        "pregunta": feedback_usuario,
        "texto_actual": texto_actual
    })

if __name__ == "__main__":
    print("=== INICIANDO SISTEMA RAG ITERATIVO ===")
    
    db_vectorial = inicializar_bd()
    modelo_llm = inicializar_llm()
    documento = BorradorPliego()
    
    print("\nMotor listo. Escribe 'salir' para terminar el programa.")
    
    # Ejemplo de flujo simulando el orquestador
    while True:
        print(documento.mostrar_documento())
        
        accion = input("\nElige accion (1: Nueva Seccion, 2: Corregir Seccion, salir): ").strip()
        if accion.lower() == "salir": break
            
        if accion == "1":
            titulo = input("Nombre de la seccion (ej. 'Penalidades'): ")
            peticion = input(f"Instrucciones para generar '{titulo}': ")
            
            print("\nGenerando borrador inicial...")
            texto_generado = generar_seccion_nueva(db_vectorial, modelo_llm, peticion, {"tipo_documento": "PCAP"})
            documento.actualizar_seccion(titulo, texto_generado)
            
        elif accion == "2":
            titulo = input("Nombre de la seccion a corregir: ")
            texto_actual = documento.obtener_seccion(titulo)
            
            if not texto_actual:
                print("Esa seccion no existe en el borrador.")
                continue
                
            feedback = input("¿Que cambio quieres aplicar?: ")
            print("\nAplicando correccion...")
            texto_corregido = corregir_seccion_existente(db_vectorial, modelo_llm, texto_actual, feedback, {"tipo_documento": "PCAP"})
            documento.actualizar_seccion(titulo, texto_corregido)