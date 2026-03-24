import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Importamos tus clases y funciones
from scripts.modelo.pliego import BorradorPliego
from scripts.sistema_rag.generador_rag import (
    inicializar_bd, inicializar_reranker, consultar_duda_legal, 
    generar_seccion_nueva, corregir_seccion_existente
)
from scripts.modelo.exportador import exportar_a_word

# ==========================================
# 1. DEFINICION DE HERRAMIENTAS (TOOLS)
# ==========================================

@tool
def herramienta_ver_rutas() -> str:
    """
    Devuelve las rutas de las secciones que existen actualmente en el borrador. 
    Usala SIEMPRE antes de modificar o eliminar para saber la ruta exacta.
    """
    rutas = st.session_state.documento.obtener_rutas_secciones()
    return f"Secciones actuales: {rutas}" if rutas else "El documento esta vacio."

@tool
def herramienta_consultar_ley(pregunta: str) -> str:
    """Busca informacion legal en la base de datos RAG para responder dudas del usuario."""
    return consultar_duda_legal(
        st.session_state.db_vectorial, 
        st.session_state.llm, 
        st.session_state.reranker, 
        pregunta
    )

@tool
def herramienta_crear_seccion(titulo: str, instrucciones: str) -> str:
    """
    Redacta una seccion principal nueva usando RAG y la guarda en el borrador.
    """
    borrador = generar_seccion_nueva(
        st.session_state.db_vectorial, 
        st.session_state.llm, 
        st.session_state.reranker, 
        f"Redacta {titulo}. {instrucciones}"
    )
    st.session_state.documento.actualizar_seccion_infinita([titulo], borrador)
    return f"Seccion '{titulo}' creada y guardada con exito."

@tool
def herramienta_eliminar_seccion(ruta_exacta: str) -> str:
    """Elimina una seccion usando su ruta exacta (ej: '1. Objeto > 1.1. Garantia')."""
    exito = st.session_state.documento.eliminar_por_ruta(ruta_exacta)
    return "Eliminada correctamente." if exito else "No se encontro la ruta."

@tool
def herramienta_exportar() -> str:
    """Exporta el pliego a un documento Word (DOCX)."""
    ruta = "datos/pliego_final.docx"
    exportar_a_word(st.session_state.documento.secciones, ruta)
    return f"Documento exportado en {ruta}"

# Lista de herramientas disponibles para el agente
tools = [
    herramienta_ver_rutas, 
    herramienta_consultar_ley, 
    herramienta_crear_seccion, 
    herramienta_eliminar_seccion,
    herramienta_exportar
]

# ==========================================
# 2. INICIALIZACION DEL SISTEMA (SE EJECUTA UNA VEZ)
# ==========================================
st.set_page_config(page_title="Asistente de Pliegos", layout="wide")

if "inicializado" not in st.session_state:
    with st.spinner("Arrancando el sistema..."):
        st.session_state.db_vectorial = inicializar_bd()
        st.session_state.reranker = inicializar_reranker()
        
        st.session_state.llm = ChatOllama(model="llama3.1", temperature=0.0)
        st.session_state.documento = BorradorPliego()
        st.session_state.chat_history = []
        
        # Configuracion del Agente de LangChain
        prompt_agente = ChatPromptTemplate.from_messages([
            ("system", "Eres un Letrado Inteligente del Gobierno de La Rioja. Ayudas a redactar pliegos. Tienes herramientas para consultar la ley, y crear/modificar el documento. Si el usuario saluda, responde amablemente sin usar herramientas. Si te pide varias cosas (ej: crea esto y expórtalo), usa las herramientas en orden."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agente = create_tool_calling_agent(st.session_state.llm, tools, prompt_agente)
        st.session_state.agent_executor = AgentExecutor(agent=agente, tools=tools, verbose=True)
        st.session_state.inicializado = True

# ==========================================
# 3. INTERFAZ GRAFICA (UI)
# ==========================================

# Barra lateral para ver el documento en tiempo real
with st.sidebar:
    st.header("Borrador Actual")
    st.text_area("Vista previa", st.session_state.documento.mostrar_documento(), height=600)

st.title("🏛️ Asistente de Pliegos - La Rioja")

# Mostrar historial de chat
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Caja de texto para el usuario
if prompt_usuario := st.chat_input("Escribe tu peticion (ej: Hola, redacta la seccion de objeto del contrato)"):
    
    # 1. Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt_usuario)
    
    # 2. Añadir a la memoria
    st.session_state.chat_history.append(HumanMessage(content=prompt_usuario))
    
    # 3. Ejecutar el Agente
    with st.chat_message("assistant"):
        with st.spinner("Pensando y ejecutando herramientas..."):
            respuesta = st.session_state.agent_executor.invoke({
                "input": prompt_usuario,
                "chat_history": st.session_state.chat_history
            })
            
            texto_respuesta = respuesta["output"]
            st.markdown(texto_respuesta)
            
    # 4. Guardar respuesta en memoria
    st.session_state.chat_history.append(AIMessage(content=texto_respuesta))
    st.rerun() # Fuerza a recargar la UI para actualizar la barra lateral