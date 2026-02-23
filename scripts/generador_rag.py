import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURACION DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent 
DIR_DB = PROJECT_ROOT / "datos" / "base_datos_vectorial"

def inicializar_bd():
    """Inicializa la conexion a la base de datos vectorial una sola vez."""
    print("[SISTEMA] Conectando a ChromaDB...")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return Chroma(
        persist_directory=str(DIR_DB),
        embedding_function=embeddings,
        collection_name="pliegos_oficiales"
    )

def inicializar_llm():
    """Inicializa el modelo de lenguaje Llama 3."""
    print("[SISTEMA] Conectando a Llama 3...")
    return ChatOllama(model="llama3", temperature=0.1)

def generar_borrador_seccion(vector_db, llm, peticion_usuario, tipo_doc=None):
    """
    Ejecuta la cadena RAG con un filtro dinamico.
    Esta es la funcion pura que llamara tu Orquestador en el futuro.
    """
    # 1. Configurar los argumentos de busqueda (Filtro Hibrido)
    search_kwargs = {"k": 3}
    if tipo_doc in ["PCAP", "PPT"]:
        search_kwargs["filter"] = {"tipo_documento": tipo_doc}
        print(f"\n[INFO] Aplicando filtro hibrido en base de datos: {search_kwargs['filter']}")
    else:
        print("\n[INFO] Buscando en toda la base de datos (sin filtro).")

    # 2. Crear el Retriever dinamico para esta peticion especifica
    retriever = vector_db.as_retriever(search_kwargs=search_kwargs)

    # 3. Prompt Maestro
    template = """
    Eres un Letrado experto en Contratacion Publica del Gobierno de La Rioja.
    Tu objetivo es redactar un borrador para una seccion de un pliego administrativo basandote UNICAMENTE en el contexto historico proporcionado.
    
    REGLAS CRITICAS:
    1. Usa un tono formal, juridico y propio de la administracion publica espanola.
    2. No inventes leyes, normativas, plazos ni penalizaciones que no aparezcan en el contexto.
    3. Si el contexto no contiene informacion suficiente para responder, indicalo claramente y no inventes texto.

    CONTEXTO RECUPERADO DE PLIEGOS ANTERIORES:
    {contexto}

    PETICION DEL USUARIO:
    {pregunta}

    REDACCION DEL BORRADOR:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Funcion auxiliar de formateo
    def formatear_documentos(docs):
        if not docs:
            return "No se encontro contexto relevante en la base de datos."
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # 5. Construir la Cadena RAG (LCEL)
    cadena_rag = (
        {"contexto": retriever | formatear_documentos, "pregunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 6. Ejecutar y devolver el resultado
    return cadena_rag.invoke(peticion_usuario)

if __name__ == "__main__":
    print("=== INICIANDO SISTEMA DE REDACCION ASISTIDA ===")
    
    # Inicializamos recursos al arrancar (Patron Singleton manual)
    db_vectorial = inicializar_bd()
    modelo_llm = inicializar_llm()
    
    print("\nMotor listo. Escribe 'salir' para terminar el programa.")
    
    while True:
        peticion = input("\n¿Que seccion del pliego necesitas redactar?: ")
        if peticion.lower() == "salir":
            break
            
        # Simulamos lo que hara el Orquestador: decidir el filtro
        filtro = input("¿Filtrar por tipo (PCAP/PPT)? (Pulsa Enter para omitir): ").strip().upper()
        if filtro not in ["PCAP", "PPT"]:
            filtro = None
            
        print("\nGenerando borrador (analizando historico y redactando)...")
        print("-" * 50)
        
        try:
            # Llamada limpia a nuestra funcion pura
            respuesta_final = generar_borrador_seccion(db_vectorial, modelo_llm, peticion, filtro)
            print(respuesta_final)
            print("-" * 50)
        except Exception as e:
            print(f"\nError critico durante la generacion: {e}")