import json
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
DIR_RESPALDO = PROJECT_ROOT / "datos" / "respaldo" / "borrador_actual.json"

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
    return ChatOllama(model="llama3", temperature=0.1)

class BorradorPliego:
    def __init__(self, archivo_respaldo=DIR_RESPALDO):
        self.archivo_respaldo = archivo_respaldo
        self.secciones = self.cargar_respaldo()
        
    def actualizar_seccion(self, titulo, contenido):
        self.secciones[titulo] = contenido
        self.guardar_respaldo()
        
    def obtener_seccion(self, titulo):
        return self.secciones.get(titulo, None)
        
    def mostrar_documento(self):
        doc = "\n" + "="*50 + "\nBORRADOR ACTUAL DEL PLIEGO\n" + "="*50 + "\n"
        if not self.secciones:
            return doc + "El documento esta vacio.\n" + "="*50
            
        for titulo, texto in self.secciones.items():
            doc += f"\n--- {titulo.upper()} ---\n{texto}\n"
        return doc + "="*50
    
    def guardar_respaldo(self):
        """Serializa el diccionario y lo guarda en disco de forma segura."""
        try:
            with open(self.archivo_respaldo, 'w', encoding='utf-8') as f:
                json.dump(self.secciones, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR PERSISTENCIA] No se pudo guardar el borrador: {e}")

    def cargar_respaldo(self):
        """Lee el archivo JSON del disco para restaurar el estado en caso de caida."""
        if os.path.exists(self.archivo_respaldo):
            try:
                with open(self.archivo_respaldo, 'r', encoding='utf-8') as f:
                    datos_recuperados = json.load(f)
                    print("[SISTEMA] Borrador anterior recuperado con exito.")
                    return datos_recuperados
            except Exception as e:
                print(f"[ERROR PERSISTENCIA] Archivo corrupto, iniciando vacio: {e}")
        return {}
        
    def limpiar_borrador(self):
        """Borra el estado cuando el pliego se da por finalizado y exportado."""
        self.secciones = {}
        if os.path.exists(self.archivo_respaldo):
            os.remove(self.archivo_respaldo)

def generar_seccion_nueva(vector_db, llm, peticion_usuario, filtros=None):
    """Genera una seccion desde cero usando RAG estandar."""
    search_kwargs = {"k": 3}
    
    if filtros:
        condiciones = []
        for clave, valor in filtros.items():
            # Si el valor ya trae operadores de Chroma (ej: {"$gte": 100000})
            if isinstance(valor, dict):
                condiciones.append({clave: valor})
            # Si es un valor directo (ej: True, "PCAP", 50), aplicamos igualdad exacta
            else:
                condiciones.append({clave: {"$eq": valor}})
                
        if len(condiciones) == 1:
            search_kwargs["filter"] = condiciones[0]
        else:
            search_kwargs["filter"] = {"$and": condiciones}


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
        condiciones = []
        for clave, valor in filtros.items():
            # Si el valor ya trae operadores de Chroma (ej: {"$gte": 100000})
            if isinstance(valor, dict):
                condiciones.append({clave: valor})
            # Si es un valor directo (ej: True, "PCAP", 50), aplicamos igualdad exacta
            else:
                condiciones.append({clave: {"$eq": valor}})
                
        if len(condiciones) == 1:
            search_kwargs["filter"] = condiciones[0]
        else:
            search_kwargs["filter"] = {"$and": condiciones}

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

    # Inyectamos el texto_actual en el pipeline dinamicamente
    cadena_correccion = (
        {
            "contexto": retriever | formatear_documentos, 
            "pregunta": RunnablePassthrough(),
            "texto_actual": lambda x: texto_actual
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena_correccion.invoke(feedback_usuario)

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