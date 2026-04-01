import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain   
from langchain_core.prompts import ChatPromptTemplate

# Importamos tus herramientas (ajusta la ruta)
from scripts.preprocesamiento.chunking_embeddings import configurar_chunkeador, limpiar_texto

def crear_rag_temporal(ruta_pdf_temporal, embeddings_model, llm):
    """
    Lee un PDF y crea una cadena RAG completa usando ChromaDB en RAM.
    """
    print(f"[RAG EFIMERO] Procesando documento temporal: {ruta_pdf_temporal}")
    
    docs_lista = []
    
    try:
        with fitz.open(ruta_pdf_temporal) as pdf_doc:
            for pagina in pdf_doc:
                texto_limpio = limpiar_texto(pagina.get_text("text"))
                if texto_limpio:
                    # Metadatos minimos, solo para saber la pagina
                    meta = {"fuente": "Documento Adjunto", "pagina": pagina.number + 1}
                    docs_lista.append(Document(page_content=texto_limpio, metadata=meta))
                    
    except Exception as e:
        print(f"[ERROR] No se pudo leer el PDF temporal: {e}")
        return None

    # Chunking
    chunky = configurar_chunkeador(600, 100) # Chunks mas pequeños para consultas rapidas
    chunks = chunky.split_documents(docs_lista)
    
    # Creamos Chroma en RAM (no le pasamos persist_directory)
    vector_db_ram = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_model,
        collection_name="coleccion_efimera"
    )
    
    # Creamos el Retriever
    retriever = vector_db_ram.as_retriever(search_kwargs={"k": 5})
    
    # Creamos un Prompt especializado para este RAG
    prompt_qa = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente util. Responde a la pregunta del usuario basandote UNICAMENTE en el contexto proporcionado del documento adjunto. Si la respuesta no esta en el contexto, di que no lo sabes.\n\nContexto:\n{context}"),
        ("human", "{input}")
    ])
    
    # Montamos la cadena completa
    cadena_qa = create_stuff_documents_chain(llm, prompt_qa)
    cadena_rag = create_retrieval_chain(retriever, cadena_qa)
    
    print("[RAG EFIMERO] Cadena temporal lista en RAM.")
    return cadena_rag