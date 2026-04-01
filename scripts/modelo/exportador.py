from docx import Document
from docx.shared import Pt
import io

def generar_bytes_word(diccionario_secciones):
    """
    Genera un documento Word en memoria y devuelve los bytes 
    para que Streamlit pueda descargarlo.
    """
    doc = Document()
    
    titulo_doc = doc.add_heading('PLIEGO DE PRESCRIPCIONES TECNICAS', 0)
    titulo_doc.alignment = 1 
    
    def procesar_nodos(nodos, nivel=1):
        for titulo, nodo in nodos.items():
            nivel_word = min(nivel, 9) 
            doc.add_heading(titulo.upper(), level=nivel_word)
            
            if nodo.contenido:
                parrafo = doc.add_paragraph(nodo.contenido)
                parrafo.alignment = 3 
                
            if nodo.subsecciones:
                procesar_nodos(nodo.subsecciones, nivel + 1)

    procesar_nodos(diccionario_secciones)
    
    # NUEVO: Guardar en un buffer de memoria en lugar del disco duro
    buffer_memoria = io.BytesIO()
    doc.save(buffer_memoria)
    
    # Mover el puntero del buffer al principio antes de leerlo
    buffer_memoria.seek(0) 
    
    return buffer_memoria