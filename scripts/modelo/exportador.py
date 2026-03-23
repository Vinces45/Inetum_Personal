from docx import Document
from docx.shared import Pt
import os

def exportar_a_word(diccionario_secciones, ruta_salida="pliego_generado.docx"):

    print(f"\n[SISTEMA] Generando documento Word en: {ruta_salida}")
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
    
    try:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
        doc.save(ruta_salida)
        return True
    except Exception as e:
        print(f"[ERROR EXPORTACION]: {e}")
        return False