from docx import Document
from docx.shared import Pt
import os

def exportar_a_word(diccionario_secciones, ruta_salida="pliego_generado.docx"):
    """
    Recorre el arbol de BorradorPliego y genera un documento Word con estilos nativos.
    """
    print(f"\n[SISTEMA] Generando documento Word en: {ruta_salida}")
    doc = Document()
    
    # Titulo principal del documento
    titulo_doc = doc.add_heading('PLIEGO DE PRESCRIPCIONES TECNICAS', 0)
    titulo_doc.alignment = 1 # Centrado
    
    # Funcion recursiva para recorrer el arbol y aplicar estilos de Word
    def procesar_nodos(nodos, nivel=1):
        for titulo, nodo in nodos.items():
            # Añadimos el encabezado. Nivel 1 = Heading 1, Nivel 2 = Heading 2, etc.
            # python-docx soporta hasta 9 niveles de titulos
            nivel_word = min(nivel, 9) 
            doc.add_heading(titulo.upper(), level=nivel_word)
            
            # Añadimos el contenido si lo hay
            if nodo.contenido:
                parrafo = doc.add_paragraph(nodo.contenido)
                parrafo.alignment = 3 # Justificado
                
            # Procesamos los hijos bajando un nivel
            if nodo.subsecciones:
                procesar_nodos(nodo.subsecciones, nivel + 1)

    # Lanzamos la recursividad desde la raiz
    procesar_nodos(diccionario_secciones)
    
    # Guardamos el archivo de forma segura
    try:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
        doc.save(ruta_salida)
        return True
    except Exception as e:
        print(f"[ERROR EXPORTACION]: {e}")
        return False