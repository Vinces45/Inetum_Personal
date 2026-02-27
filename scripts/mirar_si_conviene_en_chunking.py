import os
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

def procesar_pdf_a_markdown(ruta_pdf, id_documento, metadatos_json):
    print(f"-> Convirtiendo a Markdown: {ruta_pdf}")
    
    try:
        # 1. Convertir PDF a Markdown
        # Esta funcion lee el PDF, detecta titulos, tablas y listas
        md_text = pymupdf4llm.to_markdown(ruta_pdf)
        
        # 2. Definir la jerarquia de cabeceras para el corte
        # Adaptamos esto a la estructura tipica de un PCAP o PPT
        headers_to_split_on = [
            ("#", "Capitulo"),
            ("##", "Seccion"),
            ("###", "Clausula"),
        ]
        
        # 3. Configurar el Splitter de Markdown
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            # Dejamos el titulo dentro del chunk para que el LLM sepa de que trata
            strip_headers=False 
        )
        
        # 4. Generar los chunks semanticos
        md_header_splits = markdown_splitter.split_text(md_text)
        
        # 5. Enriquecer con tus metadatos
        for chunk in md_header_splits:
            chunk.metadata["doc_id"] = id_documento
            
            for clave, valor in metadatos_json.items():
                if clave != "archivo":
                    chunk.metadata[clave] = str(valor)
                    
        return md_header_splits
        
    except Exception as e:
        print(f"Error procesando {ruta_pdf}: {e}")
        return []