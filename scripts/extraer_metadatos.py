import os
import re
import pypdf
import json
import spacy
from pathlib import Path
from fuzzywuzzy import process 

def buscar_patrones(texto, patrones): 
    for p in patrones: 
        m = re.search(p, texto, re.IGNORECASE) 
        if m: 
            return m.group(1) 
        return None

def fuzzy_keyword(texto, palabra_objetivo): 
    palabras = texto.split() 
    match, score = process.extractOne(palabra_objetivo, palabras) 
    return match if score > 80 else None

def extraer_entidades(texto): 
    doc = nlp(texto) 
    return [(ent.text, ent.label_) for ent in doc.ents]



def analizar_documento(ruta_archivo):
    #Preparar JSON con mis "Variables Críticas"
    datos = {
        "archivo": os.path.basename(ruta_archivo),
        "expediente": "No encontrado",
        "presupuesto": "No encontrado",
        "plazo": "No encontrado",
        "tipo_pliego": "PCAP" if "PCAP" in ruta_archivo.upper() else "PPT"
    }

    try:
        reader = pypdf.PdfReader(ruta_archivo)
        #Extraer texto de las 5 primeras páginas donde están los datos clave
        texto = ""
        for i in range(len(reader.pages)):
            texto += reader.pages[i].extract_text() + "\n"

        #Búsqueda de Expedientes
        expediente_encontrado = re.search(r"EXPEDIENTE:\s*([\w\.\-/]+)", texto, re.IGNORECASE)
        if expediente_encontrado:
            datos["expediente"] = expediente_encontrado.group(1)
        
        #Búsqueda de presupuesto
        presupuesto_encontrado = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{2}))\s*(?:euros|€)", texto, re.IGNORECASE)
        if presupuesto_encontrado:
            datos["presupuesto"] = presupuesto_encontrado.group(1)
        
        #Búsqueda de plazo
        plazo_encontrado = re.search(r"plazo.*?(\d+\s*(?:d[ií]a(?:s)?|semana(?:s)?|mes(?:es)?|año(?:s)?))", texto, re.IGNORECASE | re.DOTALL)
        if plazo_encontrado:
            datos["plazo"] = plazo_encontrado.group(1)
        
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
    
    return datos



nlp = spacy.load("es_core_news_md")

BASE_DIR = Path(__file__).resolve().parent        # scripts/
PROJECT_ROOT = BASE_DIR.parent                    # Proyecto/
carpeta_datos = PROJECT_ROOT / "datos" / "pdfs"
resultados = []
print("--------------Iniciando lectura de documentos automática---------------")

for archivo in carpeta_datos.glob("*.pdf"):
    ruta_completa = os.path.join(carpeta_datos, archivo)
    info = analizar_documento(ruta_completa)
    resultados.append(info)

with open("metadatos_tfg.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)
print("Lectura de datos completada")