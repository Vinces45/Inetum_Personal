import os
import re
import requests
import pypdf
import json
from pathlib import Path


def limpiar_texto(texto):
    if texto:
        return " ".join(texto.split()).strip()
    return "No encontrado"



def analizar_documento(ruta_archivo):
    datos = {
        "archivo": os.path.basename(ruta_archivo),
        "expediente": "No encontrado",
        "presupuesto": "No encontrado",
        "cpv": "No encontrado",
        "tramitacion": "Ordinaria",
        "plazo": "No encontrado",
        "tipo_pliego": "PCAP" if "PCAP" in ruta_archivo.upper() else "PPT"
    }

    try:
        reader = pypdf.PdfReader(ruta_archivo)
        texto_completo = ""
        # Leemos todo el documento para no dejarnos nada
        for page in reader.pages:
            texto_completo += page.extract_text() + " "
        
        texto_limpio = limpiar_texto(texto_completo)

        # 1. Expediente: Busqueda por patron con este formato (XX-X-X.XX-XXXX/202X)
        patron_expediente = r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})"
        expediente_encontrado = re.search(patron_expediente, texto_limpio)
        if expediente_encontrado:
            datos["expediente"] = expediente_encontrado.group(1)
        
        # 2. CPV
        cpv_encontrado = re.search(r"(\d{8}\s*-\s*\d)", texto_limpio)
        if cpv_encontrado:
            datos["cpv"] = cpv_encontrado.group(1).replace(" ", "")

        # 3. Tramitacion
        if "URGENTE" in texto_limpio.upper():
            datos["tramitacion"] = "Urgente"

        # 4. Presupuesto: Solo acepta cifras con punto de miles: ej 56.264,00
        presupuesto_encontrado = re.search(r"(?:licitacion|base|total).*?(\d{1,3}(?:\.\d{3})+(?:,\d{2}))\s*(?:euros|€)", texto_limpio, re.IGNORECASE)
        if presupuesto_encontrado:
            datos["presupuesto"] = presupuesto_encontrado.group(1)
        
        # 5. Plazo
        patrones_plazo = r"(?:meses|mes|dias|dia|a.os|a.o|semanas|semana)"
        plazo_encontrado = re.search(r"(?:plazo|duracion).*?(\d+\s*" + patrones_plazo + r")", texto_limpio, re.IGNORECASE)
        if plazo_encontrado:
            datos["plazo"] = plazo_encontrado.group(1)
        
    except Exception as e:
        print(f"Error: {e}")
    
    return datos

def consultar_llama(texto_pdf, variables_solicitadas):
    url = "http://localhost:11434/api/generate"
    
    # El prompt le pide a la IA que complete lo que el Regex no puede
    prompt = f"""
    Eres un experto legal. Analiza este fragmento de pliego y extrae: {variables_solicitadas}.
    Responde solo en formato JSON.
    
    Texto: {texto_pdf[:3000]}
    """
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        return json.loads(response.json()["response"])
    except:
        return None


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