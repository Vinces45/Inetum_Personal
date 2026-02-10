import os
import re
import pypdf
import json
from pathlib import Path


def limpiar_valor(texto):
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
        
        texto_limpio = limpiar_valor(texto_completo)

        # 1. Expediente: Busqueda por patron de La Rioja (XX-X-X.XX-XXXX/202X)
        patron_rioja = r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})"
        match_exp = re.search(patron_rioja, texto_limpio)
        if match_exp:
            datos["expediente"] = match_exp.group(1)
        
        # 2. CPV: Flexible con espacios
        match_cpv = re.search(r"(\d{8}\s*-\s*\d)", texto_limpio)
        if match_cpv:
            datos["cpv"] = match_cpv.group(1).replace(" ", "")

        # 3. Tramitacion
        if "URGENTE" in texto_limpio.upper():
            datos["tramitacion"] = "Urgente"

        # 4. Presupuesto: Filtro de miles para evitar ruidos (como el 6,49)
        # Solo acepta cifras con punto de miles: ej 56.264,00
        match_pres = re.search(r"(?:licitacion|base|total).*?(\d{1,3}(?:\.\d{3})+(?:,\d{2}))\s*(?:euros|€)", texto_limpio, re.IGNORECASE)
        if match_pres:
            datos["presupuesto"] = match_pres.group(1)
        
        # 5. Plazo: Ajustado para no cortar palabras (meses, dias, etc)
        # Ordenamos de mas largo a mas corto (meses antes que mes)
        patrones_plazo = r"(?:meses|mes|dias|dia|anos|ano|semanas|semana)"
        match_plazo = re.search(r"(?:plazo|duracion).*?(\d+\s*" + patrones_plazo + r")", texto_limpio, re.IGNORECASE)
        if match_plazo:
            datos["plazo"] = match_plazo.group(1)
        
    except Exception as e:
        print(f"Error: {e}")
    
    return datos




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