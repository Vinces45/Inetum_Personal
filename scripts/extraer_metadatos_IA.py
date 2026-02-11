import json
import requests
from pathlib import Path
import os

def analizar_requisitos_ppt(texto_ppt):
    url = "http://localhost:11434/api/generate"
    
    # Configuramos el "System Prompt" para que la IA se porte como un consultor
    prompt = f"""
    Eres un consultor experto en licitaciones de La Rioja. 
    Analiza el siguiente fragmento de un PPT y dime cuales son los 3 requisitos 
    tecnicos mas importantes que debe cumplir el adjudicatario.
    Responde de forma esquematica.

    Texto del pliego:
    {texto_ppt[:4000]} 
    """
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        print("Enviando texto a Ollama (procesando localmente)...")
        response = requests.post(url, json=payload)
        return response.json()["response"]
    except Exception as e:
        return f"Error de conexion: {e}"



BASE_DIR = Path(__file__).resolve().parent        # scripts/
PROJECT_ROOT = BASE_DIR.parent                    # Proyecto/
carpeta_datos = PROJECT_ROOT / "datos" / "pdfs" / "01_PCAP_Conserjeria.pdf"
resultados = []
print("--------------Iniciando lectura de documento---------------")
print(carpeta_datos)
# for archivo in carpeta_datos.glob("*.pdf"):
#     ruta_completa = os.path.join(carpeta_datos, archivo)
#     info = analizar_documento(ruta_completa)
#     resultados.append(info)

# with open("metadatos_tfg.json", "w", encoding="utf-8") as f:
#     json.dump(resultados, f, indent=4, ensure_ascii=False)
# print("Lectura de datos completada")



texto_de_prueba = "El sistema de IA debera integrarse con la API del Gobierno de La Rioja y garantizar un SLA del 99%..."
analisis = analizar_requisitos_ppt(texto_de_prueba)

print("\n--- ANALISIS TECNICO DE LA IA ---")
print(analisis)