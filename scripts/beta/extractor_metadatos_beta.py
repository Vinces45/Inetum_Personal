import os
import re
import json
import requests
import pypdf
from pathlib import Path

# Se ha añadido más campos al JSON
# En principio se ha mejorado la eficiencia de las llamadas LLM porque al meter nuevos metadatos que son buscados por LLM pues en lugar de hacer una
# llamada por metadato, se ha hecho que en una misma llamada se encuentran todos los metadatos
# esto será más eficiente pero igual no es lo mejor en precisión, falta probarlo. Aun así se podría separar en llamada por metadato que aunque tarde
# como solo es una única vez pues se podría llegar a tolerar. Es simplemente buscar.
# Otra modificación es la obtención de un array de CVP para aquellos que tengan un PCAP por lotes y guardar todos los CVP

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_LLM = "llama3" 

def limpiar_texto_basico(texto):
    return " ".join(texto.split())

def limpiar_y_extraer_json(texto):
    if not texto: 
        return None
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass 

    texto_json = re.search(r"(\{.*\})", texto, re.DOTALL)
    if texto_json:
        json_final = texto_json.group(1)
        try:
            return json.loads(json_final)
        except json.JSONDecodeError:
            return None
    return None

def llamada_ollama(prompt):
    payload = {
        "model": MODELO_LLM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,       
            "num_predict": 400,   
            "stop": ["Instrucciones"]
        }
    }
    try:
        respuesta = requests.post(OLLAMA_URL, json=payload, timeout=90)
        if respuesta.status_code == 200:
            return respuesta.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("La IA tardo demasiado (Timeout).")
        return "Error Timeout"
    except Exception as e:
        print(f"Error conectando con LLM: {e}")
    return None

def extraer_metadatos_avanzados(texto_completo, candidato_presupuesto, candidato_plazo):
    print("---- Obteniendo metadatos semanticos mediante LLM...")
    
    # El PCAP suele concentrar esta informacion en las primeras paginas
    contexto_inicial = texto_completo[:8000]
    
    prompt = f"""
    Eres un auditor estricto de contratacion publica. Tu objetivo es extraer datos clave del pliego.
    
    CANDIDATO PRESUPUESTO REGEX: {candidato_presupuesto}
    CANDIDATO PLAZO REGEX: {candidato_plazo}
    
    CONTEXTO DEL DOCUMENTO (Primeras paginas): 
    '''{contexto_inicial}'''
    
    REGLAS CRITICAS:
    1. PRESUPUESTO BASE: Si encuentras dos cifras, elige SIEMPRE la menor (Sin Impuestos / IVA excluido).
    2. TIPO CONTRATO: Debe ser exactamente "Obras", "Servicios" o "Suministros".
    3. PROCEDIMIENTO: Identifica si es "Abierto", "Abierto Simplificado", etc.
    4. Responde SOLO con un JSON valido con esta estructura exacta, sin texto adicional:
    
    {{
        "presupuesto_base": "27.664,00",
        "plazo_ejecucion": "24 meses",
        "tipo_contrato": "Servicios",
        "procedimiento": "Abierto Simplificado",
        "lotes": false,
        "criterios_adjudicacion": "Precio y otros",
        "financiacion_europea": false,
        "contrato_reservado": true
    }}
    """
    
    respuesta = llamada_ollama(prompt)
    objeto_json = limpiar_y_extraer_json(respuesta)
    
    if objeto_json:
        return objeto_json
    return {}

def procesar_documento(ruta):
    nombre = os.path.basename(ruta)
    datos = {
        "archivo": nombre,
        "expediente": "No encontrado",
        "cpv": [],
        "tramitacion": "Ordinaria",
        "presupuesto": "No encontrado", 
        "plazo": "No encontrado",
        "tipo_contrato": "No encontrado",
        "procedimiento": "No encontrado",
        "lotes": False,
        "criterios_adjudicacion": "No encontrado",
        "financiacion_europea": False,
        "contrato_reservado": False
    }

    try:
        reader = pypdf.PdfReader(ruta)
        texto_completo = ""
        
        for pagina in reader.pages:
            texto = pagina.extract_text()
            if texto: 
                texto_completo += texto + "\n"
        
        texto_limpio = limpiar_texto_basico(texto_completo)
        
        # 1. EXPEDIENTE (Regex)
        expediente_encontrado = re.search(r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})", texto_limpio)
        if expediente_encontrado: 
            datos["expediente"] = expediente_encontrado.group(1)

        # 2. TRAMITACION (Regex)
        if re.search(r"tramita.{0,30}urgente", texto_limpio, re.IGNORECASE):
            datos["tramitacion"] = "Urgente"

        # 3. CPV (Regex Multiple)
        cpvs_encontrados = re.findall(r"(?i)CPV.*?(\d{8})[-\s]*(\d)", texto_limpio)
        if cpvs_encontrados:
            datos["cpv"] = list(set([f"{match[0]}-{match[1]}" for match in cpvs_encontrados]))

        # 4. Busqueda previa Regex para guiar al LLM
        patron_dinero = r"(?i)(?:presupuesto base|valor estimado|importe).{1,50}?(\d{1,3}(?:\.\d{3})*,\d{2})"
        presupuesto_regex = re.search(patron_dinero, texto_limpio)
        candidato_pre = presupuesto_regex.group(1) if presupuesto_regex else "null"
        
        patron_plazo = r"(?i)(?:plazo|duracion).{1,50}?(\d+\s*(?:meses|dias|anos))"
        plazo_regex = re.search(patron_plazo, texto_limpio, re.IGNORECASE)
        candidato_plazo = plazo_regex.group(1) if plazo_regex else "null"

        # 5. LLAMADA UNICA AL LLM PARA CAMPOS SEMANTICOS
        datos_llm = extraer_metadatos_avanzados(texto_completo, candidato_pre, candidato_plazo)
        
        if datos_llm:
            datos["presupuesto"] = datos_llm.get("presupuesto_base", candidato_pre)
            datos["plazo"] = datos_llm.get("plazo_ejecucion", candidato_plazo)
            datos["tipo_contrato"] = datos_llm.get("tipo_contrato", datos["tipo_contrato"])
            datos["procedimiento"] = datos_llm.get("procedimiento", datos["procedimiento"])
            datos["lotes"] = datos_llm.get("lotes", datos["lotes"])
            datos["criterios_adjudicacion"] = datos_llm.get("criterios_adjudicacion", datos["criterios_adjudicacion"])
            datos["financiacion_europea"] = datos_llm.get("financiacion_europea", datos["financiacion_europea"])
            datos["contrato_reservado"] = datos_llm.get("contrato_reservado", datos["contrato_reservado"])

    except Exception as e:
        print(f"Error critico en {nombre}: {e}")

    return datos

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent      
    PROJECT_ROOT = BASE_DIR.parent                  
    carpeta_datos = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
    ruta_final_json = PROJECT_ROOT / "datos" / "metadatos" / "metadatos_finales.json"
    
    resultados = []
    print(f"--- Procesando documentos en {carpeta_datos} ---")
    
    for f in carpeta_datos.glob("*.pdf"):
        print(f"\nAnalizando: {f.name}")
        resultados.append(procesar_documento(f))
        
    with open(ruta_final_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Proceso finalizado ---")
    print(f"Archivo guardado en: {ruta_final_json}")