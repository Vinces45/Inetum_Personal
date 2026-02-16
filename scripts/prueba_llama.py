import os
import re
import json
import requests
import pypdf
from pathlib import Path

# --- CONFIGURACION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO_LLM = "llama3" 

def limpiar_texto_basico(texto):
    # Limpieza suave para no perder la estructura de frases
    if texto:
        return " ".join(texto.split())
    return ""

# def llamada_ollama(prompt):
#     payload = {
#         "model": MODELO_LLM,
#         "prompt": prompt,
#         "stream": False,
#         "options": {"temperature": 0.0} 
#     }
#     try:
#         r = requests.post(OLLAMA_URL, json=payload)
#         if r.status_code == 200:
#             return r.json().get("response", "").strip()
#     except Exception as e:
#         print(f"Error LLM: {e}")
#     return None

def llamada_ollama(prompt):
    payload = {
        "model": MODELO_LLM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,       # Determinista
            "num_predict": 100,       # <--- NUEVO: Corta a las 100 palabras (evita bucles)
            "stop": ["\n", "Instrucciones"] # <--- NUEVO: Para si intenta escribir instrucciones nuevas
        }
    }
    try:
        # TIMEOUT: Si en 60 segundos no contesta, cortamos y seguimos
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("      [!] ALERTA: La IA tardó demasiado (Timeout). Saltando...")
        return "Error Timeout"
    except Exception as e:
        print(f"      [!] Error conectando con LLM: {e}")
    return None

# --- AGENTE 1: ESPECIALISTA EN DINERO ---
def auditar_presupuesto(texto_contexto, valor_candidato):
    print(f"      [$] Auditando Presupuesto (Candidato: {valor_candidato})...")
    
    prompt = f"""
    Eres un auditor financiero experto en Contratos Publicos.
    Tu objetivo es identificar el PRESUPUESTO BASE DE LICITACION (Sin IVA).
    
    TEXTO DEL CONTRATO:
    '''{texto_contexto[:6000]}'''
    
    CANDIDATO ENCONTRADO POR REGEX: "{valor_candidato}"
    
    INSTRUCCIONES CRITICAS:
    1. Si el candidato "{valor_candidato}" es correcto (coincide con el Presupuesto Base sin impuestos), confirmalo.
    2. ALERTA DE FALSOS POSITIVOS:
       - Si el valor es 140.000, 100.000 o cifras redondas exactas, suele ser el umbral de la ley o un seguro. RECHAZALO si no dice explicitamente "Presupuesto Base".
       - Ignora el "Valor Estimado" si es diferente al Presupuesto Base (el estimado suele incluir prorrogas).
    3. Si el candidato es incorrecto o es None, busca el valor real en el texto.
    
    RESPUESTA:
    - Solo dame el numero (ej: 125.035,62).
    - Si no encuentras nada fiable, responde "No encontrado".
    """
    
    respuesta = llamada_ollama(prompt)
    
    # Limpieza post-procesado
    if respuesta:
        # A veces la IA dice "El presupuesto es 100". Nos quedamos lo ultimo
        if ":" in respuesta:
            respuesta = respuesta.split(":")[-1].strip()
        # Limpiamos simbolos de moneda si la IA los puso
        respuesta = respuesta.replace("€", "").replace("euros", "").strip()
        return respuesta
    return "Error IA"




# --- AGENTE 2: ESPECIALISTA EN TIEMPO ---
def auditar_plazo(texto_contexto, valor_candidato):
    print(f"      [T] Auditando Plazo (Candidato: {valor_candidato})...")
    
    prompt = f"""
    Eres un experto legal en Pliegos Administrativos.
    Tu objetivo es identificar el PLAZO DE EJECUCION o DURACION del contrato.
    
    TEXTO DEL CONTRATO:
    '''{texto_contexto[:6000]}'''
    
    CANDIDATO ENCONTRADO POR REGEX: "{valor_candidato}"
    
    INSTRUCCIONES CRITICAS:
    1. Buscamos la duracion principal del servicio/obra.
    2. ALERTA DE CONFUSION:
       - NO confundir con "Plazo de Garantia".
       - NO sumar las prorrogas, solo el plazo inicial.
       - NO confundir con plazos de presentacion de ofertas.
    
    RESPUESTA:
    - Si el candidato es correcto, repitelo.
    - Si es incorrecto, dame el dato real (ej: "4 meses", "1 ano").
    - Si no aparece, responde "No encontrado".
    """
    
    respuesta = llamada_ollama(prompt)
    
    if respuesta:
        if ":" in respuesta:
            respuesta = respuesta.split(":")[-1].strip()
        return respuesta
    return "Error IA"



def procesar_documento(ruta):
    nombre = os.path.basename(ruta)
    datos = {
        "archivo": nombre,
        "expediente": "No encontrado",
        "presupuesto": "No encontrado", #Falta
        "tramitacion": "Ordinaria",     #Falta
        "cpv": "No encontrado", 
        "plazo": "No encontrado",       #Falta
        "metodo": "FALLO", 
        "tipo_pliego": "PCAP" if "PCAP" in nombre.upper() else "PPT"
    }

    try:
        reader = pypdf.PdfReader(ruta)
        texto_completo = ""
        texto_inicio = "" # Para la IA (primeras 10 paginas)
        
        for i, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt:
                texto_completo += txt + "\n"
                if i < 10: texto_inicio += txt + "\n"
        
        texto_limpio = limpiar_texto_basico(texto_completo)
        
        # 1. EXPEDIENTE (Regex)
        match_exp = re.search(r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})", texto_limpio)
        if match_exp:
            datos["expediente"] = match_exp.group(1)

        # 2. TRAMITACION (Regex ajustada corta distancia)
        if re.search(r"tramita.{0,30}urgente", texto_limpio, re.IGNORECASE):
            datos["tramitacion"] = "Urgente"

        # 3. CPV (Regex)
        match_cpv = re.search(r"(\d{8}-\d)", texto_limpio)
        if match_cpv:
            datos["cpv"] = match_cpv.group(1)

        # 4. PRESUPUESTO (Funcion Especialista)
        # Primero buscamos candidato regex
        patron_dinero = r"(?i)(?:presupuesto base|valor estimado|importe).{1,50}?(\d{1,3}(?:\.\d{3})*,\d{2})"
        match_pre = re.search(patron_dinero, texto_limpio)
        candidato_pre = match_pre.group(1) if match_pre else None

        # Llamamos al Agente Presupuestario
        datos["presupuesto"] = auditar_presupuesto(texto_inicio, candidato_pre)

        # 5. PLAZO (Funcion Especialista)
        # Primero buscamos candidato regex
        patron_plazo = r"(?:plazo|duracion).{1,50}?(\d+\s*(?:meses|dias|anos))"
        match_plazo = re.search(patron_plazo, texto_limpio, re.IGNORECASE)
        candidato_plazo = match_plazo.group(1) if match_plazo else None
        
        # Llamamos al Agente de Plazos
        datos["plazo"] = auditar_plazo(texto_inicio, candidato_plazo)

    except Exception as e:
        print(f"Error en {nombre}: {e}")

    return datos

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent      
    PROJECT_ROOT = BASE_DIR.parent                  
    carpeta_datos = PROJECT_ROOT / "datos" / "pdfs"
    
    resultados = []
    print(f"--- Procesando documentos en {carpeta_datos} ---")
    
    for f in carpeta_datos.glob("*.pdf"):
        print(f"Analizando: {f.name}")
        resultados.append(procesar_documento(f))
        
    with open("dataset_final_validado.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print("--- Proceso finalizado. Revisa dataset_final_validado.json ---")