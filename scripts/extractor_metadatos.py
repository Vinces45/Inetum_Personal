import os
import re
import json
import requests
import pypdf
from pathlib import Path

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

def obtener_contexto_relevante(texto_completo, keywords, ventana=500):

    texto_lower = texto_completo.lower()
    recortes = []
    
    for keyword in keywords:
        indice = 0
        while True:
            indice = texto_lower.find(keyword, indice)
            if indice == -1: 
                break
            
            inicio = max(0, indice - ventana)
            fin = min(len(texto_completo), indice + ventana)
            recorte = texto_completo[inicio:fin]
            #Para evitar alucinaciones de la IA sobre contexto anterior al recorte
            recortes.append(f"...[Pagina irrelevante]...\n{recorte}\n")
            indice += len(keyword)
            
    if not recortes:
        return texto_completo[:3000]
    
    contexto_final = "\n".join(recortes)

    #Ver si es suficiente o dar más contexto
    return contexto_final[:6000]


def llamada_ollama(prompt):
    payload = {
        "model": MODELO_LLM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,       
            "num_predict": 200,   
            "stop": ["Instrucciones"]
        }
    }
    try:
        respuesta = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if respuesta.status_code == 200:
            return respuesta.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("La IA tardó demasiado (Timeout). ")
        return "Error Timeout"
    except Exception as e:
        print(f"Error conectando con LLM: {e}")
    return None


def auditar_presupuesto(texto_completo_doc, valor_candidato):
    print(f"---- Obteniendo Presupuesto (Candidato Regex: {valor_candidato})...")
    
    # palabras_clave_dinero = ["presupuesto base", "valor estimado", "importe neto", "precio del contrato", "cuantia", "lotes"]
    # contexto_filtrado = obtener_contexto_relevante(texto_completo_doc, palabras_clave_dinero, ventana=1000)
   
    # if valor_candidato:
    #     candidato_str = valor_candidato 
    # else:
    #     candidato_str= "null"

    # prompt = f"""
    # Eres un experto financiero. Extrae el PRESUPUESTO BASE (Sin Impuestos).
    
    # CANDIDATO REGEX: {candidato_str}
    # CONTEXTO: '''{contexto_filtrado}'''
    
    # INSTRUCCIONES:
    # 1. Prioriza el "Presupuesto Base de Licitacion" sobre el "Valor Estimado".
    # 2. Si hay varios lotes, busca la SUMA TOTAL.
    # 3. Responde SOLO con este JSON:
    # {{
    #     "monto": "100.000,00",
    #     "razon": "Justificacion breve"
    # }}
    # """

    # Añadimos "IVA" y "Total" para que el LLM vea la diferencia
    palabras_clave_dinero = ["presupuesto base", "valor estimado", "importe neto", "excluido", "IVA", "base de licitacion"]
    contexto_filtrado = obtener_contexto_relevante(texto_completo_doc, palabras_clave_dinero, ventana=800)
    
    prompt = f"""
    Eres un auditor financiero estricto. Tu objetivo es extraer el PRESUPUESTO BASE DE LICITACIÓN (SIN IMPUESTOS / IVA EXCLUIDO).

    CANDIDATO DETECTADO POR REGEX: {valor_candidato}
    CONTEXTO DEL DOCUMENTO: 
    '''{contexto_filtrado}'''
    
    REGLAS CRÍTICAS:
    1. Si encuentras dos cifras (una con IVA y otra sin IVA), ELIGE SIEMPRE LA MENOR (Sin IVA).
    2. Ignora el "Valor Estimado" si es diferente al Presupuesto Base (el valor estimado suele incluir prórrogas).
    3. Busca términos como "Importe neto", "IVA excluido".
    
    Responde SOLO con este JSON:
    {{
        "monto": "27.664,00",
        "tipo": "Base sin impuestos",
        "razon": "Aparece junto al texto 'IVA excluido'"
    }}
    """
    
    respuesta_inicial = llamada_ollama(prompt)
    
    objeto_json = limpiar_y_extraer_json(respuesta_inicial)
    
    if objeto_json and objeto_json.get("monto"):
        dato = str(objeto_json["monto"]).replace("€", "").strip()
        if "null" not in dato.lower():
            return dato

    if valor_candidato: 
        return valor_candidato
    return "No encontrado"


def auditar_plazo(texto_completo_doc, valor_candidato):
    print(f"---- Obteniendo Plazo (Candidato Regex: {valor_candidato})...")
    
    keywords_plazo = ["plazo de ejecucion", "duracion", "vigencia", "meses", "dias", "periodo", "desde"]
    contexto_filtrado = obtener_contexto_relevante(texto_completo_doc, keywords_plazo, ventana=1000)
    
     
    if valor_candidato: 
        candidato_str = valor_candidato
    else:
        candidato_str = "null"
    
    prompt = f"""
    Extrae el PLAZO DE EJECUCION del contrato.
    
    CANDIDATO REGEX: {candidato_str}
    CONTEXTO: '''{contexto_filtrado}'''
    
    INSTRUCCIONES:
    1. Busca "Plazo de ejecucion", "Duracion" o fechas.
    2. Responde SOLO con este JSON:
    {{
        "plazo": "X meses" (o "X dias", o "null"),
        "razon": "Justificacion"
    }}
    """
    
    respuesta_inicial = llamada_ollama(prompt)
    
    objeto_json = limpiar_y_extraer_json(respuesta_inicial)
    
    if objeto_json and objeto_json.get("plazo"):
        dato = str(objeto_json["plazo"])
        if "null" not in dato.lower():
            return dato

    if valor_candidato: 
        return valor_candidato
    return "No encontrado"


def procesar_documento(ruta):
    nombre = os.path.basename(ruta)
    datos = {
        "archivo": nombre,
        "expediente": "No encontrado",
        "presupuesto": "No encontrado", 
        "tramitacion": "Ordinaria",     
        "cpv": "No encontrado", 
        "plazo": "No encontrado"       
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
        if expediente_encontrado: datos["expediente"] = expediente_encontrado.group(1)

        # 2. TRAMITACION (Regex)
        if re.search(r"tramita.{0,30}urgente", texto_limpio, re.IGNORECASE):
            datos["tramitacion"] = "Urgente"

        # 3. CPV (Regex)
        # cpv_encontrado = re.search(r"(\d{8})[\s\n-]*(\d)", texto_limpio)
        # if cpv_encontrado:
        #     datos["cpv"] = f"{cpv_encontrado.group(1)}-{cpv_encontrado.group(2)}"
        cpv_encontrado = re.search(r"(?i)CPV.*?(\d{8})[-\s]*(\d)", texto_limpio)
        if cpv_encontrado:
            datos["cpv"] = f"{cpv_encontrado.group(1)}-{cpv_encontrado.group(2)}"
        else:
            # Plan B: Buscar solo el formato numérico si está muy aislado, pero es arriesgado
            pass

        # 4. PRESUPUESTO (Regex + IA)
        
        patron_dinero = r"(?i)(?:presupuesto base|valor estimado|importe).{1,50}?(\d{1,3}(?:\.\d{3})*,\d{2})"
        presupuesto_encontrado = re.search(patron_dinero, texto_limpio)
        candidato_pre = None
        if presupuesto_encontrado:
            candidato_pre = presupuesto_encontrado.group(1) 
        
        datos["presupuesto"] = auditar_presupuesto(texto_completo, candidato_pre)

        # 5. PLAZO (Regex + IA)
        patron_plazo = r"(?i)(?:plazo|duracion).{1,50}?(\d+\s*(?:meses|dias|anos))"
        plazo_encontrado = re.search(patron_plazo, texto_limpio, re.IGNORECASE)
        candidato_plazo = None
        if plazo_encontrado:
            candidato_plazo = plazo_encontrado.group(1)
        
        datos["plazo"] = auditar_plazo(texto_completo, candidato_plazo)

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
        print(f"Analizando: {f.name}")
        resultados.append(procesar_documento(f))
        
    
        
    with open(ruta_final_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print(f"--- Proceso finalizado ---")
    print(f"Archivo guardado en: {ruta_final_json}")