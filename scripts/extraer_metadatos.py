import os
import re
import json
import pypdf
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. MICRO-ESQUEMAS PYDANTIC (Moldes)
# ==========================================

class DatoTexto(BaseModel):
    resultado: Optional[str] = Field(default=None, description="El texto exacto extraido.")

class DatoEntero(BaseModel):
    resultado: Optional[int] = Field(default=None, description="El numero entero extraido.")

class DatoBooleano(BaseModel):
    resultado: Optional[bool] = Field(default=None, description="True o False segun corresponda.")

# ==========================================
# 2. FUNCIONES DE UTILIDAD Y PARSEO
# ==========================================

def limpiar_texto_basico(texto):
    return " ".join(texto.split())

def obtener_contexto_relevante(texto_completo, keywords, ventana=800):
    texto_lower = texto_completo.lower()
    recortes = []
    for keyword in keywords:
        indice = 0
        while True:
            indice = texto_lower.find(keyword, indice)
            if indice == -1: break
            inicio = max(0, indice - ventana)
            fin = min(len(texto_completo), indice + ventana)
            recortes.append(f"...[Contexto]...\n{texto_completo[inicio:fin]}\n")
            indice += len(keyword)
            
    if not recortes:
        return texto_completo[:3000]
    return "\n".join(recortes)[:8000]

def parsear_moneda_espanola(valor_str):
    if not valor_str or valor_str.lower() in ["null", "none"]:
        return None
    
    # 1. Limpiamos simbolos y letras
    limpio = re.sub(r'[^\d.,]', '', str(valor_str))
    if not limpio: return None

    # 2. Logica de parseo espanol (ej: 1.500.250,50 -> 1500250.50)
    if "," in limpio and "." in limpio:
        limpio = limpio.replace(".", "")   # Quitamos el punto de los miles
        limpio = limpio.replace(",", ".")  # Cambiamos la coma decimal por punto
    elif "," in limpio:
        limpio = limpio.replace(",", ".")  # Solo hay coma, asumimos que es decimal
    
    try:
        return float(limpio)
    except ValueError:
        return None

# ==========================================
# 3. EXTRACCION ATOMICA (Una funcion por campo)
# ==========================================

def extraer_presupuesto_base(texto_doc, llm):
    print("    - Extrayendo Presupuesto Base...")
    contexto = obtener_contexto_relevante(texto_doc, ["presupuesto base", "importe neto", "excluido"], 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el IMPORTE NETO del Presupuesto Base de Licitacion (SIEMPRE SIN IVA).
    Si encuentras una tabla con 'IVA incluido' e 'IVA no incluido', selecciona la cifra SIN IVA.
    Devuelve EXACTAMENTE el numero (ej: "27.664,00").
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return parsear_moneda_espanola(res.resultado) if res else None

def extraer_valor_estimado(texto_doc, llm):
    print("    - Extrayendo Valor Estimado...")
    contexto = obtener_contexto_relevante(texto_doc, ["valor estimado", "prorrogas", "total"], 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el VALOR ESTIMADO DEL CONTRATO (SIN IVA, que suele incluir prorrogas).
    Devuelve EXACTAMENTE el texto original que veas (ej: "250.000,00"). No hagas conversiones.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return parsear_moneda_espanola(res.resultado) if res else None

def extraer_iva(texto_doc, llm):
    print("    - Extrayendo IVA (Afinado)...")
    contexto = obtener_contexto_relevante(texto_doc, ["iva", "tipo de iva", "porcentaje"], 600)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Busca el porcentaje de IVA. No asumas el 21%. 
    Busca en las tablas de presupuesto donde diga "IVA %" o "Tipo".
    Devuelve solo el numero entero (ej: 10).
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None









def extraer_plazo_ejecucion(texto_doc, llm):
    print("    - Extrayendo Plazo de Ejecucion...")
    contexto = obtener_contexto_relevante(texto_doc, ["plazo de ejecucion", "duracion", "meses"], 800)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el PLAZO DE EJECUCION inicial expresado en meses (solo el numero entero).
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_prorroga(texto_doc, llm):
    print("    - Extrayendo Prorroga...")
    contexto = obtener_contexto_relevante(texto_doc, ["prorroga", "meses"], 800)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el tiempo maximo de PRORROGA permitido, expresado en meses (solo el numero entero).
    Si no hay prorroga, no devuelvas nada.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_tipo_contrato(texto_doc, llm):
    print("    - Extrayendo Tipo de Contrato (Afinado)...")
    contexto = obtener_contexto_relevante(texto_doc, ["contrato de", "suministros", "obras", "servicios"], 500)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Analiza el titulo y el objeto del contrato. 
    Opciones: "Obras", "Servicios", "Suministros".
    Si el contrato es para comprar productos (como herbicidas o abono), es "Suministros".
    Si es para realizar una actividad, es "Servicios".
    Devuelve solo una palabra.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_tramitacion(texto_doc, llm):
    print("    - Extrayendo Tramitacion...")
    contexto = obtener_contexto_relevante(texto_doc, ["tramitacion ordinaria", "urgente", "emergencia"], 600)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Clasifica la tramitacion. Opciones: "Ordinaria", "Urgente", "Emergencia".
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_procedimiento(texto_doc, llm):
    print("    - Extrayendo Procedimiento (Revision)...")
    contexto = obtener_contexto_relevante(texto_doc, ["procedimiento abierto", "procedimiento negociado", "articulo 156", "articulo 168"], 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Identifica el procedimiento de adjudicacion. 
    Busca palabras clave como 'Abierto' o 'Negociado'. 
    Si menciona el 'Articulo 156', es 'Abierto'. 
    Si menciona el 'Articulo 168', es 'Negociado'.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_lotes(texto_doc, llm):
    print("    - Extrayendo Lotes...")
    contexto = obtener_contexto_relevante(texto_doc, ["lotes", "division", "lote unico"], 600)
    llm_estructurado = llm.with_structured_output(DatoBooleano)
    
    prompt = ChatPromptTemplate.from_template("""
    ¿El contrato esta dividido en lotes? Devuelve True si hay lotes, False si es lote unico o no se divide.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

def extraer_financiacion_europea(texto_doc, llm):
    print("    - Extrayendo Financiacion Europea...")
    contexto = obtener_contexto_relevante(texto_doc, ["feder", "europea", "next generation", "mecanismo"], 600)
    llm_estructurado = llm.with_structured_output(DatoBooleano)
    
    prompt = ChatPromptTemplate.from_template("""
    ¿El contrato cuenta con financiacion de fondos europeos (FEDER, Next Generation, etc)? Devuelve True o False.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    return res.resultado if res else None

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================

def procesar_documento(ruta, llm):
    nombre = os.path.basename(ruta)
    print(f"\nProcesando: {nombre}")
    
    datos_finales = {"archivo": nombre}

    try:
        reader = pypdf.PdfReader(ruta)
        texto_completo = "".join([p.extract_text() or "" for p in reader.pages])
        texto_limpio = limpiar_texto_basico(texto_completo)
        
        # 1. Regex (Determinista)
        exp = re.search(r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})", texto_limpio)
        if exp: datos_finales["expediente"] = exp.group(1)
        
        cpvs = re.findall(r"(?i)CPV.*?(\d{8})[-\s]*(\d)", texto_limpio)
        if cpvs: datos_finales["cpv"] = list(set([f"{match[0]}-{match[1]}" for match in cpvs]))

        # 2. Funciones Atomicas (LLM)
        datos_finales["presupuesto_base_licitacion"] = extraer_presupuesto_base(texto_limpio, llm)
        datos_finales["valor_estimado_contrato"] = extraer_valor_estimado(texto_limpio, llm)
        datos_finales["iva_porcentaje"] = extraer_iva(texto_limpio, llm)
        datos_finales["plazo_ejecucion_meses"] = extraer_plazo_ejecucion(texto_limpio, llm)
        datos_finales["tiempo_prorroga_meses"] = extraer_prorroga(texto_limpio, llm)
        datos_finales["tipo_contrato"] = extraer_tipo_contrato(texto_limpio, llm)
        datos_finales["tramitacion"] = extraer_tramitacion(texto_limpio, llm)
        datos_finales["procedimiento"] = extraer_procedimiento(texto_limpio, llm)
        datos_finales["lotes"] = extraer_lotes(texto_limpio, llm)
        datos_finales["financiacion_europea"] = extraer_financiacion_europea(texto_limpio, llm)

        # Limpiar Nones
        datos_finales = {k: v for k, v in datos_finales.items() if v is not None}

    except Exception as e:
        print(f"Error procesando {nombre}: {e}")

    return datos_finales

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent                
    PROJECT_ROOT = BASE_DIR.parent           
    
    carpeta_datos = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
    ruta_final_json = PROJECT_ROOT / "datos" / "metadatos" / "metadatos.json"
    
    ruta_final_json.parent.mkdir(parents=True, exist_ok=True)
    
    if not carpeta_datos.exists():
        print(f"[ERROR] La carpeta {carpeta_datos} no existe.")
        exit(1)
        
    archivos_pdf = list(carpeta_datos.glob("*.pdf"))
    resultados = []
    
    print("Inicializando modelo LLM...")
    llm_global = ChatOllama(model="llama3.1", temperature=0.0)
    
    print(f"--- Iniciando Extraccion Atomica ({len(archivos_pdf)} documentos) ---")
    for f in archivos_pdf:
        resultados.append(procesar_documento(f, llm_global))
    
    with open(ruta_final_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Proceso finalizado ---")
    print(f"Guardado en: {ruta_final_json}")