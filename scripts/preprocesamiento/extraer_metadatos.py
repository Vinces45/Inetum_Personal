import os
import re
import json
import pypdf
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import fitz

class DatoTexto(BaseModel):
    resultado: Optional[str] = Field(default=None, description="El texto exacto extraido.")

class DatoEntero(BaseModel):
    resultado: Optional[int] = Field(default=None, description="El numero entero extraido.")

class DatoBooleano(BaseModel):
    resultado: Optional[bool] = Field(default=None, description="True o False segun corresponda.")

def limpiar_texto_basico(texto):
    return " ".join(texto.split())

def obtener_contexto_relevante(texto_completo, lista_palabras_clave, ventana=800):
    texto_lower = texto_completo.lower()
    recortes = []
    for palabra_clave in lista_palabras_clave:
        indice = 0
        while True:
            indice = texto_lower.find(palabra_clave, indice)
            if indice == -1: 
                break
            inicio = max(0, indice - ventana)
            fin = min(len(texto_completo), indice + ventana)
            recortes.append(f"...[Contexto]...\n{texto_completo[inicio:fin]}\n")
            indice += len(palabra_clave)
            
    if not recortes:
        return texto_completo[:3000]
    return "\n".join(recortes)[:8000]

def parsear_moneda_espanola(valor_str):
    if not valor_str or valor_str.lower() in ["null", "none"]:
        return None
    
    limpio = re.sub(r'[^\d.,]', '', str(valor_str))
    if not limpio: return None

    if "," in limpio and "." in limpio:
        limpio = limpio.replace(".", "")
        limpio = limpio.replace(",", ".")
    elif "," in limpio:
        limpio = limpio.replace(",", ".")
    elif "." in limpio:
        partes = limpio.split(".")
        if len(partes[-1]) == 3:
            limpio = limpio.replace(".", "")
    try:
        return float(limpio)
    except ValueError:
        return None

def extraer_presupuesto_base(texto_doc, llm):
    print("    - Extrayendo Presupuesto Base...")
    palabras_clave = ["presupuesto base", "importe neto", "excluido"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el IMPORTE NETO del Presupuesto Base de Licitacion (SIEMPRE SIN IVA).
    Si encuentras una tabla con 'IVA incluido' e 'IVA no incluido', selecciona la cifra SIN IVA.
    Devuelve EXACTAMENTE el numero (ej: "27.664,00").
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return parsear_moneda_espanola(res.resultado) 
    else:
        return None

def extraer_valor_estimado(texto_doc, llm):
    print("    - Extrayendo Valor Estimado...")
    palabras_clave = ["valor estimado", "prorrogas", "total"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el VALOR ESTIMADO DEL CONTRATO (SIN IVA, que suele incluir prorrogas).
    Devuelve EXACTAMENTE el texto original que veas (ej: "250.000,00"). No hagas conversiones.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return parsear_moneda_espanola(res.resultado) 
    else:
        return None

def extraer_iva(texto_doc, llm):
    print("    - Extrayendo IVA (Afinado)...")
    palabras_clave = ["iva", "tipo de iva", "porcentaje"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 600)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Busca el porcentaje de IVA. No asumas el 21%. 
    Busca en las tablas de presupuesto donde diga "IVA %" o "Tipo".
    Devuelve solo el numero entero (ej: 10).
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None


def extraer_plazo_ejecucion(texto_doc, llm):
    print("    - Extrayendo Plazo de Ejecucion...")
    palabras_clave = ["plazo de ejecucion", "duracion", "meses"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el PLAZO DE EJECUCION inicial expresado en meses (solo el numero entero). 
    Si aparece en años, pasalo al numero de meses equivalente. Si encuentras una fecha tal que así, de mayo a agosto, 
    pasalo al número de meses equivalente.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def extraer_prorroga(texto_doc, llm):
    print("    - Extrayendo Prorroga...")
    palabras_clave = ["prorroga", "meses"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoEntero)
    
    prompt = ChatPromptTemplate.from_template("""
    Extrae el tiempo maximo de PRORROGA permitido, expresado en meses (solo el numero entero).
    Si no hay prorroga, no devuelvas nada.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def extraer_tipo_contrato(texto_doc, llm):
    print("    - Extrayendo Tipo de Contrato (Afinado)...")
    palabras_clave = ["contrato de", "suministros", "obras", "servicios"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 500)
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
    if res:
        return res.resultado
    else:
        return None

def extraer_tramitacion(texto_doc, llm):
    print("    - Extrayendo Tramitacion...")
    palabras_clave = ["tramitacion ordinaria", "urgente", "emergencia"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Clasifica la tramitacion. Opciones: "Ordinaria", "Urgente", "Emergencia".
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def extraer_procedimiento(texto_doc, llm):
    print("    - Extrayendo Procedimiento (Revision)...")
    palabras_clave = ["procedimiento abierto", "procedimiento negociado", "articulo 156", "articulo 168"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave , 800)
    llm_estructurado = llm.with_structured_output(DatoTexto)
    
    prompt = ChatPromptTemplate.from_template("""
    Identifica el procedimiento de adjudicacion. 
    Busca palabras clave como 'Abierto' o 'Negociado'. 
    Si menciona el 'Articulo 156', es 'Abierto'. 
    Si menciona el 'Articulo 168', es 'Negociado'.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def extraer_lotes(texto_doc, llm):
    print("    - Extrayendo Lotes...")
    palabras_clave = ["lotes", "division", "lote unico"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 800)
    llm_estructurado = llm.with_structured_output(DatoBooleano)
    
    prompt = ChatPromptTemplate.from_template("""
    ¿El contrato esta dividido en lotes? Devuelve True si hay lotes, False si es lote unico o no se divide.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def extraer_financiacion_europea(texto_doc, llm):
    print("    - Extrayendo Financiacion Europea...")
    palabras_clave = ["feder", "europea", "next generation", "mecanismo"]
    contexto = obtener_contexto_relevante(texto_doc, palabras_clave, 600)
    llm_estructurado = llm.with_structured_output(DatoBooleano)
    
    prompt = ChatPromptTemplate.from_template("""
    ¿El contrato cuenta con financiacion de fondos europeos (FEDER, Next Generation, etc)? Devuelve True o False.
    CONTEXTO: '''{contexto}'''
    """)
    res = (prompt | llm_estructurado).invoke({"contexto": contexto})
    if res:
        return res.resultado
    else:
        return None

def procesar_documento(ruta, llm):
    nombre = os.path.basename(ruta)
    print(f"\nProcesando: {nombre}")
    
    datos_finales = {"archivo": nombre}

    try:
        # reader = pypdf.PdfReader(ruta)
        # texto_completo = "".join([p.extract_text() or "" for p in reader.pages])
        # texto_limpio = limpiar_texto_basico(texto_completo)

        texto_completo = ""
        with fitz.open(ruta) as doc:
            for pagina in doc:
                texto_completo += pagina.get_text("text") +"\n"  

        texto_limpio = limpiar_texto_basico(texto_completo)   

        expediente = re.search(r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})", texto_limpio)
        if expediente: 
            datos_finales["expediente"] = expediente.group(1)
        
        cpvs = re.findall(r"(?i)CPV.*?(\d{8})[-\s]*(\d)", texto_limpio)
        if cpvs: 
            datos_finales["cpv"] = list(set([f"{match[0]}-{match[1]}" for match in cpvs]))

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

        datos_finales = {k: v for k, v in datos_finales.items() if v is not None}

    except Exception as e:
        print(f"Error procesando {nombre}: {e}")

    return datos_finales

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent                
    PROJECT_ROOT = BASE_DIR.parent           
    DIR_PLIEGOS = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
    DIR_JSON = PROJECT_ROOT / "datos" / "jsons" / "metadatos.jsonl"
    
    if not DIR_PLIEGOS.exists():
        print(f"[ERROR] La carpeta {DIR_PLIEGOS} no existe.")
        exit(1)
        
    archivos_pdf = list(DIR_PLIEGOS.glob("*.pdf"))
    
    print("Inicializando modelo LLM...")
    llm_global = ChatOllama(model="llama3.1", temperature=0.0)
    
    print(f"--- Iniciando Extraccion de ({len(archivos_pdf)} documentos) ---")

    with open(DIR_JSON, "w", encoding="utf-8") as metadatos_finales:
        for archivo in archivos_pdf:
            diccionario_metadatos_documento = procesar_documento(archivo, llm_global)
            json_metadatos = json.dumps(diccionario_metadatos_documento, ensure_ascii=False)
            metadatos_finales.write(json_metadatos + "\n")

    print(f"\n--- Proceso finalizado ---")
    print(f"Guardado en: {DIR_JSON}")