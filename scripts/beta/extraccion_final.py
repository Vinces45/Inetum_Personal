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
# 1. MICRO-ESQUEMAS PYDANTIC
# ==========================================

class ExtraccionDinero(BaseModel):
    valor: Optional[float] = Field(
        default=None, 
        description="El valor economico extraido. Usa formato ingles (ej. 1500000.50). Si no se encuentra, no devuelvas nada."
    )

class ExtraccionPlazo(BaseModel):
    meses: Optional[int] = Field(
        default=None, 
        description="El plazo extraido en meses (numero entero). Si no se encuentra, no devuelvas nada."
    )

class ExtraccionCategoria(BaseModel):
    resultado: Optional[str] = Field(
        default=None, 
        description="La opcion exacta extraida de la lista permitida. Si no esta claro, no devuelvas nada."
    )

class ExtraccionBooleano(BaseModel):
    resultado: Optional[bool] = Field(
        default=None, 
        description="True si se cumple la condicion, False si no se menciona o no se cumple."
    )

# ==========================================
# 2. FUNCIONES DE CONTEXTO
# ==========================================

def limpiar_texto_basico(texto):
    return " ".join(texto.split())

def parsear_moneda_espanola(valor_str):
    if not valor_str:
        return None
    # Eliminamos el simbolo de euro y espacios
    limpio = valor_str.replace("€", "").strip()
    # Eliminamos los puntos de los miles
    limpio = limpio.replace(".", "")
    # Cambiamos la coma decimal por el punto de Python
    limpio = limpio.replace(",", ".")
    try:
        return float(limpio)
    except ValueError:
        return None

def obtener_contexto_relevante(texto_completo, keywords, ventana=800):
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
            recortes.append(f"...[Contexto para {keyword}]...\n{recorte}\n")
            indice += len(keyword)
            
    if not recortes:
        return texto_completo[:3000]
    
    contexto_final = "\n".join(recortes)
    return contexto_final[:8000]

# ==========================================
# 3. EXTRACCIONES ESPECIFICAS (Campo a Campo)
# ==========================================

def auditar_dinero_pydantic(texto_completo_doc, llm, tipo_busqueda):
    print(f"    - Obteniendo {tipo_busqueda}...")
    palabras_clave = ["presupuesto base", "valor estimado", "importe", "excluido", "iva", "licitacion"]
    contexto = obtener_contexto_relevante(texto_completo_doc, palabras_clave, ventana=800)
    
    instruccion = ""
    if tipo_busqueda == "presupuesto_base":
        instruccion = "el PRESUPUESTO BASE DE LICITACION (IMPORTE NETO SIN IVA)."
    elif tipo_busqueda == "valor_estimado":
        instruccion = "el VALOR ESTIMADO DEL CONTRATO (SIN IVA, incluyendo posibles prorrogas)."
    else:
        instruccion = "el PORCENTAJE DE IVA (un numero entero como 21, 10 o 4)."

    llm_estructurado = llm.with_structured_output(ExtraccionDinero)
    prompt = ChatPromptTemplate.from_template("""
    Eres un auditor financiero estricto. Extrae {instruccion}
    CONTEXTO DEL PLIEGO: '''{contexto}'''
    """)
    
    try:
        resultado = (prompt | llm_estructurado).invoke({
            "instruccion": instruccion, 
            "contexto": contexto
        })
        return resultado.valor if resultado else None
    except Exception as e:
        print(f"      [ERROR] {e}")
        return None

def auditar_plazos_pydantic(texto_completo_doc, llm, tipo_busqueda):
    print(f"    - Obteniendo {tipo_busqueda}...")
    palabras_clave = ["plazo de ejecucion", "duracion", "meses", "prorroga", "vigencia"]
    contexto = obtener_contexto_relevante(texto_completo_doc, palabras_clave, ventana=800)
    
    instruccion = "el PLAZO DE EJECUCION INICIAL en meses." if tipo_busqueda == "plazo_ejecucion" else "el PLAZO DE PRORROGA en meses."

    llm_estructurado = llm.with_structured_output(ExtraccionPlazo)
    prompt = ChatPromptTemplate.from_template("""
    Extrae {instruccion}
    CONTEXTO DEL PLIEGO: '''{contexto}'''
    """)
    
    try:
        resultado = (prompt | llm_estructurado).invoke({
            "instruccion": instruccion, 
            "contexto": contexto
        })
        return resultado.meses if resultado else None
    except Exception as e:
        print(f"      [ERROR] {e}")
        return None

def auditar_categoria_pydantic(texto_completo_doc, llm, campo, opciones, palabras_clave, es_booleano=False):
    print(f"    - Obteniendo {campo}...")
    contexto = obtener_contexto_relevante(texto_completo_doc, palabras_clave, ventana=600)
    
    modelo_pydantic = ExtraccionBooleano if es_booleano else ExtraccionCategoria
    llm_estructurado = llm.with_structured_output(modelo_pydantic)
    
    prompt = ChatPromptTemplate.from_template("""
    Analiza el texto y clasifica el contrato para el campo '{campo}'.
    Opciones permitidas: {opciones}
    CONTEXTO DEL PLIEGO: '''{contexto}'''
    """)
    
    try:
        resultado = (prompt | llm_estructurado).invoke({
            "campo": campo,
            "opciones": opciones,
            "contexto": contexto
        })
        return resultado.resultado if resultado else None
    except Exception as e:
        print(f"      [ERROR] {e}")
        return None

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================

def procesar_documento(ruta, llm):
    nombre = os.path.basename(ruta)
    print(f"\nProcesando documento: {nombre}")
    
    datos_finales = {"archivo": nombre}

    try:
        reader = pypdf.PdfReader(ruta)
        texto_completo = "".join([p.extract_text() or "" for p in reader.pages])
        texto_limpio = limpiar_texto_basico(texto_completo)
        
        # 1. Regex (Determinista)
        print("    - Extrayendo Regex (Expediente y CPV)...")
        exp = re.search(r"(\d{2}-\d-\d\.\d{2}-\d{4}/\d{4})", texto_limpio)
        if exp: datos_finales["expediente"] = exp.group(1)
        
        cpvs = re.findall(r"(?i)CPV.*?(\d{8})[-\s]*(\d)", texto_limpio)
        if cpvs: datos_finales["cpv"] = list(set([f"{match[0]}-{match[1]}" for match in cpvs]))

        # 2. Llamadas individuales al LLM con Pydantic
        datos_finales["presupuesto_base_licitacion"] = auditar_dinero_pydantic(texto_limpio, llm, "presupuesto_base")
        datos_finales["valor_estimado_contrato"] = auditar_dinero_pydantic(texto_limpio, llm, "valor_estimado")
        datos_finales["iva_porcentaje"] = auditar_dinero_pydantic(texto_limpio, llm, "iva")
        
        datos_finales["plazo_ejecucion_meses"] = auditar_plazos_pydantic(texto_limpio, llm, "plazo_ejecucion")
        datos_finales["tiempo_prorroga_meses"] = auditar_plazos_pydantic(texto_limpio, llm, "prorroga")
        
        datos_finales["tipo_contrato"] = auditar_categoria_pydantic(
            texto_limpio, llm, "tipo_contrato", ["Obras", "Servicios", "Suministros"], ["contrato de obras", "servicios", "suministros"]
        )
        datos_finales["tramitacion"] = auditar_categoria_pydantic(
            texto_limpio, llm, "tramitacion", ["Ordinaria", "Urgente", "Emergencia"], ["tramitacion", "urgencia", "ordinaria"]
        )
        datos_finales["procedimiento"] = auditar_categoria_pydantic(
            texto_limpio, llm, "procedimiento", ["Abierto", "Abierto Simplificado", "Negociado"], ["procedimiento", "abierto", "simplificado", "adjudicacion"]
        )
        datos_finales["lotes"] = auditar_categoria_pydantic(
            texto_limpio, llm, "lotes", ["True", "False"], ["lotes", "division", "lote unico"], es_booleano=True
        )
        datos_finales["financiacion_europea"] = auditar_categoria_pydantic(
            texto_limpio, llm, "financiacion_europea", ["True", "False"], ["feder", "europea", "next generation", "mecanismo"], es_booleano=True
        )

        # Limpiar Nones
        datos_finales = {k: v for k, v in datos_finales.items() if v is not None}

    except Exception as e:
        print(f"Error procesando {nombre}: {e}")

    return datos_finales

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent      
    CARPETA_SCRIPTS = BASE_DIR.parent               
    PROJECT_ROOT = CARPETA_SCRIPTS.parent           
    
    carpeta_datos = PROJECT_ROOT / "datos" / "pdfs" / "pcap"
    ruta_final_json = PROJECT_ROOT / "datos" / "metadatos" / "metadatos_finales_campo_a_campo_pydantic.json"
    
    ruta_final_json.parent.mkdir(parents=True, exist_ok=True)
    
    if not carpeta_datos.exists():
        print(f"[ERROR] La carpeta {carpeta_datos} no existe.")
        exit(1)
        
    archivos_pdf = list(carpeta_datos.glob("*.pdf"))
    resultados = []
    
    # Inicializamos el LLM una sola vez y lo pasamos por parametro
    print("Inicializando modelo LLM...")
    llm_global = ChatOllama(model="llama3.1", temperature=0.0)
    
    print(f"--- Iniciando Extraccion Campo a Campo con Pydantic ({len(archivos_pdf)} documentos) ---")
    for f in archivos_pdf:
        resultados.append(procesar_documento(f, llm_global))
    
    with open(ruta_final_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- Proceso finalizado ---")
    print(f"Guardado en: {ruta_final_json}")