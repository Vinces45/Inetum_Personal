import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from generador_rag import inicializar_bd, inicializar_llm, generar_seccion_nueva, corregir_seccion_existente, BorradorPliego
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# def analizar_peticion_usuario(peticion, estado_documento):
#     print("\n[ORQUESTADOR] Analizando intencion del usuario...")
    
#     # Llama 3 configurado para JSON
#     llm_orquestador = ChatOllama(model="llama3", temperature=0.0, format="json")
    
#     secciones_existentes = list(estado_documento.secciones.keys())
    
#     # PROMPT ACTUALIZADO: Reflejando los metadatos reales y forzando nulls
#     prompt_orquestador = """
#     Eres el Orquestador de una IA legal del Gobierno de La Rioja.
#     Tu UNICA tarea es analizar la peticion y devolver un objeto JSON valido.
#     NO escribas introducciones, NO escribas saludos, SOLO el JSON.
    
#     SECCIONES EN EL BORRADOR: {secciones_actuales}
    
#     REGLAS DE EXTRACCION PARA EL JSON:
#     1. "accion": "crear" (para texto nuevo) o "modificar" (para cambiar el borrador).
#     2. "secciones_a_crear": Lista de nombres (solo si es crear).
#     3. "seccion_a_modificar": Nombre de la seccion (solo si es modificar). Debe coincidir EXACTAMENTE con un nombre de las SECCIONES EN EL BORRADOR.
#     4. "feedback": Instruccion de cambio (solo si es modificar).
#     5. "filtros": Extrae las condiciones legales explicitamente mencionadas en la peticion. SI EL USUARIO NO MENCIONA UN DATO, SU VALOR DEBE SER null OBLIGATORIAMENTE.
#        - "tipo_contrato": (Obras, Servicios o Suministros)
#        - "tipo_documento": (PCAP o PPT)
#        - "tramitacion": (Ordinaria, Urgente, Emergencia)
#        - "procedimiento": (Abierto, Menor, Negociado)
#        - "lotes": true (si exige lotes) o false (si exige lote unico). Si no dice nada, null.
#        - "presupuesto_base_licitacion": numero entero (ej: 150000). Si no dice nada, null.
#        - "plazo_ejecucion_meses": numero entero de meses. Si no dice nada, null.
#        - "valor_estimado_contrato": el costo total del proyecto incluyendo cualquier gasto. Si no dice nada, null
#        - "iva": el porcentaje de iva que hay en el proyectom solo el número sin el símbolo del porcentaje

#     FORMATO EXACTO ESPERADO:
#     {{
#         "accion": "crear",
#         "secciones_a_crear": ["Objeto del contrato"],
#         "seccion_a_modificar": null,
#         "feedback": null,
#         "filtros": {{
#             "tipo_contrato": "Servicios",
#             "tipo_documento": "PCAP",
#             "tramitacion": null,
#             "procedimiento": null,
#             "lotes": true,
#             "presupuesto_base_licitacion": 150000,
#             "plazo_ejecucion_meses": null
#         }}
#     }}

#     PETICION DEL USUARIO: "{peticion}"
#     """
    
#     prompt = ChatPromptTemplate.from_template(prompt_orquestador)
#     cadena = prompt | llm_orquestador | JsonOutputParser()
    
#     try:
#         plan = cadena.invoke({
#             "peticion": peticion, 
#             "secciones_actuales": secciones_existentes
#         })
#         # Limpieza de nulls
#         if plan.get("filtros"):
#             filtros_limpios = {k: v for k, v in plan["filtros"].items() if v is not None}
#             plan["filtros"] = filtros_limpios if filtros_limpios else None
            
#         return plan
        
#     except Exception as e:
#         print(f"[ERROR] El Orquestador fallo al crear el JSON: {e}")
#         return None

# ==========================================
# 1. ESQUEMAS PYDANTIC PARA EL ORQUESTADOR
# ==========================================

class FiltrosMetadatos(BaseModel):
    presupuesto_base_licitacion: Optional[float] = Field(
        default=None, description="Importe neto del presupuesto base en euros (ej: 150000.50)."
    )
    valor_estimado_contrato: Optional[float] = Field(
        default=None, description="Valor estimado total del contrato en euros."
    )
    iva_porcentaje: Optional[int] = Field(
        default=None, description="Porcentaje de IVA (ej: 21 o 10)."
    )
    plazo_ejecucion_meses: Optional[int] = Field(
        default=None, description="Plazo de ejecucion expresado en meses."
    )
    tiempo_prorroga_meses: Optional[int] = Field(
        default=None, description="Tiempo de prorroga permitido en meses."
    )
    tipo_contrato: Optional[Literal["Obras", "Servicios", "Suministros"]] = Field(
        default=None, description="Tipo principal del contrato."
    )
    tramitacion: Optional[Literal["Ordinaria", "Urgente", "Emergencia"]] = Field(
        default=None, description="Velocidad o tipo de tramitacion del expediente."
    )
    procedimiento: Optional[Literal["Abierto", "Menor", "Negociado"]] = Field(
        default=None, description="Procedimiento de adjudicacion."
    )
    lotes: Optional[bool] = Field(
        default=None, description="True si el contrato esta dividido en lotes, False si es lote unico."
    )
    financiacion_europea: Optional[bool] = Field(
        default=None, description="True si cuenta con financiacion de fondos europeos (FEDER, Next Generation), False si no."
    )

class PlanOrquestador(BaseModel):
    accion: Literal["crear", "modificar"] = Field(
        description="Si el usuario pide algo nuevo es 'crear'. Si pide cambiar algo existente es 'modificar'."
    )
    secciones_a_crear: Optional[List[str]] = Field(
        default=None, description="Lista de nombres de secciones a redactar."
    )
    seccion_a_modificar: Optional[str] = Field(
        default=None, description="Nombre EXACTO de la seccion a cambiar (debe existir en el borrador)."
    )
    feedback: Optional[str] = Field(
        default=None, description="La instruccion de lo que hay que cambiar en la seccion."
    )
    filtros: Optional[FiltrosMetadatos] = Field(
        default=None, description="Filtros extraidos de la peticion del usuario."
    )

# ==========================================
# 2. FUNCION DEL ORQUESTADOR ACTUALIZADA
# ==========================================

def analizar_peticion_usuario(peticion, estado_documento):
    print("\n[ORQUESTADOR] Analizando intencion del usuario con Pydantic...")
    
    # Instanciamos el modelo y le acoplamos el esquema Pydantic
    llm = ChatOllama(model="llama3", temperature=0.0)
    llm_estructurado = llm.with_structured_output(PlanOrquestador)
    
    secciones_existentes = list(estado_documento.secciones.keys())
    
    # El prompt ahora es muchisimo mas limpio gracias a Pydantic
    prompt_orquestador = """
    Eres el Orquestador de una IA legal del Gobierno de La Rioja.
    Analiza la peticion del usuario y extrae la informacion solicitada.
    
    SECCIONES ACTUALMENTE EN EL BORRADOR (Usa solo estos nombres exactos si la accion es 'modificar'):
    {secciones_actuales}
    
    INSTRUCCIONES CLAVE:
    1. Extrae los metadatos de la peticion del usuario para rellenar los filtros.
    2. SI EL USUARIO NO MENCIONA UN DATO ESPECIFICO (por ejemplo, no dice el presupuesto o si hay lotes), DEJALO VACIO (null). NO TE INVENTES DATOS.
    
    PETICION DEL USUARIO: "{peticion}"
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_orquestador)
    cadena = prompt | llm_estructurado
    
    try:
        # Pydantic devuelve un objeto Python real tipado
        plan_obj = cadena.invoke({
            "peticion": peticion, 
            "secciones_actuales": secciones_existentes
        })
        
        if not plan_obj: return None
        
        # Convertimos el objeto a diccionario para el resto de tu codigo
        plan_dict = plan_obj.model_dump()
        
        # Limpiamos los filtros que sean None para no enviarlos a ChromaDB y romper la busqueda
        if plan_dict.get("filtros"):
            filtros_limpios = {k: v for k, v in plan_dict["filtros"].items() if v is not None}
            plan_dict["filtros"] = filtros_limpios if filtros_limpios else None
            
        return plan_dict
        
    except Exception as e:
        print(f"[ERROR] El Orquestador fallo al estructurar la peticion: {e}")
        return None

if __name__ == "__main__":
    print("Arrancando el cerebro del sistema...")
    db_vectorial = inicializar_bd()
    modelo_llm_rag = inicializar_llm()
    
    # Instanciamos la memoria del documento
    documento_en_progreso = BorradorPliego()
    
    print("\n" + "="*50)
    print("SISTEMA GENERADOR DE PLIEGOS - GOBIERNO DE LA RIOJA")
    print("="*50)
    print("Escribe tu peticion (ej. 'Genera un PCAP de Servicios con objeto y penalizaciones')")
    print("O pide cambios (ej. 'Modifica las penalizaciones para que sean del 10%')")
    print("Escribe 'ver' para mostrar el documento o 'salir' para terminar.")
    
    while True:
        peticion_usuario = input("\nUsuario: ")
        
        if peticion_usuario.lower() == "salir": 
            break
        elif peticion_usuario.lower() == "ver":
            print(documento_en_progreso.mostrar_documento())
            continue
            
        # PASO 1: Analizar intencion con el LLM
        plan_accion = analizar_peticion_usuario(peticion_usuario, documento_en_progreso)
        
        if not plan_accion:
            continue
            
        print(f"\n[PLAN ESTRATEGICO] {json.dumps(plan_accion, indent=2)}")
        
        accion = plan_accion.get("accion")
        filtros = plan_accion.get("filtros")
        
        # PASO 2: Ejecutar la rama correspondiente (Crear o Modificar)
        if accion == "crear":
            secciones = plan_accion.get("secciones_a_crear", [])
            print(f"\n[SISTEMA] Iniciando redaccion de {len(secciones)} secciones con filtros: {filtros}")
            
            for seccion in secciones:
                print(f"\n--- Redactando: {seccion} ---")
                instruccion = f"Redacta el apartado de: {seccion}"
                borrador = generar_seccion_nueva(db_vectorial, modelo_llm_rag, instruccion, filtros)
                # Para versión sin filtros
                #borrador = generar_seccion_nueva(db_vectorial, modelo_llm_rag, instruccion)
                documento_en_progreso.actualizar_seccion(seccion, borrador)
                
            print("\n[EXITO] Secciones generadas. Escribe 'ver' para leer el documento.")
            
        elif accion == "modificar":
            seccion_objetivo = plan_accion.get("seccion_a_modificar")
            feedback = plan_accion.get("feedback")
            
            texto_actual = documento_en_progreso.obtener_seccion(seccion_objetivo)
            
            if not texto_actual:
                print(f"\n[AVISO] No se ha encontrado la seccion '{seccion_objetivo}' en el borrador actual.")
                print("Secciones disponibles:", list(documento_en_progreso.secciones.keys()))
            else:
                print(f"\n[SISTEMA] Aplicando correcciones a la seccion '{seccion_objetivo}'...")
                texto_corregido = corregir_seccion_existente(db_vectorial, modelo_llm_rag, texto_actual, feedback, filtros)
                documento_en_progreso.actualizar_seccion(seccion_objetivo, texto_corregido)
                print("\n[EXITO] Seccion actualizada. Escribe 'ver' para revisar el cambio.")