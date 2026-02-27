import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from generador_rag import inicializar_bd, inicializar_llm, generar_seccion_nueva, corregir_seccion_existente, BorradorPliego

def analizar_peticion_usuario(peticion, estado_documento):
    print("\n[ORQUESTADOR] Analizando intencion del usuario...")
    
    # Llama 3 configurado para JSON
    llm_orquestador = ChatOllama(model="llama3", temperature=0.0, format="json")
    
    secciones_existentes = list(estado_documento.secciones.keys())
    
    # PROMPT MEJORADO: Más restrictivo y con todos tus filtros
    prompt_orquestador = """
    Eres el Orquestador de una IA legal del Gobierno de La Rioja.
    Tu UNICA tarea es analizar la peticion y devolver un objeto JSON valido.
    NO escribas introducciones, NO escribas saludos, SOLO el JSON.
    
    SECCIONES EN EL BORRADOR: {secciones_actuales}
    
    REGLAS DE EXTRACCIÓN PARA EL JSON:
    1. "accion": "crear" (para texto nuevo) o "modificar" (para cambiar el borrador).
    2. "secciones_a_crear": Lista de nombres (solo si es crear).
    3. "seccion_a_modificar": Nombre de la seccion (solo si es modificar).
    4. "feedback": Instruccion de cambio (solo si es modificar).
    5. "filtros": Extrae las condiciones legales mencionadas. SI NO SE MENCIONAN, USA null.
       - "tipo_contrato": (Obras, Servicios o Suministros)
       - "tipo_documento": (PCAP o PPT)
       - "tramitacion": (Ordinaria, Urgente, Emergencia)
       - "procedimiento": (Abierto, Menor, Negociado)
       - "lotes": (true si menciona lotes, false si dice sin lotes)

    FORMATO EXACTO ESPERADO:
    {{
        "accion": "crear",
        "secciones_a_crear": ["Objeto del contrato"],
        "seccion_a_modificar": null,
        "feedback": null,
        "filtros": {{
            "tipo_contrato": "Servicios",
            "tipo_documento": "PCAP",
            "tramitacion": "Ordinaria",
            "procedimiento": null,
            "lotes": false
        }}
    }}

    PETICION DEL USUARIO: "{peticion}"
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_orquestador)
    cadena = prompt | llm_orquestador | JsonOutputParser()
    
    try:
        plan = cadena.invoke({
            "peticion": peticion, 
            "secciones_actuales": secciones_existentes
        })
        # Limpieza de nulls
        if plan.get("filtros"):
            filtros_limpios = {k: v for k, v in plan["filtros"].items() if v is not None}
            plan["filtros"] = filtros_limpios if filtros_limpios else None
            
        return plan
        
    except Exception as e:
        print(f"[ERROR] El Orquestador fallo al crear el JSON: {e}")
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