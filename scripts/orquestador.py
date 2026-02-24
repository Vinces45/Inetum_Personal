import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Importamos tu motor RAG y el gestor de estado (BorradorPliego)
from generador_rag import inicializar_bd, inicializar_llm, generar_seccion_nueva, corregir_seccion_existente, BorradorPliego

def analizar_peticion_usuario(peticion, estado_documento):
    """
    Fase 1: Enrutador Inteligente
    Evalua si el usuario quiere crear algo nuevo o modificar el borrador actual.
    """
    print("\n[ORQUESTADOR] Analizando intencion del usuario...")
    llm_orquestador = ChatOllama(model="llama3", temperature=0.0, format="json")
    
    # Le pasamos al LLM las secciones que ya existen para que sepa de que le habla el usuario
    secciones_existentes = list(estado_documento.secciones.keys())
    
    prompt_orquestador = """
    Eres el Orquestador del sistema de pliegos del Gobierno de La Rioja.
    Tu tarea es leer la peticion del usuario y generar UNICAMENTE un JSON valido.
    
    SECCIONES ACTUALMENTE EN EL BORRADOR: {secciones_actuales}
    
    REGLAS PARA EL JSON:
    - "accion": Debe ser "crear" (si pide redactar algo nuevo) o "modificar" (si pide cambiar una seccion existente).
    - "secciones_a_crear": Lista de nombres de secciones a generar (solo si accion es "crear").
    - "seccion_a_modificar": Nombre de la seccion que quiere cambiar (solo si accion es "modificar").
    - "feedback": Lo que el usuario quiere cambiar (solo si accion es "modificar").
    - "filtros": Extrae "tipo_contrato" (Obras, Servicios o Suministros) y "tipo_documento" (PCAP o PPT). Usa null si no se mencionan.

    FORMATO EXACTO ESPERADO:
    {{
        "accion": "crear",
        "secciones_a_crear": ["Objeto del contrato", "Penalidades"],
        "seccion_a_modificar": null,
        "feedback": null,
        "filtros": {{
            "tipo_contrato": "Servicios",
            "tipo_documento": "PCAP"
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
        
        # Limpiamos los filtros null
        if "filtros" in plan and plan["filtros"]:
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