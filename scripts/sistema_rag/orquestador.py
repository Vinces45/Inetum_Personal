import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from scripts.sistema_rag.generador_rag import inicializar_bd, inicializar_llm, generar_seccion_nueva, corregir_seccion_existente
from scripts.modelo.pliego import BorradorPliego
from scripts.modelo.esquemas_pydantic import PlanOrquestador


def analizar_peticion_usuario(peticion, estado_documento):
    print("\n[ORQUESTADOR] Analizando intencion del usuario con Pydantic...")
    
    llm = ChatOllama(model="llama3.1", temperature=0.0)
    llm_estructurado = llm.with_structured_output(PlanOrquestador)
    
    secciones_existentes = list(estado_documento.secciones.keys())
    
    # SERA NECESARIO MODIFICARLO PARA QUE TENGA EN CUENTA LAS NUEVAS FUNCIONALIDADES
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
        
        # Convertimos el objeto a diccionario excluyendo directamente los nulos (Magia de Pydantic)
        plan_dict = plan_obj.model_dump(exclude_none=True)
        
        # Como usamos exclude_none, si el diccionario de filtros se queda vacio {}, lo pasamos a None
        if "filtros" in plan_dict and not plan_dict["filtros"]:
            plan_dict["filtros"] = None
            
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