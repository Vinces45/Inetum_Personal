import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from scripts.sistema_rag.generador_rag import inicializar_bd, inicializar_llm, inicializar_reranker, generar_seccion_nueva, corregir_seccion_existente
from scripts.modelo.pliego import BorradorPliego
# Supongo que en tu esquema Pydantic has añadido "eliminar" como accion permitida
from scripts.modelo.esquemas_pydantic import PlanOrquestador

def analizar_peticion_usuario(peticion, estado_documento, llm):
    print("\n[ORQUESTADOR] Analizando intencion del usuario con Pydantic...")
    
    llm_estructurado = llm.with_structured_output(PlanOrquestador)
    
    secciones_existentes = list(estado_documento.secciones.keys())
    
    prompt_orquestador = """
    Eres el Orquestador de una IA legal del Gobierno de La Rioja.
    Analiza la peticion del usuario y extrae la informacion solicitada.
    
    SECCIONES ACTUALMENTE EN EL BORRADOR (Usa solo estos nombres exactos si la accion es 'modificar' o 'eliminar'):
    {secciones_actuales}
    
    INSTRUCCIONES CLAVE:
    1. Determina la ACCION principal: "crear", "modificar" o "eliminar".
    2. Extrae los metadatos de la peticion del usuario para rellenar los filtros.
    3. SI EL USUARIO NO MENCIONA UN DATO ESPECIFICO, DEJALO VACIO (null). NO TE INVENTES DATOS.
    
    PETICION DEL USUARIO: "{peticion}"
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_orquestador)
    cadena = prompt | llm_estructurado
    
    try:
        plan_obj = cadena.invoke({
            "peticion": peticion, 
            "secciones_actuales": secciones_existentes
        })
        
        if not plan_obj: return None
        
        plan_dict = plan_obj.model_dump(exclude_none=True)
        
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
    modelo_reranker = inicializar_reranker()
    
    # Usamos un LLM dedicado pero persistente para el orquestador (mas rapido y no lo recargamos en cada peticion)
    modelo_llm_orquestador = ChatOllama(model="llama3.1", temperature=0.0)
    
    documento_en_progreso = BorradorPliego()
    
    print("\n" + "="*50)
    print("SISTEMA GENERADOR DE PLIEGOS - GOBIERNO DE LA RIOJA")
    print("="*50)
    print("Escribe tu peticion (ej. 'Genera un PCAP de Servicios con objeto y penalizaciones')")
    print("O pide cambios (ej. 'Modifica las penalizaciones para que sean del 10%')")
    print("O elimina algo (ej. 'Elimina la seccion de penalizaciones')")
    print("Escribe 'ver' para mostrar el documento o 'salir' para terminar.")
    
    while True:
        peticion_usuario = input("\nUsuario: ")
        
        if peticion_usuario.lower() == "salir": 
            break
        elif peticion_usuario.lower() == "ver":
            print(documento_en_progreso.mostrar_documento())
            continue
            
        # Pasamos el modelo_llm_orquestador como argumento
        plan_accion = analizar_peticion_usuario(peticion_usuario, documento_en_progreso, modelo_llm_orquestador)
        
        if not plan_accion:
            continue
            
        print(f"\n[PLAN ESTRATEGICO] {json.dumps(plan_accion, indent=2)}")
        
        accion = plan_accion.get("accion")
        filtros = plan_accion.get("filtros")
        
        if accion == "crear":
            secciones_plan = plan_accion.get("secciones_a_crear", [])
            print(f"\n[SISTEMA] Iniciando redaccion de {len(secciones_plan)} secciones raiz.")
            
            for seccion_obj in secciones_plan:
                titulo_sec = seccion_obj.get("titulo")
                instruccion_sec = seccion_obj.get("instruccion_especifica")
                subsecciones = seccion_obj.get("subsecciones", [])
                
                print(f"\n--- Procesando Seccion: {titulo_sec} ---")
                
                # 1. RAG para la seccion principal (si tiene instrucciones)
                if instruccion_sec:
                    prompt_rag = f"Redacta la seccion '{titulo_sec}'. Instrucciones: {instruccion_sec}"
                    borrador = generar_seccion_nueva(db_vectorial, modelo_llm_rag, modelo_reranker, prompt_rag, filtros)
                    documento_en_progreso.actualizar_seccion(titulo_principal=titulo_sec, contenido=borrador)
                else:
                    # Si no hay texto principal, creamos el titulo vacio para anidar las subsecciones
                    documento_en_progreso.actualizar_seccion(titulo_principal=titulo_sec, contenido="")

                # 2. RAG para cada subseccion (si existen)
                if subsecciones:
                    for sub_obj in subsecciones:
                        titulo_sub = sub_obj.get("titulo")
                        instruccion_sub = sub_obj.get("instruccion_especifica")
                        
                        print(f"  -> Generando subseccion: {titulo_sub}")
                        prompt_rag_sub = f"Redacta la subseccion '{titulo_sub}' de la seccion '{titulo_sec}'. Instrucciones: {instruccion_sub}"
                        
                        borrador_sub = generar_seccion_nueva(db_vectorial, modelo_llm_rag, modelo_reranker, prompt_rag_sub, filtros)
                        documento_en_progreso.actualizar_seccion(
                            titulo_principal=titulo_sec, 
                            contenido=borrador_sub, 
                            titulo_subseccion=titulo_sub
                        )
                        
            print("\n[EXITO] Generacion modular completada. Escribe 'ver' para leer el documento.")
            
        elif accion == "modificar":
            seccion_objetivo = plan_accion.get("seccion_a_modificar")
            feedback = plan_accion.get("feedback")
            
            # Nota: Por ahora obtener_seccion solo busca en el primer nivel (secciones principales)
            texto_actual = documento_en_progreso.obtener_seccion(seccion_objetivo)
            
            if not texto_actual:
                print(f"\n[AVISO] No se ha encontrado la seccion principal '{seccion_objetivo}' en el borrador.")
            else:
                print(f"\n[SISTEMA] Aplicando correcciones a '{seccion_objetivo}'...")
                # texto_corregido = corregir_seccion_existente(db_vectorial, modelo_llm_rag, modelo_reranker, texto_actual, feedback, filtros)
                # documento_en_progreso.actualizar_seccion(seccion_objetivo, texto_corregido)
                print("\n[EXITO] Seccion actualizada.")
                
        elif accion == "eliminar":
            seccion_objetivo = plan_accion.get("seccion_a_eliminar")
            
            if not seccion_objetivo:
                print("\n[AVISO] El sistema no pudo identificar que seccion quieres eliminar.")
            else:
                exito = documento_en_progreso.eliminar_seccion(seccion_objetivo)
                if exito:
                    print(f"\n[EXITO] La seccion '{seccion_objetivo}' ha sido borrada. Escribe 'ver' para comprobar el documento.")
                else:
                    print(f"\n[AVISO] No se ha encontrado la seccion '{seccion_objetivo}' en el borrador.")
                    print("Secciones disponibles para borrar:", list(documento_en_progreso.secciones.keys()))