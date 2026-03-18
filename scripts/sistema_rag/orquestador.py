import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from scripts.sistema_rag.generador_rag import inicializar_bd, inicializar_llm, inicializar_reranker, generar_seccion_nueva, corregir_seccion_existente, recolectar_texto_rama, resumir_seccion, consultar_duda_legal
from scripts.modelo.pliego import BorradorPliego
from scripts.modelo.esquemas_pydantic import PlanOrquestador

def analizar_peticion_usuario(peticion, estado_documento, llm):
    print("\n[ORQUESTADOR] Analizando intencion del usuario con Pydantic...")
    
    llm_estructurado = llm.with_structured_output(PlanOrquestador)
    
    # Obtenemos las rutas completas para evitar colisiones de nombres
    secciones_existentes = estado_documento.obtener_rutas_secciones()
    
    prompt_orquestador = """
    Eres el Orquestador de una IA legal del Gobierno de La Rioja.
    Analiza la peticion del usuario y extrae la informacion para crear, modificar, eliminar, resumir o consultar.
    
    SECCIONES ACTUALMENTE EN EL BORRADOR:
    {secciones_actuales}
    
    INSTRUCCIONES CRITICAS:
    1. Determina la ACCION principal: "crear", "modificar", "eliminar", "resumir" o "consultar".
    2. SI LA ACCION ES 'crear', ES OBLIGATORIO RELLENAR LA LISTA 'secciones_a_crear'.
    3. SI LA ACCION ES 'modificar', 'eliminar' o 'resumir', copia la RUTA EXACTA de la lista superior. Si pide resumir TODO el documento, deja la ruta en null. 
    4. SI LA ACCION ES 'consultar', significa que el usuario tiene una duda legal teorica. Extrae esa duda exacta en el campo 'pregunta_legal'.
    5. Extrae los metadatos de la peticion del usuario para rellenar los filtros. Si no menciona un dato, dejalo en null.

  
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
    modelo_llm_orquestador = ChatOllama(model="llama3.1", temperature=0.0)
    
    documento_en_progreso = BorradorPliego()
    
    print("\n" + "="*50)
    print("SISTEMA GENERADOR DE PLIEGOS - GOBIERNO DE LA RIOJA")
    print("="*50)
    print("Escribe 'ver' para mostrar el documento o 'salir' para terminar.")
    
    while True:
        peticion_usuario = input("\nUsuario: ")
        
        if peticion_usuario.lower() == "salir": 
            break
        elif peticion_usuario.lower() == "ver":
            print(documento_en_progreso.mostrar_documento())
            continue
            
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
                
                # Definimos la ruta de la seccion principal como una lista de 1 elemento
                ruta_principal = [titulo_sec]
                
                if instruccion_sec or not subsecciones:
                    instruccion_final = instruccion_sec if instruccion_sec else f"Redacta el contenido de {titulo_sec}"
                    prompt_rag = f"Redacta la seccion '{titulo_sec}'. Instrucciones: {instruccion_final}"
                    
                    borrador = generar_seccion_nueva(db_vectorial, modelo_llm_rag, modelo_reranker, prompt_rag, filtros)
                    
                    # NUEVO: Usamos el metodo infinito pasandole la lista
                    documento_en_progreso.actualizar_seccion_infinita(ruta_titulos=ruta_principal, contenido=borrador)
                else:
                    documento_en_progreso.actualizar_seccion_infinita(ruta_titulos=ruta_principal, contenido="")

                if subsecciones:
                    for sub_obj in subsecciones:
                        titulo_sub = sub_obj.get("titulo")
                        instruccion_sub = sub_obj.get("instruccion_especifica") or f"Redacta {titulo_sub}"
                        
                        print(f"  -> Generando subseccion: {titulo_sub}")
                        prompt_rag_sub = f"Redacta la subseccion '{titulo_sub}' de la seccion '{titulo_sec}'. Instrucciones: {instruccion_sub}"
                        
                        borrador_sub = generar_seccion_nueva(db_vectorial, modelo_llm_rag, modelo_reranker, prompt_rag_sub, filtros)
                        
                        # NUEVO: Construimos la ruta anidada como una lista de 2 elementos
                        ruta_hijo = [titulo_sec, titulo_sub]
                        documento_en_progreso.actualizar_seccion_infinita(ruta_titulos=ruta_hijo, contenido=borrador_sub)
                        
            print("\n[EXITO] Generacion modular completada. Escribe 'ver' para leer el documento.")
            
        elif accion == "modificar":
            seccion_objetivo = plan_accion.get("seccion_a_modificar")
            feedback = plan_accion.get("feedback")
            
            if not seccion_objetivo or not feedback:
                print("\n[AVISO] Faltan datos. Asegurate de decirme que seccion quieres cambiar y que quieres que haga exactamente.")
                continue
            
            # Buscamos usando la ruta completa
            texto_actual = documento_en_progreso.buscar_texto_por_ruta(seccion_objetivo)
            
            if not texto_actual:
                print(f"\n[AVISO] No se ha encontrado la seccion o subseccion '{seccion_objetivo}'.")
            else:
                print(f"\n[SISTEMA] Aplicando correcciones a '{seccion_objetivo}' mediante RAG...")
                
                texto_corregido = corregir_seccion_existente(
                    vector_db=db_vectorial, 
                    llm=modelo_llm_rag, 
                    modelo_reranker=modelo_reranker, 
                    titulo_seccion=seccion_objetivo, 
                    texto_actual=texto_actual, 
                    feedback_usuario=feedback, 
                    filtros=filtros
                )

                # NUEVO: Convertimos el string "Padre > Hijo" en una lista ["Padre", "Hijo"]
                ruta_lista = [t.strip() for t in seccion_objetivo.split(">")]
                
                # Usamos el metodo infinito para que lo guarde exactamente en su lugar del arbol
                documento_en_progreso.actualizar_seccion_infinita(ruta_titulos=ruta_lista, contenido=texto_corregido) 
                
                print("\n[EXITO] Seccion actualizada. Escribe 'ver' para revisar el cambio.")
                
        elif accion == "eliminar":
            seccion_objetivo = plan_accion.get("seccion_a_eliminar")
            
            if not seccion_objetivo:
                print("\n[AVISO] El sistema no pudo identificar que seccion quieres eliminar.")
            else:
                # Ahora usamos el nuevo metodo pasandole la ruta literal "Padre > Hijo"
                exito = documento_en_progreso.eliminar_por_ruta(seccion_objetivo)
                
                if exito:
                    print(f"\n[EXITO] Se ha borrado correctamente: '{seccion_objetivo}'. Escribe 'ver' para comprobarlo.")
                else:
                    print(f"\n[AVISO] No se ha encontrado la ruta '{seccion_objetivo}' en el borrador.")

        elif accion == "resumir":
            seccion_objetivo = plan_accion.get("seccion_a_resumir")
            
            print("\n" + "*"*50)
            print("GENERANDO RESUMEN EJECUTIVO...")
            print("*"*50)
            
            if seccion_objetivo:
                # Opcion A: El usuario solo quiere resumir una ruta concreta
                titulos = [t.strip() for t in seccion_objetivo.split(">")]
                
                # Navegamos hasta el nodo
                nodo_actual = documento_en_progreso.secciones.get(titulos[0])
                for tit in titulos[1:]:
                    if nodo_actual and tit in nodo_actual.subsecciones:
                        nodo_actual = nodo_actual.subsecciones[tit]
                    else:
                        nodo_actual = None
                        
                if nodo_actual:
                    texto_rama = recolectar_texto_rama(nodo_actual)
                    resumen = resumir_seccion(modelo_llm_rag, seccion_objetivo, texto_rama)
                    print(f"\n# RESUMEN DE: {seccion_objetivo.upper()}\n{resumen}")
                else:
                    print(f"[AVISO] No se encontro la ruta '{seccion_objetivo}' para resumir.")
                    
            else:
                # Opcion B: El usuario quiere resumir TODO el pliego (Map-Reduce por raices)
                if not documento_en_progreso.secciones:
                    print("[AVISO] El documento esta vacio.")
                else:
                    for titulo_prin, nodo_prin in documento_en_progreso.secciones.items():
                        print(f"\n[SISTEMA] Analizando rama: {titulo_prin}...")
                        texto_rama = recolectar_texto_rama(nodo_prin)
                        resumen = resumir_seccion(modelo_llm_rag, titulo_prin, texto_rama)
                        
                        print(f"\n# {titulo_prin.upper()}\n{resumen}")
            
            print("\n" + "*"*50)

        elif accion == "consultar":
            duda = plan_accion.get("pregunta_legal")
            
            if not duda:
                print("\n[AVISO] No he entendido bien tu pregunta legal. ¿Puedes reformularla?")
                continue
                
            print(f"\n[CONSULTOR LEGAL] Buscando respuesta en la normativa para: '{duda}'...")
            
            # Llamamos a la nueva funcion sin tocar el documento_en_progreso
            respuesta = consultar_duda_legal(
                vector_db=db_vectorial, 
                llm=modelo_llm_rag, 
                modelo_reranker=modelo_reranker, 
                pregunta=duda, 
                filtros=filtros
            )
            
            print("\n" + "-"*50)
            print("DICTAMEN JURIDICO:")
            print("-"*50)
            print(respuesta)
            print("-"*50)