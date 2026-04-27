import os
import json
from typing import List, Optional
from langchain_core.tools import tool

from scripts.modelo.esquemas_pydantic import PeticionSeccion
from scripts.sistema_rag.generador_rag import (
    corregir_seccion_existente, generar_seccion_nueva, 
    recolectar_texto_rama, resumir_seccion, consultar_duda_legal
)
from scripts.modelo.exportador import generar_bytes_word
from scripts.sistema_rag.ingestor_dinamico import ingestar_documento_individual
from scripts.modelo.esquemas_pydantic import PeticionSeccion, FiltrosBusqueda

RUTA_CONTADOR = "datos/jsons/contador_db.json"

def obtener_siguiente_id():
    """Lee el ultimo ID usado, le suma 1, lo guarda y lo devuelve."""
    # Si el archivo no existe (primera vez), lo creamos con ID 0
    if not os.path.exists(RUTA_CONTADOR):
        os.makedirs(os.path.dirname(RUTA_CONTADOR), exist_ok=True)
        with open(RUTA_CONTADOR, "w", encoding="utf-8") as f:
            json.dump({"ultimo_id": 0}, f)
            
    # Leemos el ID actual
    with open(RUTA_CONTADOR, "r", encoding="utf-8") as f:
        datos = json.load(f)
        
    nuevo_id = datos["ultimo_id"] + 1
    
    # Sobrescribimos con el nuevo ID para el proximo documento
    with open(RUTA_CONTADOR, "w", encoding="utf-8") as f:
        json.dump({"ultimo_id": nuevo_id}, f)
        
    return nuevo_id


def obtener_lista_herramientas(motor):
    """
    Fabrica de herramientas. Recibe el 'motor' inicializado desde Streamlit
    y crea las funciones @tool inyectandoles el contexto del motor.
    """

    @tool
    def herramienta_ver_rutas() -> str:
        """Devuelve las rutas de las secciones actuales. Usala antes de modificar, eliminar o si se pide crear una subseccion en la seccion solicitada."""
        print("\n" + "="*50)
        print("[TOOL CALL] 📍 Ejecutando: herramienta_ver_rutas")
        
        # Calculo
        rutas = motor.documento.obtener_rutas_secciones()
        
        # Salida y formateo
        if rutas:
            respuesta_llm = f"Secciones actuales: {rutas}"
            print(f"[TOOL LOG] 🟢 Datos extraidos: {len(rutas)} rutas encontradas.")
            print(f"[TOOL RETURN] 📤 Enviando al LLM: {respuesta_llm}")
        else:
            respuesta_llm = "El documento esta vacio."
            print("[TOOL LOG] 🟡 Estado: Arbol de nodos vacio.")
            print(f"[TOOL RETURN] 📤 Enviando al LLM: {respuesta_llm}")
            
        print("="*50 + "\n")
        return respuesta_llm

    @tool
    def herramienta_crear_secciones(secciones_a_crear: List[PeticionSeccion], filtros_llm: Optional[FiltrosBusqueda] = None) -> str:
        """
        Crea una o multiples secciones (con o sin subsecciones) en el pliego de forma recursiva (niveles infinitos).
        Usa esta herramienta cuando el usuario pida redactar contenido nuevo.
        """
        print("\n" + "="*50)
        print(f"[TOOL CALL] ✍️ Ejecutando: herramienta_crear_secciones")
        
        # Procesamos los filtros inteligentes directamente desde el Agente
        if filtros_llm:
            filtros_diccionario = filtros_llm.model_dump(exclude_none=True)
            print(f"[TOOL LOG] 🔍 Filtros detectados por el agente: {filtros_diccionario}")
        else:
            filtros_diccionario = None
            print(f"[TOOL LOG] 🔍 No se detectaron metadatos para filtrar.")

        print(f"[TOOL LOG] 🟢 Se han solicitado {len(secciones_a_crear)} secciones principales.")
        
        # Listas para registrar el estado real de cada operacion
        secciones_exito = []
        secciones_vacias = []
        
        # Funcion interna recursiva para procesar N niveles de profundidad
        def procesar_nivel(lista_secciones, ruta_padre=[]):
            for sec in lista_secciones:
                titulo = sec.titulo
                instruccion = sec.instruccion_especifica
                sub_lista = sec.subsecciones or []
                
                # Construimos la ruta completa (ej: ["Criterios", "Oferta Economica"])
                ruta_actual = ruta_padre + [titulo]
                nivel_str = " > ".join(ruta_actual)
                
                print(f"[TOOL LOG] --- Procesando: {nivel_str} ---")
                
                # Solo llamamos al RAG si hay una instruccion especifica o si es un nodo hoja (no tiene hijos)
                if instruccion or not sub_lista:
                    instruccion_final = instruccion if instruccion else f"Redacta el contenido de {titulo}"
                    prompt_rag = f"Redacta la seccion '{titulo}'. Instrucciones: {instruccion_final}"
                    print(f"[TOOL LOG] 🧠 Llamando al RAG para: {titulo}")
                    
                    borrador = generar_seccion_nueva(motor.db, motor.llm, motor.reranker, prompt_rag, filtros=filtros_diccionario)
                    
                    if "Falta contexto legal" in borrador:
                        print(f"[TOOL LOG] ❌ RAG vacio para '{titulo}'. Dejando advertencia y continuando...")
                        motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_actual, contenido="[Seccion vacia: No se encontro contexto suficiente en la BD]")
                        secciones_vacias.append(titulo)
                    else:
                        motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_actual, contenido=borrador)
                        secciones_exito.append(titulo)
                else:
                    # Es un nodo contenedor (padre), lo creamos vacio en la estructura y seguimos bajando
                    motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_actual, contenido="")

                # LLAMADA RECURSIVA: Si tiene hijos, volvemos a llamar a esta misma funcion
                if sub_lista:
                    procesar_nivel(sub_lista, ruta_actual)

        # Iniciamos la recursion pasandole la lista principal que viene del LLM
        procesar_nivel(secciones_a_crear)
        
        # ---------------------------------------------------------
        # BLOQUE DE AUTO-GUARDADO (Transaccion completada)
        # ---------------------------------------------------------
        try:
            motor.documento.guardar_respaldo()
            print("[TOOL LOG] 💾 Auto-guardado ejecutado con exito tras la creacion.")
        except Exception as e:
            print(f"[TOOL LOG] ⚠️ Aviso: Fallo en el auto-guardado: {e}")
        # ---------------------------------------------------------

        # Generamos un reporte dinamico basado en los resultados reales
        respuesta_llm = "Resumen de la ejecucion de la herramienta:\n"
        if secciones_exito:
            respuesta_llm += f"- Se redactaron con exito: {', '.join(secciones_exito)}.\n"
        if secciones_vacias:
            respuesta_llm += f"- NO se pudo redactar por falta de contexto legal en la BD (se inserto aviso vacio): {', '.join(secciones_vacias)}.\n"
            
        print(f"[TOOL RETURN] 📤 Enviando al LLM:\n{respuesta_llm}")
        print("="*50 + "\n")
        
        return respuesta_llm

    # @tool
    # def herramienta_crear_secciones(secciones_a_crear: List[PeticionSeccion], filtros_llm: Optional[FiltrosBusqueda] = None) -> str:
    #     """
    #     Crea una o multiples secciones (con o sin subsecciones) en el pliego.
    #     Usa esta herramienta cuando el usuario pida redactar contenido nuevo.
    #     Si el usuario da pistas sobre el tipo de contrato (obras, servicios) o la urgencia, rellena los filtros_llm.
    #     """
    #     print("\n" + "="*50)
    #     print(f"[TOOL CALL] ✍️ Ejecutando: herramienta_crear_secciones")
        
    #     # Procesamos los filtros inteligentes directamente desde el Agente
    #     if filtros_llm:
    #         filtros_diccionario = filtros_llm.model_dump(exclude_none=True)
    #         print(f"[TOOL LOG] 🔍 Filtros detectados por el agente: {filtros_diccionario}")
    #     else:
    #         filtros_diccionario = None
    #         print(f"[TOOL LOG] 🔍 No se detectaron metadatos para filtrar.")

    #     print(f"[TOOL LOG] 🟢 Se han solicitado {len(secciones_a_crear)} secciones principales.")
        
    #     # NUEVO: Listas para registrar el estado real de cada operacion
    #     secciones_exito = []
    #     secciones_vacias = []
        
    #     for seccion_obj in secciones_a_crear:
    #         titulo_sec = seccion_obj.titulo
    #         instruccion_sec = seccion_obj.instruccion_especifica
    #         subsecciones = seccion_obj.subsecciones or []
            
    #         print(f"[TOOL LOG] --- Procesando Seccion: {titulo_sec} ---")
    #         ruta_principal = [titulo_sec]
            
    #         if instruccion_sec or not subsecciones:
    #             instruccion_final = instruccion_sec if instruccion_sec else f"Redacta el contenido de {titulo_sec}"
    #             prompt_rag = f"Redacta la seccion '{titulo_sec}'. Instrucciones: {instruccion_final}"
    #             print(f"[TOOL LOG] 🧠 Llamando al RAG para: {titulo_sec}")
                
    #             borrador = generar_seccion_nueva(motor.db, motor.llm, motor.reranker, prompt_rag, filtros=filtros_diccionario)
                
    #             if "Falta contexto legal" in borrador:
    #                 print(f"[TOOL LOG] RAG vacio para '{titulo_sec}'. Dejando advertencia y continuando...")
    #                 motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_principal, contenido="[Seccion vacia: No se encontro contexto suficiente en la BD]")
    #                 secciones_vacias.append(titulo_sec)
    #             else:
    #                 motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_principal, contenido=borrador)
    #                 secciones_exito.append(titulo_sec)
    #         else:
    #             motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_principal, contenido="")

    #         if subsecciones:
    #             for sub_obj in subsecciones:
    #                 titulo_sub = sub_obj.titulo
    #                 instruccion_sub = sub_obj.instruccion_especifica or f"Redacta {titulo_sub}"
                    
    #                 print(f"[TOOL LOG]  -> Generando subseccion: {titulo_sub}")
    #                 prompt_rag_sub = f"Redacta la subseccion '{titulo_sub}' de la seccion '{titulo_sec}'. Instrucciones: {instruccion_sub}"
    #                 print(f"[TOOL LOG] 🧠 Llamando al RAG para: {titulo_sub}")
                    
    #                 borrador_sub = generar_seccion_nueva(motor.db, motor.llm, motor.reranker, prompt_rag_sub, filtros=filtros_diccionario)
                    
    #                 if "Falta contexto legal" in borrador_sub:
    #                     print(f"[TOOL LOG] ❌ RAG vacio para '{titulo_sub}'. Dejando advertencia y continuando...")
    #                     ruta_hijo = [titulo_sec, titulo_sub]
    #                     motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_hijo, contenido="[Subseccion vacia: No se encontro contexto suficiente]")
    #                     secciones_vacias.append(titulo_sub)
    #                 else:
    #                     ruta_hijo = [titulo_sec, titulo_sub]
    #                     motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_hijo, contenido=borrador_sub)
    #                     secciones_exito.append(titulo_sub)
                    
    #     # NUEVO: Generamos un reporte dinamico basado en los resultados reales
    #     respuesta_llm = "Resumen de la ejecucion de la herramienta:\n"
    #     if secciones_exito:
    #         respuesta_llm += f"- Se redactaron con exito: {', '.join(secciones_exito)}.\n"
    #     if secciones_vacias:
    #         respuesta_llm += f"- NO se pudo redactar por falta de contexto legal en la BD (se inserto aviso vacio): {', '.join(secciones_vacias)}.\n"
            
    #     print(f"[TOOL RETURN] 📤 Enviando al LLM: {respuesta_llm}")
    #     print("="*50 + "\n")
        
    #     return respuesta_llm


    # @tool
    # def herramienta_modificar_seccion(ruta_exacta: str, instrucciones_cambio: str) -> str:
    #     """
    #     Modifica el contenido de una seccion o subseccion que ya existe en el pliego.
    #     """
    #     print("\n" + "="*50)
    #     print(f"[TOOL CALL] ✏️ Ejecutando: herramienta_modificar_seccion")
    #     print(f"[TOOL LOG] 🎯 Objetivo: '{ruta_exacta}'")
    #     print(f"[TOOL LOG] 🗣️ Peticion: '{instrucciones_cambio}'")
        
    #     texto_actual = motor.documento.buscar_texto_por_ruta(ruta_exacta)
        
    #     if not texto_actual:
    #         # NUEVA LOGICA: Si el LLM falla, le damos las rutas correctas en la propia bofetada
    #         rutas_validas = motor.documento.obtener_rutas_secciones()
    #         respuesta_error = (
    #             f"ERROR CRITICO: La ruta '{ruta_exacta}' no existe. "
    #             f"ESTAS SON LAS UNICAS RUTAS VALIDAS AHORA MISMO: {rutas_validas}. "
    #             f"OBLIGATORIO: Vuelve a ejecutar esta herramienta inmediatamente usando una de las rutas validas."
    #         )
    #         print(f"[TOOL LOG] ❌ Fallo: La ruta es incorrecta. Forzando al LLM a reintentar.")
    #         print(f"[TOOL RETURN] 📤 Enviando error y mapa de rutas al LLM.")
    #         print("="*50 + "\n")
    #         return respuesta_error
            
    #     print(f"[TOOL LOG] 🧠 Llamando al RAG para aplicar las correcciones...")
    #     texto_corregido = corregir_seccion_existente(
    #         vector_db=motor.db, 
    #         llm=motor.llm, 
    #         modelo_reranker=motor.reranker, 
    #         titulo_seccion=ruta_exacta, 
    #         texto_actual=texto_actual, 
    #         feedback_usuario=instrucciones_cambio
    #     )
        
    #     ruta_lista = [t.strip() for t in ruta_exacta.split(">")]
    #     motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_lista, contenido=texto_corregido) 
        
    #     respuesta_exito = f"La seccion '{ruta_exacta}' ha sido modificada con exito."
    #     print(f"[TOOL LOG] ✅ Arbol de nodos actualizado correctamente.")
    #     print(f"[TOOL RETURN] 📤 Enviando al LLM: {respuesta_exito}")
    #     print("="*50 + "\n")
        
    #     return respuesta_exito




    @tool
    def herramienta_modificar_seccion(ruta_exacta: str, instrucciones_cambio: str, filtros_llm: Optional[FiltrosBusqueda] = None) -> str:
        """
        Modifica el contenido de una seccion o subseccion que ya existe en el pliego.
        Si las instrucciones de cambio mencionan tipos de contrato, urgencias o procedimientos, usa filtros_llm.
        """
        print("\n" + "="*50)
        print(f"[TOOL CALL] ✏️ Ejecutando: herramienta_modificar_seccion")
        print(f"[TOOL LOG] 🎯 Objetivo: '{ruta_exacta}'")
        print(f"[TOOL LOG] 🗣️ Peticion: '{instrucciones_cambio}'")
        
        # Procesamos los filtros si el agente los ha deducido
        if filtros_llm:
            filtros_diccionario = filtros_llm.model_dump(exclude_none=True)
            print(f"[TOOL LOG] 🔍 Filtros expandidos por el agente: {filtros_diccionario}")
        else:
            filtros_diccionario = None
        
        texto_actual = motor.documento.buscar_texto_por_ruta(ruta_exacta)
        
        if not texto_actual:
            rutas_validas = motor.documento.obtener_rutas_secciones()
            respuesta_error = (
                f"ERROR CRITICO: La ruta '{ruta_exacta}' no existe. "
                f"ESTAS SON LAS UNICAS RUTAS VALIDAS AHORA MISMO: {rutas_validas}. "
                f"OBLIGATORIO: Vuelve a ejecutar esta herramienta inmediatamente usando una de las rutas validas."
            )
            print(f"[TOOL LOG] ❌ Fallo: La ruta es incorrecta. Forzando al LLM a reintentar.")
            print(f"[TOOL RETURN] 📤 Enviando error y mapa de rutas al LLM.")
            print("="*50 + "\n")
            return respuesta_error
            
        print(f"[TOOL LOG] 🧠 Llamando al RAG para aplicar las correcciones...")
        texto_corregido = corregir_seccion_existente(
            vector_db=motor.db, 
            llm=motor.llm, 
            modelo_reranker=motor.reranker, 
            titulo_seccion=ruta_exacta, 
            texto_actual=texto_actual, 
            feedback_usuario=instrucciones_cambio,
            filtros=filtros_diccionario # <--- PASAMOS LOS FILTROS AQUI
        )
        
        ruta_lista = [t.strip() for t in ruta_exacta.split(">")]
        motor.documento.actualizar_seccion_infinita(ruta_titulos=ruta_lista, contenido=texto_corregido) 
        
        # ---------------------------------------------------------
        # BLOQUE DE AUTO-GUARDADO (Transaccion completada)
        # ---------------------------------------------------------
        try:
            motor.documento.guardar_respaldo()
            print("[TOOL LOG] 💾 Auto-guardado ejecutado con exito tras la modificacion.")
        except Exception as e:
            print(f"[TOOL LOG] ⚠️ Aviso: Fallo en el auto-guardado: {e}")
        # ---------------------------------------------------------
        
        respuesta_exito = f"La seccion '{ruta_exacta}' ha sido modificada con exito."
        print(f"[TOOL LOG] ✅ Arbol de nodos actualizado correctamente.")
        print(f"[TOOL RETURN] 📤 Enviando al LLM: {respuesta_exito}")
        print("="*50 + "\n")
        
        return respuesta_exito




    @tool
    def herramienta_eliminar_secciones(rutas_exactas: List[str]) -> str:
        """
        Elimina una o multiples secciones/subsecciones del documento.
        Usa 'herramienta_ver_rutas' primero para obtener la lista de rutas exactas que debes borrar.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] 🗑️ Ejecutando: herramienta_eliminar_secciones")
        print(f"[TOOL LOG] 🎯 Se ha pedido borrar {len(rutas_exactas)} rutas: {rutas_exactas}")
        
        resultados = []
        hubo_exito = False # Bandera para controlar el auto-guardado
        
        for ruta in rutas_exactas:
            # Usamos tu misma logica defensiva
            exito = motor.documento.eliminar_por_ruta(ruta)
            
            if exito:
                mensaje = f"Exito: '{ruta}' eliminada correctamente."
                resultados.append(mensaje)
                print(f"[TOOL LOG] ✅ {mensaje}")
                hubo_exito = True # Marcamos que hubo al menos un borrado real
            else:
                mensaje = f"Error: No se encontro la ruta '{ruta}'."
                resultados.append(mensaje)
                print(f"[TOOL LOG] ❌ {mensaje}")

        # ---------------------------------------------------------
        # BLOQUE DE AUTO-GUARDADO (Transaccion completada)
        # ---------------------------------------------------------
        if hubo_exito:
            try:
                motor.documento.guardar_respaldo()
                print("[TOOL LOG] 💾 Auto-guardado ejecutado con exito tras la eliminacion.")
            except Exception as e:
                print(f"[TOOL LOG] ⚠️ Aviso: Fallo en el auto-guardado: {e}")
        # ---------------------------------------------------------

        # Juntamos todos los resultados en un solo texto para que el LLM los lea
        respuesta_final = "\n".join(resultados)
        
        print(f"[TOOL RETURN] 📤 Enviando al LLM el reporte de borrado.")
        print("="*50 + "\n")
        
        return respuesta_final

    @tool
    def herramienta_resumir_seccion(ruta_exacta: Optional[str] = None) -> str:
        """
        Genera un resumen de una seccion especifica o de todo el pliego.
        Si el usuario pide resumir TODO el documento, NO le pases ningun argumento (deja ruta_exacta vacio).
        Si pide resumir una parte concreta, pasa la 'ruta_exacta' (ej: '1. Objeto > 1.1. Garantias').
        Usa 'herramienta_ver_rutas' antes si no sabes el nombre exacto de la seccion.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] 📝 Ejecutando: herramienta_resumir_seccion")
        
        # CASO 1: Resumir TODO el documento (ruta_exacta es None o vacio)
        if not ruta_exacta:
            print("[TOOL LOG] 🎯 Objetivo: Resumir TODO el documento por ramas.")
            if not motor.documento.secciones:
                print("[TOOL LOG] ⚠️ El documento esta vacio.")
                print("="*50 + "\n")
                return "El documento esta vacio, no hay nada que resumir."
            
            resumenes_totales = []
            for titulo_prin, nodo_prin in motor.documento.secciones.items():
                print(f"[TOOL LOG] 🧠 Extrayendo y resumiendo rama: {titulo_prin}...")
                texto_rama = recolectar_texto_rama(nodo_prin)
                resumen = resumir_seccion(motor.llm, titulo_prin, texto_rama)
                resumenes_totales.append(f"**Resumen de {titulo_prin}:**\n{resumen}")
                
            respuesta_final = "\n\n".join(resumenes_totales)
            print("[TOOL LOG] ✅ Resumen global completado.")
            print("="*50 + "\n")
            return respuesta_final

        # CASO 2: Resumir una seccion especifica
        print(f"[TOOL LOG] 🎯 Objetivo: Resumir la ruta '{ruta_exacta}'")
        titulos = [t.strip() for t in ruta_exacta.split(">")]
        
        # Navegacion manual del arbol (tu misma logica del orquestador)
        nodo_actual = motor.documento.secciones.get(titulos[0])
        for tit in titulos[1:]:
            if nodo_actual and tit in nodo_actual.subsecciones:
                nodo_actual = nodo_actual.subsecciones[tit]
            else:
                nodo_actual = None
                break # Rompemos el bucle si nos perdemos en el arbol
                
        # Programacion defensiva si la ruta no existe
        if not nodo_actual:
            error_msg = f"Error: No se encontro la ruta '{ruta_exacta}'. Usa herramienta_ver_rutas para comprobar el nombre."
            print(f"[TOOL LOG] ❌ {error_msg}")
            print("="*50 + "\n")
            return error_msg
            
        print(f"[TOOL LOG] 🧠 Extrayendo texto y resumiendo la seccion especifica...")
        texto_rama = recolectar_texto_rama(nodo_actual)
        resumen_especifico = resumir_seccion(motor.llm, ruta_exacta, texto_rama)
        
        respuesta_final = f"**Resumen de {ruta_exacta}:**\n{resumen_especifico}"
        print("[TOOL LOG] ✅ Resumen especifico completado.")
        print("="*50 + "\n")
        
        return respuesta_final

    # @tool
    # def herramienta_consultar_ley(pregunta: str) -> str:
    #     """
    #     Busca informacion legal o normativa en la base de datos RAG de pliegos.
    #     Usala SOLO cuando el usuario haga una pregunta explicita sobre pliegos, leyes, plazos o normativas.
    #     """
    #     print("\n" + "="*50)
    #     print("[TOOL CALL] ⚖️ Ejecutando: herramienta_consultar_ley")
    #     print(f"[TOOL LOG] 📥 Duda legal extraida: '{pregunta}'")
        
    #     print("[TOOL LOG] 🧠 Consultando ChromaDB y generando dictamen con Llama 3.1...")
        
    #     # Llamamos a tu funcion original usando nuestro motor global
    #     respuesta = consultar_duda_legal(
    #         vector_db=motor.db, 
    #         llm=motor.llm, 
    #         modelo_reranker=motor.reranker, 
    #         pregunta=pregunta
    #     )
        
    #     # Hacemos un pequeño truncado solo para la consola, para no ensuciar toda la pantalla
    #     # si la respuesta legal es larguísima. Al LLM se le envia completa.
    #     #resumen_consola = respuesta[:150].replace('\n', ' ') + "..." if len(respuesta) > 150 else respuesta
        
    #     print(f"[TOOL LOG] ✅ Respuesta RAG obtenida: {respuesta}")
    #     print("[TOOL RETURN] 📤 Enviando al LLM para que se la comunique al usuario.")
    #     print("="*50 + "\n")
        
    #     return respuesta



    @tool
    def herramienta_consultar_ley(pregunta: str, filtros_llm: Optional[FiltrosBusqueda] = None) -> str:
        """
        Busca informacion legal o normativa en la base de datos RAG de pliegos.
        Si el usuario da pistas sobre el tipo de contrato, rellenalo en filtros_llm.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] ⚖️ Ejecutando: herramienta_consultar_ley")
        print(f"[TOOL LOG] 📥 Duda legal extraida: '{pregunta}'")
        
        if filtros_llm:
            filtros_diccionario = filtros_llm.model_dump(exclude_none=True)
            print(f"[TOOL LOG] 🔍 Filtros expandidos por el agente: {filtros_diccionario}")
        else:
            filtros_diccionario = None

        print("[TOOL LOG] 🧠 Consultando ChromaDB y generando dictamen...")
        
        respuesta = consultar_duda_legal(
            vector_db=motor.db, 
            llm=motor.llm, 
            modelo_reranker=motor.reranker, 
            pregunta=pregunta,
            filtros=filtros_diccionario # <--- Conectado aqui
        )
        
        print(f"[TOOL LOG] ✅ Respuesta RAG obtenida: {respuesta}")
        print("[TOOL RETURN] 📤 Enviando al LLM para que se la comunique al usuario.")
        print("="*50 + "\n")
        
        return respuesta




    @tool
    def herramienta_exportar() -> str:
        """
        Exporta el pliego actual a un documento de Word (DOCX).
        Usa esta herramienta cuando el usuario pida descargar, guardar o exportar el documento final.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] 💾 Ejecutando: herramienta_exportar")
        
        if not motor.documento.secciones:
            error_msg = "Error: El documento esta vacio. No hay nada que exportar."
            print(f"[TOOL LOG] ❌ {error_msg}")
            print("="*50 + "\n")
            return error_msg
            
        exito_msg = (
            "El documento esta listo. Dile al usuario que puede descargarlo "
            "haciendo clic en el boton 'Descargar Pliego en Word' que ha aparecido "
            "en el apartado de 'Editor del Pliego'. "
        )
        
        print("[TOOL LOG] ✅ Indicando al LLM que redirija a la UI.")
        print("="*50 + "\n")
        return exito_msg


    @tool
    def herramienta_memorizar_borrador(confirmacion_usuario: bool = False) -> str:
        """
        Guarda el borrador actual en la base de datos de conocimiento (origen: generado).
        Usa esta herramienta SOLO cuando el usuario te haya dicho explicitamente que SI quiere guardar el pliego.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] 🧠 Ejecutando: herramienta_memorizar_borrador")
        
        if not confirmacion_usuario:
            return "ERROR: Debes pedir confirmacion explicita al usuario antes de memorizar."

        if not motor.documento.secciones:
            return "El documento esta vacio, no hay nada que guardar."

        try:
            # 1. Conseguimos el ID persistente
            id_nuevo = obtener_siguiente_id()
            nombre_seguro = f"{id_nuevo}_borrador_ia.docx"
            
            carpeta_temp = "datos/temp_uploads"
            os.makedirs(carpeta_temp, exist_ok=True)
            ruta_temporal = os.path.join(carpeta_temp, nombre_seguro)
            
            # 2. Convertimos el borrador a Word y lo guardamos temporalmente
            print("[TOOL LOG] Generando Word temporal para ingesta...")
            archivo_word_bytes = generar_bytes_word(motor.documento.secciones)
            
            with open(ruta_temporal, "wb") as f:
                # AQUI ESTA EL CAMBIO: Extraemos los bytes del objeto virtual
                f.write(archivo_word_bytes.getvalue())
                
            # 3. LLAMAMOS A TU FUNCION ESTRELLA! Pasando origen "generado"
            print("[TOOL LOG] Llamando a ingestar_documento_individual...")
            exito, mensaje = ingestar_documento_individual(
                ruta_archivo=ruta_temporal, 
                vector_db=motor.db, 
                llm=motor.llm, 
                origen_tipo="generado"
            )
            
            if exito:
                print(f"[TOOL LOG] ✅ Exito: {mensaje}")
                return f"Exito: El pliego ha sido analizado, etiquetado como 'generado' y guardado en la memoria con el ID {id_nuevo}."
            else:
                print(f"[TOOL LOG] ❌ Fallo interno: {mensaje}")
                return f"Lo siento, hubo un error al guardar el pliego: {mensaje}"
                
        except Exception as e:
            print(f"[TOOL LOG] ❌ Error critico: {e}")
            return f"Error critico al intentar memorizar: {e}"
            
        finally:
            # 4. Limpieza (El camion de la basura siempre pasa)
            if 'ruta_temporal' in locals() and os.path.exists(ruta_temporal):
                os.remove(ruta_temporal)
                print("[TOOL LOG] 🧹 Archivo Word temporal destruido.")

    @tool
    def herramienta_ver_historial() -> str:
        """
        Muestra el historial de versiones guardadas del pliego.
        Usala cuando el usuario pida ver las versiones anteriores, quiera deshacer un cambio o pregunte si se puede volver atras.
        """
        print("\n" + "="*50)
        print("[TOOL CALL] 🕒 Ejecutando: herramienta_ver_historial")
        
        versiones = motor.documento.listar_versiones()
        
        if not versiones:
            respuesta = "No hay versiones anteriores guardadas en el historial. Esta es la primera version."
            print("[TOOL LOG] 🟡 Historial vacio.")
            print("="*50 + "\n")
            return respuesta
            
        lineas = ["Historial de versiones disponibles:"]
        for v in versiones:
            # Mostramos el ID inmutable entre comillas simples para que el LLM lo capture bien
            lineas.append(f" - ID: '{v['id']}' | Fecha: {v['fecha']}")
            
        respuesta_llm = "\n".join(lineas)
        print(f"[TOOL LOG] 🟢 {len(versiones)} versiones encontradas.")
        print("="*50 + "\n")
        return respuesta_llm

    @tool
    def herramienta_restaurar_version(id_version: str, confirmacion_usuario: bool = False) -> str:
        """
        Restaura el pliego a una version anterior usando su ID de texto (ej. '20260423_124753').
        REGLA CRITICA: El parametro 'confirmacion_usuario' debe ser False por defecto. 
        SOLO puedes ponerlo a True si le has advertido al usuario de las consecuencias y este ha respondido afirmativamente.
        """
        print("\n" + "="*50)
        print(f"[TOOL CALL] ⏪ Ejecutando: herramienta_restaurar_version (ID: {id_version}, Confirmado: {confirmacion_usuario})")
        
        # 1. BARRERA DE SEGURIDAD
        if not confirmacion_usuario:
            alerta = (
                f"SISTEMA BLOQUEADO: No puedes restaurar la version '{id_version}' todavia. "
                "Debes decirle al usuario: '¿Estas completamente seguro de que quieres cargar esta version? Se perderan los cambios no guardados.' "
                "Si el usuario responde que si, vuelve a llamar a esta herramienta pasando confirmacion_usuario=True."
            )
            print("[TOOL LOG] 🛑 Bloqueado por falta de confirmacion. Obligando al LLM a preguntar.")
            print("="*50 + "\n")
            return alerta

        # 2. EJECUCION REAL
        exito, fecha = motor.documento.restaurar_version(id_version)
        
        if exito:
            respuesta = f"Exito: El documento ha sido restaurado correctamente a la version del {fecha}."
            print(f"[TOOL LOG] ✅ Restaurado con exito a {fecha}.")
        else:
            respuesta = f"Error: No se pudo encontrar o restaurar la version con ID '{id_version}'. Revisa si el ID es correcto."
            print(f"[TOOL LOG] ❌ Fallo al restaurar ID {id_version}.")
            
        print("="*50 + "\n")
        return respuesta

    tools = [
        herramienta_ver_rutas, 
        herramienta_crear_secciones, 
        herramienta_eliminar_secciones, 
        herramienta_modificar_seccion,
        herramienta_resumir_seccion,
        herramienta_consultar_ley, 
        herramienta_exportar,
        herramienta_ver_historial,
        herramienta_restaurar_version,
        herramienta_memorizar_borrador
    ]
    return tools