import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Importamos tu motor RAG (Asegurate de que generador_rag.py este en la misma carpeta)
from generador_rag import inicializar_bd, inicializar_llm, generar_borrador_seccion

def analizar_peticion_usuario(peticion):
    """
    Fase 1: LLM Puro (Sin RAG)
    Usa el LLM para extraer la intencion del usuario y generar un plan de accion en JSON.
    """
    print("\n[ORQUESTADOR] Analizando peticion del usuario...")
    llm_orquestador = ChatOllama(model="llama3", temperature=0.0, format="json")
    
    prompt_orquestador = """
    Eres el Orquestador de un sistema de generacion de pliegos del Gobierno de La Rioja.
    Analiza la peticion del usuario y devuelve UNICAMENTE un objeto JSON con esta estructura exacta:
    {{
        "filtros": {{
            "tipo_documento": "PCAP o PPT (si se deduce, si no null)",
            "tramitacion": "Urgente u Ordinaria (si se deduce, si no null)"
        }},
        "secciones_a_generar": [
            "Nombre de la seccion 1",
            "Nombre de la seccion 2"
        ]
    }}

    PETICION DEL USUARIO: "{peticion}"
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_orquestador)
    cadena = prompt | llm_orquestador | JsonOutputParser()
    
    try:
        plan = cadena.invoke({"peticion": peticion})
        # Limpiamos los filtros null para no pasarselos a ChromaDB
        filtros_limpios = {k: v for k, v in plan.get("filtros", {}).items() if v is not None}
        plan["filtros"] = filtros_limpios if filtros_limpios else None
        return plan
    except Exception as e:
        print(f"[ERROR] El Orquestador fallo al crear el JSON: {e}")
        return None

def ejecutar_plan_rag(plan, db, llm):
    """
    Fase 2: LLM con RAG
    Itera sobre el plan y llama a tu motor RAG por cada seccion.
    """
    documento_final = ""
    filtros = plan.get("filtros")
    secciones = plan.get("secciones_a_generar", [])
    
    print(f"\n[ORQUESTADOR] Iniciando redaccion de {len(secciones)} secciones con filtros: {filtros}")
    
    for i, seccion in enumerate(secciones, 1):
        print(f"\n--- Redactando seccion {i}/{len(secciones)}: {seccion} ---")
        
        # Llamamos a TU funcion pura del archivo generador_rag.py
        borrador_seccion = generar_borrador_seccion(db, llm, f"Redacta el apartado de: {seccion}", filtros)
        
        # Ensamblamos el documento
        documento_final += f"\n\n### {i}. {seccion.upper()}\n\n"
        documento_final += borrador_seccion
        
    return documento_final

if __name__ == "__main__":
    # 1. Inicializamos los recursos pesados una sola vez
    print("Arrancando el cerebro del sistema...")
    db_vectorial = inicializar_bd()
    modelo_llm_rag = inicializar_llm()
    
    print("\n" + "="*50)
    print("CHATBOT ADMINISTRATIVO - GOBIERNO DE LA RIOJA")
    print("="*50)
    
    while True:
        peticion_usuario = input("\nUsuario: ")
        if peticion_usuario.lower() == "salir": break
            
        # PASO 1: Analizar y Planificar
        plan_accion = analizar_peticion_usuario(peticion_usuario)
        
        if plan_accion:
            print(f"\n[PLAN] {json.dumps(plan_accion, indent=2)}")
            
            # PASO 2: Ejecutar Redaccion Asistida
            resultado_completo = ejecutar_plan_rag(plan_accion, db_vectorial, modelo_llm_rag)
            
            print("\n\n" + "="*50)
            print("DOCUMENTO FINAL GENERADO:")
            print("="*50)
            print(resultado_completo)