from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class FiltrosMetadatos(BaseModel):
    presupuesto_base_licitacion: Optional[float] = Field(
        default=None, 
        description="Importe en euros SIN IVA. ATENCION: Es muy comun que el pliego muestre primero una tabla con el importe con 'IVA incluido', debes IGNORAR esa cifra. Busca siempre el 'Importe neto', 'Importe de licitacion IVA excluido' o 'Valor estimado'."
    )
    valor_estimado_contrato: Optional[float] = Field(
        default=None, 
        description="Valor estimado total del contrato en euros."
    )
    iva_porcentaje: Optional[int] = Field(
        default=None, 
        description="Porcentaje de IVA (ej: 21 o 10)."
    )
    plazo_ejecucion_meses: Optional[int] = Field(
        default=None, 
        description="Plazo de ejecucion expresado en meses. Si hay fechas, calcula los meses de diferencia."
    )
    tiempo_prorroga_meses: Optional[int] = Field(
        default=None, 
        description="Tiempo de prorroga permitido en meses. Si dice que no hay o no procede, dejalo nulo."
    )
    tipo_contrato: Optional[Literal["Obras", "Servicios", "Suministros"]] = Field(
        default=None, 
        description="Tipo principal del contrato."
    )
    tramitacion: Optional[Literal["Ordinaria", "Urgente", "Emergencia"]] = Field(
        default=None, 
        description="Velocidad o tipo de tramitacion del expediente."
    )
    procedimiento: Optional[Literal["Abierto", "Menor", "Negociado", "Simplificado", "Super Simplificado"]] = Field(
        default=None, 
        description="Procedimiento de adjudicacion. REGLA ESTRICTA: Si el texto dice 'Abierto Simplificado', debes elegir OBLIGATORIAMENTE 'Simplificado'."
    )
    lotes: Optional[bool] = Field(
        default=None, 
        description="True si el contrato esta dividido en lotes, False si es lote unico."
    )
    financiacion_europea: Optional[bool] = Field(
        default=None, 
        description="True si cuenta con financiacion de fondos europeos (FEDER, Next Generation, MRR), False si marca que no."
    )
    expediente: Optional[str] = Field(
        default=None, 
        description="Codigo oficial del expediente."
    )
    cpv: Optional[List[str]] = Field(
        default=None, 
        description="Lista de codigos CPV mencionados."
    )

    #NUEVOS METADATOS

    anio_expediente: Optional[int] = Field(
        default=None, 
        description="Ano de licitacion o del expediente (ej: 2024 o 2025)."
    )
    contrato_sara: Optional[bool] = Field(
        default=False, 
        description="True SOLO si dice expresamente 'Sujeto a Regulacion Armonizada'. ATENCION: Que mencione el 'Real Decreto-ley 36/2020' o 'fondos europeos' NO significa que sea SARA. Ante la duda o si no menciona la frase exacta, pon SIEMPRE False."
    )
    objeto_contrato: Optional[str] = Field(
        default=None, 
        description="Titulo corto o descripcion principal del objeto del contrato."
    )


class FiltrosBusqueda(BaseModel):
    tramitacion: Optional[List[Literal["Ordinaria", "Urgente", "Emergencia"]]] = Field(
        default=None, 
        description="NO ADIVINAR. Extraer SOLO si el usuario menciona textualmente 'ordinaria', 'urgente' o 'emergencia'. Si no se menciona, dejar como nulo."
    )
    procedimiento: Optional[List[Literal["Abierto", "Menor", "Negociado", "Simplificado", "Super Simplificado"]]] = Field(
        default=None,
        description="NO ADIVINAR. Extraer SOLO si el usuario menciona expresamente el tipo de procedimiento. Si no lo dice de forma explicita, dejar como nulo."
    )
    tipo_contrato: Optional[List[Literal["Obras", "Servicios", "Suministros"]]] = Field(
        default=None,
        description="Extraer SOLO si el usuario especifica claramente si es obras, servicios o suministros. NUNCA inventar por defecto."
    )
    contrato_sara: Optional[bool] = Field(
        default=None,
        description="Dejar SIEMPRE vacio (nulo) a menos que el usuario mencione explicitamente 'SARA' o 'regulacion armonizada' (True), o pida que no este sujeto (False)."
    )
    lotes: Optional[bool] = Field(
        default=None,
        description="Dejar SIEMPRE vacio (nulo). Rellenar con True o False SOLO si el prompt habla explicitamente de la division en lotes."
    )
    financiacion_europea: Optional[bool] = Field(
        default=None,
        description="Dejar SIEMPRE vacio (nulo). Pon True SOLO si el usuario menciona explicitamente fondos europeos, UE, FEDER, MRR o Next Generation."
    )
    presupuesto_base_licitacion: Optional[float] = Field(
        default=None,
        description="NO INVENTAR. Rellenar SOLO si el usuario menciona expresamente un importe o presupuesto exacto en la peticion."
    )
    plazo_ejecucion_meses: Optional[int] = Field(
        default=None,
        description="NO INVENTAR. Rellenar SOLO si el usuario menciona un plazo. Convertir a meses (ej. '1 ano' = 12). Si no hay plazo explicito, dejar nulo."
    )
    iva_porcentaje: Optional[int] = Field(
        default=None,
        description="NO INVENTAR. Rellenar SOLO si se menciona explicitamente el porcentaje de IVA. Devuelve solo el numero entero (ej. 21)."
    )

    



class PeticionSubseccion(BaseModel):
    titulo: str = Field(description="Titulo de la subseccion (ej: '1.1. Garantias').")
    instruccion_especifica: str = Field(description="Que pide exactamente el usuario para esta subseccion concreta.")

class PeticionSeccion(BaseModel):
    titulo: str = Field(
        description="Titulo de la seccion o subseccion (ej: '1. Objeto', '1.1. Garantias')."
    )
    instruccion_especifica: Optional[str] = Field(
        default=None, 
        description="Que pide exactamente el usuario para esta seccion concreta."
    )
    subsecciones: Optional[List['PeticionSeccion']] = Field(
        default=None, 
        description="Lista de subsecciones anidadas de forma recursiva, si el usuario pide desglosarlo en apartados mas pequenos."
    )

# Es obligatorio reconstruir el modelo en Pydantic cuando una clase se llama a si misma
PeticionSeccion.model_rebuild()

class PlanOrquestador(BaseModel):
    accion: Literal["crear", "modificar", "eliminar", "resumir", "consultar", "exportar"] = Field(
        description="Si pide algo nuevo es 'crear', cambiar es 'modificar', borrar es 'eliminar', resumir contenido es 'resumir', pedir información directamente es 'consultar' y descargar o exportar es 'exportar'."
    )
    secciones_a_crear: Optional[List[PeticionSeccion]] = Field(
        default=None, 
        description="Lista detallada de las secciones y subsecciones a redactar desde cero."
    )
    seccion_a_modificar: Optional[str] = Field(
        default=None, description="Nombre EXACTO de la seccion a cambiar."
    )
    seccion_a_eliminar: Optional[str] = Field(
        default=None, description="Nombre EXACTO de la seccion a borrar."
    )
    seccion_a_resumir: Optional[str] = Field(
        default=None, 
        description="Ruta de la seccion a resumir. Si el usuario pide resumir todo el documento, dejalo en null."
    )
    feedback: Optional[str] = Field(
        default=None, description="La instruccion de lo que hay que cambiar en la seccion."
    )
    filtros: Optional[FiltrosMetadatos] = Field(
        default=None, description="Filtros extraidos de la peticion del usuario."
    )
    pregunta_legal: Optional[str] = Field(
        default=None, 
        description="Si la accion es 'consultar', extrae aqui la duda legal exacta del usuario."
    )
    

    