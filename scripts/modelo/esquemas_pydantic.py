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
        description="Expande la peticion. Si el usuario pide tramites rapidos o acelerados, incluye ['Urgente', 'Emergencia']. Si pide normales, ['Ordinaria']."
    )
    procedimiento: Optional[List[Literal["Abierto", "Menor", "Negociado", "Simplificado", "Super Simplificado"]]] = Field(
        default=None,
        description="Expande la peticion. Si pide tramites sencillos, incluye ['Simplificado', 'Super Simplificado', 'Menor']. Si pide sin publicidad, ['Negociado']."
    )
    tipo_contrato: Optional[List[Literal["Obras", "Servicios", "Suministros"]]] = Field(
        default=None,
        description="Clasifica si busca construir (Obras), comprar bienes (Suministros) o contratar tareas (Servicios)."
    )


    



class PeticionSubseccion(BaseModel):
    titulo: str = Field(description="Titulo de la subseccion (ej: '1.1. Garantias').")
    instruccion_especifica: str = Field(description="Que pide exactamente el usuario para esta subseccion concreta.")

class PeticionSeccion(BaseModel):
    titulo: str = Field(description="Titulo principal de la seccion (ej: '1. Objeto del Contrato').")
    instruccion_especifica: Optional[str] = Field(
        default=None, 
        description="Instruccion general para esta seccion principal. Si solo es un titulo contenedor, dejalo vacio."
    )
    subsecciones: Optional[List[PeticionSubseccion]] = Field(
        default=None, 
        description="Lista de subsecciones si el usuario pide desglosarlo en apartados mas pequeños."
    )

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
    

    