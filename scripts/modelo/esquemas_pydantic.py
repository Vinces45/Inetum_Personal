from typing import Optional, List, Literal
from pydantic import BaseModel, Field

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
        default=None, description="True si el contrato esta dividido en lotes, " \
                                    "False si dice claramente que el contrato es de lote unico."
    )
    financiacion_europea: Optional[bool] = Field(
        default=None, description="True si se dice que cuenta con financiacion de fondos europeos (FEDER, Next Generation), " \
                                    "False si dice claramente que no cuenta con financiacion de fondos europeaos."
    )

    # CAMPOS QUE HE METIDO PERO QUE EN PRINICIPIO NO VEO INTERESANTES PARA EL FILTRADO
    # SEGURAMENTE LOS ACABE RETIRANDO

    archivo: Optional[str] = Field(
        default=None, description="Nombre del archivo PDF original si el usuario lo menciona."
    )
    expediente: Optional[str] = Field(
        default=None, description="Codigo oficial del expediente (ej: 50-7-9.02-0004/2025)."
    )
    cpv: Optional[List[str]] = Field(
        default=None, description="Lista de codigos CPV mencionados (ej: ['98341130-5'])."
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
    accion: Literal["crear", "modificar", "eliminar", "resumir", "consultar"] = Field(
        description="Si pide algo nuevo es 'crear', cambiar es 'modificar', borrar es 'eliminar', resumir contenido es 'resumir', pedir información directamente es 'consultar'."
    )
    # AQUI ESTA EL CAMBIO: Ahora recibe una lista de objetos complejos
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
    