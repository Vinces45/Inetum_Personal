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
        default=None, description="True si el contrato esta dividido en lotes, False si es lote unico."
    )
    financiacion_europea: Optional[bool] = Field(
        default=None, description="True si cuenta con financiacion de fondos europeos (FEDER, Next Generation), False si no."
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

class PlanOrquestador(BaseModel):
    accion: Literal["crear", "modificar"] = Field(
        description="Si el usuario pide algo nuevo es 'crear'. Si pide cambiar algo existente es 'modificar'."
    )
    secciones_a_crear: Optional[List[str]] = Field(
        default=None, description="Lista de nombres de secciones a redactar."
    )
    seccion_a_modificar: Optional[str] = Field(
        default=None, description="Nombre EXACTO de la seccion a cambiar (debe existir en el borrador)."
    )
    feedback: Optional[str] = Field(
        default=None, description="La instruccion de lo que hay que cambiar en la seccion."
    )
    filtros: Optional[FiltrosMetadatos] = Field(
        default=None, description="Filtros extraidos de la peticion del usuario."
    )