from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aquai.application.dtos.responses.sourceState import SourceStateResponse
from aquai.application.units import documented_unit


#
# Detalle de una columna de un dataset: nombre, tipo, unidad y significado.
#
class ColumnDetail(BaseModel):
    original_name: str
    sql_name: str
    sql_type: str
    unit: str | None = None
    meaning: str | None = None
    numeric_eligible: bool | None = None
    representation: str | None = None
    derived_from: str | None = None
    excluded_count: int | None = Field(default=None, ge=0)
    header_missing: bool = False

    #
    # Completa la unidad de una medición con la que documenta su encabezado, si falta.
    #
    @model_validator(mode="after")
    def header_documented_unit(self):
        if self.unit is None and self.numeric_eligible:
            self.unit = documented_unit(self.original_name)

        return self


#
# Ubicación de un dataset dentro del archivo fuente: hoja, rango y posible solapamiento.
#
class SourceLocator(BaseModel):
    sheet: str | None = None
    range: str | None = None
    has_possible_overlap: bool = False

    #
    # Normaliza el indicador de posible solapamiento de la ubicación de la fuente.
    #
    @model_validator(mode="before")
    @classmethod
    def project_overlap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                **value,
                "has_possible_overlap": bool(
                    value.get("has_possible_overlap") or value.get("possible_overlap")
                ),
            }
        return value


#
# Error ocurrido durante el procesamiento de una fuente: código y mensaje.
#
class ProcessingError(BaseModel):
    code: str | None = None
    message: str | None = None


#
# Informe del procesamiento de una fuente: advertencias y errores.
#
class ProcessingReport(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    warning_count: int = Field(default=0, ge=0)
    errors: list[str | ProcessingError] = Field(default_factory=list)


#
# Detalle de un dataset de una fuente: ubicación, filas, columnas y avisos de calidad.
#
class DatasetDetail(BaseModel):
    id: UUID
    name: str
    description: str | None
    source_locator: SourceLocator
    row_count: int = Field(ge=0)
    columns_metadata: list[ColumnDetail]
    quality_warnings: list[str]
    is_active: bool
    queryable: bool

    #
    # Usa la descripción semántica del dataset como su descripción cuando existe.
    #
    @model_validator(mode="before")
    @classmethod
    def project_description(cls, value: Any) -> Any:
        if isinstance(value, dict):
            description = value.get("semantic_description")
            if isinstance(description, str) and description.strip():
                return {**value, "description": description.strip()}
        return value


#
# Resumen del contenido extraído de una fuente documental.
#
class DocumentSummary(BaseModel):
    id: UUID
    title: str | None
    language: str | None
    page_count: int | None = Field(ge=1)
    duplicate_of_source_id: UUID | None
    section_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    figure_count: int = Field(ge=0)
    index_error: str | None = None


#
# Respuesta con el detalle completo de una fuente y sus datasets.
#
class SourceDetailResponse(SourceStateResponse):
    model_config = ConfigDict(populate_by_name=True)
    original_filename: str
    file_type: str
    file_size_bytes: int
    description: str | None
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime
    processing_report: ProcessingReport
    datasets: list[DatasetDetail]
    queryable: bool
    document: DocumentSummary | None = None
