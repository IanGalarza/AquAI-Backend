"""Detalle de una fuente con revisiones y vistas previas tipadas."""

from datetime import datetime

from pydantic import BaseModel, Field, JsonValue, field_serializer

from aquai.application.dtos.responses.listSources import SourceListItem
from aquai.application.dtos.responses.sourceDetails import (
    SourceBlock,
    SourceDatasetStorage,
    SourceDecision,
    SourceFinding,
    SourcePreview,
    SourceReview,
    SourceTechnicalError,
)


class SourceDetailResponse(SourceListItem):
    original_filename: str | None = None
    mime_type: str | None = None
    updated_at: datetime | None = None
    stage: str
    revision: int = Field(ge=0)
    blocks: list[SourceBlock] = Field(default_factory=list, description="Bloques detectados y sus rangos de origen.")
    preview: list[SourcePreview] = Field(default_factory=list, description="Vista previa acotada; no contiene el archivo completo.")
    findings: list[SourceFinding] = Field(default_factory=list, description="Hallazgos de la última validación.")
    reviews: list[SourceReview] = Field(default_factory=list, description="Revisiones vigentes: identificador estable, explicación, hoja/rango y acciones admitidas.")
    decisions: list[SourceDecision] = Field(default_factory=list, description="Decisiones registradas con alcance exclusivo de esta fuente.")
    technical_errors: list[SourceTechnicalError] = Field(default_factory=list, description="Fallos técnicos que no constituyen preguntas para el usuario.")
    dataset_storage: SourceDatasetStorage = Field(default_factory=SourceDatasetStorage, description="Estado real del almacenamiento de datos preparados.")
    queryable: bool = Field(description="Sólo puede ser verdadero si se publicaron datos aceptados y persistidos.")
    original_proposal: dict[str, JsonValue] | None = None
    effective_recipe: dict[str, JsonValue] | None = None
    rules: dict[str, JsonValue] = Field(default_factory=dict)

    @field_serializer(
        "blocks", "preview", "findings", "reviews", "decisions", "technical_errors", "dataset_storage",
        when_used="json",
    )
    def serialize_details(self, value: list[BaseModel] | BaseModel):
        # Los documentos existentes distinguen campos ausentes de valores null.
        if isinstance(value, list):
            return [item.model_dump(mode="json", exclude_unset=True) for item in value]
        return value.model_dump(mode="json", exclude_unset=True)
