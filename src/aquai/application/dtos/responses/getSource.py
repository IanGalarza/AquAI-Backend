"""Detalle de una fuente con revisiones y vistas previas tipadas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer

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


_EXAMPLE_RECIPE = {
    "status": "ready", "requires_human_review": False, "warnings": [],
    "sheets": [{"sheet_name": "data", "block_id": "observations", "role": "observations",
                "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
                "data_start_column": 1, "data_end_column": 2,
                "transformations": [{"operation": "convert_type", "columns": ["score"], "target_type": "integer"}]}],
}


class SourceDetailResponse(SourceListItem):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "id": "a5079a18-d2f0-4ea7-82df-d43a4fa629ad", "name": "mediciones.csv",
        "original_filename": "mediciones.csv", "file_type": "csv", "mime_type": "text/csv",
        "file_size_bytes": 1024, "created_at": "2026-10-12T00:00:00Z", "updated_at": "2026-10-12T00:01:00Z",
        "isActive": False, "processing_status": "review_required", "validation_status": "review_required",
        "stage": "awaiting_review", "revision": 0,
        "blocks": [{"region_id": "data", "sheet_name": "data", "start_row": 2, "end_row": 3,
                    "start_column": 1, "end_column": 2, "kind": "observations"}],
        "preview": [{"sheet_name": "data", "block_id": "observations", "columns": ["name", "score"],
                     "rows": [{"row_number": 2, "values": ["Ana", "invalid"]}]}],
        "findings": [{"id": "e233ac4b05a4484e83bc4630e977f40a", "kind": "review", "code": "conversion_failures",
                      "message": "Una conversión de score no pudo completarse.", "requires_review": True,
                      "sheet_name": "data", "block_id": "observations", "range": "A2:B3"}],
        "reviews": [{"id": "e233ac4b05a4484e83bc4630e977f40a", "review_id": "e233ac4b05a4484e83bc4630e977f40a",
                     "source_id": "a5079a18-d2f0-4ea7-82df-d43a4fa629ad", "revision": 0,
                     "code": "conversion_failures", "message": "Una conversión de score no pudo completarse.",
                     "sheet_name": "data", "block_id": "observations", "range": "A2:B3",
                     "options": [{"action": "correct_transformations", "description": "Corregir las operaciones del bloque.",
                                  "values": {"type": "object", "required": ["transformations"],
                                             "properties": {"transformations": {"type": "array"}}}}]}],
        "decisions": [], "technical_errors": [], "queryable": False,
        "dataset_storage": {"status": "not_implemented", "message": "Todavía no se almacenan datasets ni se crean tablas físicas."},
        "original_proposal": _EXAMPLE_RECIPE, "effective_recipe": _EXAMPLE_RECIPE, "rules": {},
    }]})

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
