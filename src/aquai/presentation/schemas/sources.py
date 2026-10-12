"""Contratos públicos de gestión de fuentes, sin originales binarios."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


ProcessingStatus = Literal[
    "uploaded", "processing", "ready", "ready_with_warnings", "review_required", "failed"
]
ValidationStatus = Literal["pending", "ready", "ready_with_warnings", "review_required", "failed"]

_EXAMPLE_RECIPE = {
    "status": "ready", "requires_human_review": False, "warnings": [],
    "sheets": [{"sheet_name": "data", "block_id": "observations", "role": "observations",
                "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
                "data_start_column": 1, "data_end_column": 2,
                "transformations": [{"operation": "convert_type", "columns": ["score"], "target_type": "integer"}]}],
}


class SourceAcceptedResponse(BaseModel):
    id: UUID
    processing_status: ProcessingStatus
    validation_status: ValidationStatus


class SourceListItem(SourceAcceptedResponse):
    name: str
    file_type: str
    file_size_bytes: int = Field(ge=0)
    created_at: datetime
    is_active: bool = Field(serialization_alias="isActive")


class SourceListResponse(BaseModel):
    items: list[SourceListItem]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)


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
    blocks: list[dict[str, JsonValue]] = Field(default_factory=list, description="Bloques detectados y sus rangos de origen.")
    preview: list[dict[str, JsonValue]] = Field(default_factory=list, description="Vista previa acotada; no contiene el archivo completo.")
    findings: list[dict[str, JsonValue]] = Field(default_factory=list, description="Hallazgos de la última validación.")
    reviews: list[dict[str, JsonValue]] = Field(default_factory=list, description="Revisiones vigentes: identificador estable, explicación, hoja/rango y acciones admitidas.")
    decisions: list[dict[str, JsonValue]] = Field(default_factory=list, description="Decisiones registradas con alcance exclusivo de esta fuente.")
    technical_errors: list[dict[str, JsonValue]] = Field(default_factory=list, description="Fallos técnicos que no constituyen preguntas para el usuario.")
    dataset_storage: dict[str, JsonValue] = Field(default_factory=dict, description="Estado real del almacenamiento de datos preparados.")
    queryable: bool = Field(description="Sólo puede ser verdadero si se publicaron datos aceptados y persistidos.")
    original_proposal: dict[str, JsonValue] | None = None
    effective_recipe: dict[str, JsonValue] | None = None
    rules: dict[str, JsonValue] = Field(default_factory=dict)


class SourceDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, json_schema_extra={
        "examples": [
            {"action": "correct_transformations", "values": {"transformations": []}},
            {"action": "clarify_record_identity", "values": {
                "identity": "distinct", "statement": "Los registros corresponden a campañas independientes.",
            }},
        ],
    })

    action: str = Field(min_length=1, max_length=80, description="Una acción admitida en la revisión vigente devuelta por GET /sources/{id}.")
    values: dict[str, JsonValue] = Field(default_factory=dict, description="Valores estructurados exigidos por esa acción; se validan contra el estado actual.")


class SourceDecisionResponse(SourceAcceptedResponse):
    review_id: str | None = None
    revision: int = Field(default=0, ge=0)
    duplicate: bool = False
