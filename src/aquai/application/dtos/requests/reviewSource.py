"""Decisión estructurada sobre una revisión vigente de una fuente."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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
