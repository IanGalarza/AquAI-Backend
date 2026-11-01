from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.dtos.responses.getSource import ColumnDetail
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus


class RetrievedColumn(ColumnDetail):
    # Conserva también header_values, unit_evidence, source_column/source_row y
    # futuros metadatos de ingesta, sin duplicar el contrato de columnas existente.
    model_config = ConfigDict(extra="allow")


class DatasetProvenance(BaseModel):
    source_name: str
    original_filename: str
    file_type: str
    content_hash: str
    source_locator: dict[str, JsonValue]
    processing_status: ProcessingStatus
    semantic_index_status: SemanticIndexStatus


class VerifiedDataset(BaseModel):
    """Proyección de PostgreSQL comprobada contra la tabla física."""

    dataset_id: UUID
    source_id: UUID
    semantic_point_id: UUID
    name: str
    description: str | None
    physical_table_name: str
    verified_version: int = Field(ge=1)
    row_count: int = Field(gt=0)
    columns_metadata: list[RetrievedColumn]
    provenance: DatasetProvenance
    warnings: list[str] = Field(default_factory=list)


class DatasetCandidate(VerifiedDataset):
    score: float = Field(allow_inf_nan=False)
    indexed_version: int = Field(ge=1)
    embedding_model: str


class ValueMatch(BaseModel):
    """Valores TEXT realmente almacenados que contienen la mención como palabras completas."""

    column: str
    values: list[str]
    rows: int = Field(ge=0)


class DatasetEvidence(BaseModel):
    """Evidencia léxica y de registros; ordena candidatos sin decidir el plan."""

    dataset_id: UUID
    value_matches: dict[str, list[ValueMatch]] = Field(default_factory=dict)
    variable_matches: dict[str, list[str]] = Field(default_factory=dict)
    dimension_matches: dict[str, list[str]] = Field(default_factory=dict)
    # Otros valores de la columna donde está un valor mencionado, de más a menos registros con la variable.
    category_values: dict[str, list[str]] = Field(default_factory=dict)


class OmittedCandidate(BaseModel):
    """Dataset con la variable y los valores pedidos que no llegó al Planner: no identifica la dimensión preguntada
    o quedó fuera del límite de candidatos."""

    dataset_id: UUID
    name: str
    source_name: str
    reason: Literal["missing_dimension", "beyond_limit"]


class DatasetRetrievalResult(BaseModel):
    question: str
    candidates: list[DatasetCandidate | VerifiedDataset]
    evidence: list[DatasetEvidence] = Field(default_factory=list)
    omitted_candidates: list[OmittedCandidate] = Field(default_factory=list)
    unmatched_values: list[str] = Field(default_factory=list)
    dimension_mentions: list[str] = Field(default_factory=list)
    confirmed_values: list[str] = Field(default_factory=list)
    retrieval_method: Literal["semantic", "explicit"] = "semantic"
    verified_at: datetime
    status: Literal["candidates_available", "no_available_candidates"]
    # La similitud coseno no confirma que las columnas permitan responder.
    relevance: Literal["unassessed"] = "unassessed"
    requires_planner_review: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)
