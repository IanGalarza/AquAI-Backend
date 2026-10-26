"""Evidencia de recuperación serializable, sin planes ni instrucciones ejecutables."""

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


class DatasetRetrievalResult(BaseModel):
    question: str
    candidates: list[DatasetCandidate]
    verified_at: datetime
    status: Literal["candidates_available", "no_available_candidates"]
    # La similitud coseno no confirma que las columnas permitan responder.
    relevance: Literal["unassessed"] = "unassessed"
    requires_planner_review: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)
