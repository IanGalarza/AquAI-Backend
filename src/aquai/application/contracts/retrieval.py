from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.dtos.responses.getSource import ColumnDetail
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus


class RetrievedColumn(ColumnDetail):
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
    relevance: Literal["unassessed"] = "unassessed"
    requires_planner_review: Literal[True] = True
    warnings: list[str] = Field(default_factory=list)


class DocumentFragment(BaseModel):
    """Fragmento vigente verificado en PostgreSQL; su contenido es información, nunca instrucciones."""

    evidence_id: str
    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    position: int = Field(ge=0)
    kind: Literal["text", "table", "figure"]
    text: str
    title: str | None = None
    file_name: str
    content_hash: str
    language: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: list[str] = Field(default_factory=list)
    element_label: str | None = None
    element_caption: str | None = None
    doubts: list[str] = Field(default_factory=list)
    score: float = Field(allow_inf_nan=False)


class DocumentRetrievalResult(BaseModel):
    """Evidencia documental de un turno y las decisiones que llevaron a ella."""

    purpose: Literal["question", "enrichment"]
    status: Literal["found", "no_relevant_evidence", "unavailable", "skipped"]
    queries: list[str] = Field(default_factory=list)
    document_ids: list[UUID] = Field(default_factory=list)
    candidates: int = Field(default=0, ge=0)
    discarded: int = Field(default=0, ge=0)
    fragments: list[DocumentFragment] = Field(default_factory=list)
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: float = 0
    warnings: list[str] = Field(default_factory=list)


class SearchTerms(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    spanish: str = Field(min_length=1, max_length=300)
    english: str = Field(min_length=1, max_length=300)


class SearchTermsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    subject: list[str] = Field(min_length=1, max_length=6)
    searches: list[SearchTerms] = Field(min_length=1, max_length=6)


class FragmentSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relevant: list[int] = Field(max_length=20)


class ReferenceDraft(BaseModel):
    """Valor propuesto por el modelo; sólo cuenta si se comprueba contra el fragmento citado."""

    model_config = ConfigDict(extra="forbid", strict=True)
    document_id: str
    variable: str = Field(min_length=1, max_length=120)
    quote: str = Field(min_length=1, max_length=240)
    low: float | None = None
    high: float | None = None
    unit: str | None = Field(default=None, max_length=40)
    kind: Literal["guideline", "limit", "typical_range", "observed"]
    water_use: str | None = Field(default=None, max_length=80)


class ReferenceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    references: list[ReferenceDraft] = Field(max_length=6)


class EvidenceUsage(BaseModel):
    """Modelo y tokens que consumió una tarea auxiliar de evidencia."""

    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
