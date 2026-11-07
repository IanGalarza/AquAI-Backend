from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

QueryRoute = Literal[
    "general", "catalog", "data", "document", "historical", "needs_clarification", "failed"
]


class HistoricalAnalysis(BaseModel):
    status: Literal["validated", "partial"]
    sources: list[dict[str, JsonValue]]
    operations: list[dict[str, JsonValue]]
    operation_references: list[dict[str, JsonValue]]
    request_history: list[str]
    verified_at: datetime
    complete: bool = True


class DataMention(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(
        min_length=1,
        max_length=200,
        description="Texto copiado literalmente de la pregunta actual o de una solicitud pendiente.",
    )
    kind: Literal["variable", "value", "dimension", "threshold", "date", "limit"] = Field(
        description="variable: parámetro medido; value: valor de columna que acota registros, como una localidad o estación; dimension: categoría que se pregunta o por la que se agrupa, como 'qué sitios'; threshold: comparación con un número, como 'mayor que 500'; date: fecha o período solicitado; limit: cantidad de registros o resultados que se pide ver."
    )


class HistoryMessage(BaseModel):
    message_id: UUID
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)
    text_complete: bool = True
    scientific: bool = False
    analysis: HistoricalAnalysis | None = None
    clarification: bool = False
    pending_requests: list[str] = Field(default_factory=list, max_length=8)
    context_message_ids: list[UUID] = Field(default_factory=list, max_length=4)
    scope: list[DataMention] = Field(default_factory=list, max_length=12)
    replay_of: UUID | None = None
    named_sources: bool = False
    recalled: bool = False
    segments: list[dict[str, JsonValue]] = Field(default_factory=list)
    source_columns: dict[str, list[dict[str, JsonValue]]] = Field(default_factory=dict)
    documents: list[dict[str, JsonValue]] = Field(default_factory=list)


class RoutingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    route: Literal["general", "catalog", "data", "document", "historical", "needs_clarification"]
    language: Literal["es", "en"]
    context_required: bool
    history_message_ids: list[UUID] = Field(max_length=4)
    source_mentions: list[str] = Field(
        max_length=4,
        description="Nombres de archivos, fuentes o datasets copiados literalmente de la pregunta actual; nunca localidades, sitios, estaciones, variables ni fechas.",
    )
    data_mentions: list[DataMention] = Field(
        max_length=12,
        description="Variables, valores de columnas, dimensiones, umbrales y fechas como menciones separadas: en 'pH de Punta Lara' son 'pH' (variable) y 'Punta Lara' (value). No son fuentes.",
    )
    clarification_question: str | None = Field(max_length=500)
    reason: str = Field(min_length=1, max_length=400)


class RoutingReply(BaseModel):
    decision: RoutingDraft
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class DatasetBinding(BaseModel):
    dataset_id: UUID
    source_id: UUID
    historical_version: int | None = None


class CatalogDataset(BaseModel):
    dataset_id: UUID
    source_id: UUID
    dataset_name: str
    source_name: str
    version: int
    available: bool
    original_filename: str = ""
    exact: bool = False
    name_match: bool = False


class RoutingResult(BaseModel):
    route: QueryRoute
    language: Literal["es", "en"]
    original_question: str
    contextualized_question: str
    history_message_ids: list[UUID] = Field(default_factory=list)
    request_history: list[str] = Field(default_factory=list)
    dataset_bindings: list[DatasetBinding] = Field(default_factory=list)
    document_ids: list[UUID] = Field(default_factory=list)
    asks_citations: bool = False
    source_mentions: list[str] = Field(default_factory=list)
    data_mentions: list[DataMention | str] = Field(default_factory=list)
    conserved_mentions: list[DataMention] = Field(default_factory=list)
    tentative_values: list[str] = Field(default_factory=list)
    requested_period: Literal["year", "month", "day"] | None = None
    reference_comparison: str | None = None
    reference_options: list[str] = Field(default_factory=list)
    reference_documents: list[str] = Field(default_factory=list)
    reselects_datasets: bool = False
    inherits_named_sources: bool = False
    context_used: bool = False
    question_contextualized: bool = False
    clarification_question: str | None = None
    reason: str
    warnings: list[str] = Field(default_factory=list)
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: float = 0
    context_sha256: str | None = None
    context_size_bytes: int = 0
    history_messages_provided: int = 0
    omitted_history_ids: list[UUID] = Field(default_factory=list)
    error: str | None = None


class GeneralDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1, max_length=6000)
    language: Literal["es", "en"]


class GeneralReply(BaseModel):
    draft: GeneralDraft
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
