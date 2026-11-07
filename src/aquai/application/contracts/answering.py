from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.contracts.execution import (
    ExecutionError,
    QueryExecutionState,
    VisualizationResult,
)
from aquai.application.contracts.planning import OutputField, PlanningInput
from aquai.application.contracts.retrieval import DatasetProvenance, DocumentRetrievalResult
from aquai.application.contracts.routing import HistoryMessage, RoutingResult

AnswerStatus = Literal[
    "validated", "partial", "needs_clarification", "insufficient_coverage", "failed"
]
Language = Literal["es", "en"]
SegmentRole = Literal["result", "documented", "relation", "hypothesis"]


class EvidenceIssue(BaseModel):
    code: str
    message: str
    operation_id: str | None = None
    severity: Literal["warning", "error"] = "error"


class EvidenceReference(BaseModel):
    evidence_id: str
    operation_id: str
    tool: Literal["sql", "python"]
    kind: Literal["table", "visualization"]
    dataset_ids: list[UUID]
    rows: int
    is_requested_output: bool


class SourceReference(BaseModel):
    source_id: UUID
    dataset_id: UUID
    dataset_name: str
    version: int
    provenance: DatasetProvenance


class EvidenceValidationResult(BaseModel):
    status: AnswerStatus
    evidence_fingerprint: str
    usable_evidence: list[EvidenceReference] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    completed_operations: list[str] = Field(default_factory=list)
    pending_operations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    issues: list[EvidenceIssue] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    clarification_question: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    verified_at: datetime
    duration_ms: float = 0


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=2000)


class ResponseContextRecord(BaseModel):
    prompt_version: str = "aquai-response-v5"
    evidence: dict[str, JsonValue]
    sha256: str
    size_bytes: int
    budget_bytes: int
    omitted_rows: dict[str, int] = Field(default_factory=dict)
    sent_to_model: bool = False


class DraftSegment(BaseModel):
    """Párrafo redactado por el modelo; sólo se publica si la verificación lo respalda."""

    model_config = ConfigDict(extra="forbid", strict=True)
    role: SegmentRole
    document_ids: list[str] = Field(default_factory=list, max_length=12)
    fact_ids: list[str] = Field(default_factory=list, max_length=24)
    text: str = Field(min_length=1, max_length=2400)
    items: list[str] = Field(default_factory=list, max_length=8)


class ResponseDraft(BaseModel):
    """Redacción libre por párrafos, cada uno vinculado a la evidencia que usa."""

    model_config = ConfigDict(extra="forbid", strict=True)
    segments: list[DraftSegment] = Field(min_length=1, max_length=10)


class ResponseReply(BaseModel):
    draft: ResponseDraft
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class GenerationRecord(BaseModel):
    status: Literal["generated", "deterministic", "failed"]
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: float = 0
    context: ResponseContextRecord | None = None
    error: ExecutionError | None = None


class ResponseOperationReference(BaseModel):
    evidence_id: str
    operation_id: str
    tool: Literal["sql", "python"]
    dataset_ids: list[UUID]
    input_operation_ids: list[str]


class ResponseResultReference(BaseModel):
    evidence_id: str
    operation_id: str
    rows: int
    columns: list[OutputField] = Field(default_factory=list)
    location: Literal["execution.operations", "python_execution.operations"]
    complete: Literal[True] = True


class AnswerSegment(BaseModel):
    """Párrafo de la respuesta vinculado a la evidencia que respalda sus afirmaciones."""

    kind: Literal["paragraph", "note"] = "paragraph"
    role: SegmentRole | None = None
    text: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    attachment: str | None = None


class DocumentReference(BaseModel):
    """Fragmento citado: la posición y el hash del archivo lo ubican aunque se reprocese."""

    evidence_id: str
    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    position: int
    content_hash: str
    file_name: str
    title: str | None = None
    kind: Literal["text", "table", "figure"]
    page_start: int | None = None
    page_end: int | None = None
    section: list[str] = Field(default_factory=list)
    element_label: str | None = None
    excerpt: str
    doubts: list[str] = Field(default_factory=list)


class ReferenceValue(BaseModel):
    """Valor de referencia leído de un fragmento y su compatibilidad con un resultado calculado."""

    evidence_id: str
    result_field: str
    variable: str
    kind: Literal["guideline", "limit", "typical_range", "observed"]
    low: float | None = Field(default=None, allow_inf_nan=False)
    high: float | None = Field(default=None, allow_inf_nan=False)
    unit: str | None = None
    basis: str | None = None
    water_use: str | None = None
    condition: str | None = None
    same_place: bool = False
    quote: str
    status: Literal["compatible", "other_unit", "other_basis", "unit_unknown", "other_use"]
    detail: str | None = None


class ComparisonFact(BaseModel):
    """Contraste calculado entre un resultado y una referencia compatible."""

    text: str
    evidence_id: str
    document_id: str
    direction: Literal["above", "below", "within", "mixed"]


class ReferenceLine(BaseModel):
    """Línea de referencia verificada; se dibuja sobre el gráfico sin alterar la figura validada."""

    operation_id: str
    evidence_id: str
    axis: Literal["x", "y"] = "y"
    value: float = Field(allow_inf_nan=False)
    unit: str
    label: str


class UnitHypothesis(BaseModel):
    """Lectura condicional de un valor sin unidad confirmada; nunca reemplaza al dato original."""

    field: str
    label: str
    column: str
    observed: str
    unit: str
    equivalents: list[str] = Field(default_factory=list)
    documented: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)


class DocumentContrast(BaseModel):
    """Lo que se pudo contrastar de un resultado con la evidencia documental del turno."""

    references: list[ReferenceValue] = Field(default_factory=list)
    comparisons: list[ComparisonFact] = Field(default_factory=list)
    hypotheses: list[UnitHypothesis] = Field(default_factory=list)
    lines: list[ReferenceLine] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    proposed: int = 0
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: float = 0


class DocumentTrace(BaseModel):
    """Decisiones de la recuperación documental de un turno, para su trazabilidad."""

    purpose: Literal["question", "enrichment"]
    status: Literal["found", "no_relevant_evidence", "unavailable", "skipped"]
    queries: list[str] = Field(default_factory=list)
    candidates: int = 0
    discarded: int = 0
    fragments: int = 0
    selected: list[dict[str, JsonValue]] = Field(default_factory=list)
    rejected_segments: list[dict[str, JsonValue]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    response_type: Literal[
        "analytical", "general", "catalog", "clarification", "historical", "document"
    ] = "analytical"
    status: AnswerStatus | Literal["answered"]
    validation_status: AnswerStatus | None
    answer: str
    language: Language
    segments: list[AnswerSegment] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    operation_references: list[ResponseOperationReference] = Field(default_factory=list)
    results: list[ResponseResultReference] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    validation_issues: list[EvidenceIssue] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    pending_operations: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationResult] = Field(default_factory=list)
    document_references: list[DocumentReference] = Field(default_factory=list)
    reference_values: list[ReferenceValue] = Field(default_factory=list)
    reference_lines: list[ReferenceLine] = Field(default_factory=list)
    hypotheses: list[UnitHypothesis] = Field(default_factory=list)
    document_trace: DocumentTrace | None = None
    generation: GenerationRecord
    timings_ms: dict[str, float] = Field(default_factory=dict)
    planner_model: str | None = None
    planner_input_tokens: int | None = None
    planner_output_tokens: int | None = None
    routing: RoutingResult | None = None
    historical_message_ids: list[UUID] = Field(default_factory=list)


MAX_HISTORY_MESSAGES = 10


class QueryResponseInput(PlanningInput):
    conversation_context: list[ConversationMessage] = Field(default_factory=list, max_length=8)
    response_language: Language | None = None
    conversation_history: list[HistoryMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )


class QueryResponseState(QueryExecutionState):
    conversation_context: list[ConversationMessage] = Field(default_factory=list, max_length=8)
    response_language: Language | None = None
    validation: EvidenceValidationResult | None = None
    response: QueryResponse | None = None
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    conversation_history: list[HistoryMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )
    routing: RoutingResult | None = None
    document_retrieval: DocumentRetrievalResult | None = None
    document_contrast: DocumentContrast | None = None
