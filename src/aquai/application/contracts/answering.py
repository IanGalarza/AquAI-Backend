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
from aquai.application.contracts.retrieval import DatasetProvenance
from aquai.application.contracts.routing import HistoryMessage, RoutingResult

AnswerStatus = Literal[
    "validated", "partial", "needs_clarification", "insufficient_coverage", "failed"
]
Language = Literal["es", "en"]


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
    prompt_version: str = "aquai-response-v3"
    evidence: dict[str, JsonValue]
    sha256: str
    size_bytes: int
    budget_bytes: int
    omitted_rows: dict[str, int] = Field(default_factory=dict)
    sent_to_model: bool = False


class DraftSentence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sentence_id: str
    variant: Literal["direct", "conversational"]
    evidence_ids: list[str] = Field(min_length=1, max_length=12)


class ResponseDraft(BaseModel):
    """Redacción por enunciados verificables; no admite afirmaciones libres."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sentences: list[DraftSentence] = Field(min_length=1, max_length=24)


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
    # La respuesta no copia tablas extensas que ya permanecen en el estado.
    complete: Literal[True] = True


class AnswerSegment(BaseModel):
    """Párrafo de la respuesta vinculado a la evidencia que respalda sus afirmaciones."""

    kind: Literal["paragraph", "note"] = "paragraph"
    text: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    attachment: str | None = None


class QueryResponse(BaseModel):
    response_type: Literal["analytical", "general", "catalog", "clarification", "historical"] = (
        "analytical"
    )
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
    generation: GenerationRecord
    timings_ms: dict[str, float] = Field(default_factory=dict)
    planner_model: str | None = None
    planner_input_tokens: int | None = None
    planner_output_tokens: int | None = None
    routing: RoutingResult | None = None
    historical_message_ids: list[UUID] = Field(default_factory=list)


# Ventana reciente de mensajes más los análisis anteriores que la pregunta recupera.
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
