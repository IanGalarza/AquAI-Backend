"""Contratos HTTP e instantáneas persistentes, sin objetos de infraestructura."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.contracts.answering import ConversationMessage, GenerationRecord, Language, QueryResponse
from aquai.application.contracts.execution import ValueExclusion
from aquai.application.contracts.planning import OutputField


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=200)


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(min_length=1, max_length=6000)
    response_language: Language | None = None


class ConversationView(BaseModel):
    id: UUID
    title: str
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime | None


class ConversationPage(BaseModel):
    items: list[ConversationView]
    total: int
    page: int
    page_size: int


class MessageStatusView(BaseModel):
    id: int
    name: str


class MessageSourceView(BaseModel):
    id: UUID
    source_id: UUID
    dataset_id: UUID | None
    reference_order: int
    details: dict[str, JsonValue]


class MessageView(BaseModel):
    id: UUID
    conversation_id: UUID
    sequence_number: int
    role: str
    content: str
    content_data: dict[str, JsonValue]
    status: MessageStatusView
    created_at: datetime
    updated_at: datetime | None
    sources: list[MessageSourceView] = Field(default_factory=list)


class MessagePage(BaseModel):
    conversation_id: UUID
    items: list[MessageView]
    total: int
    page: int
    page_size: int


class StoredTable(BaseModel):
    evidence_id: str
    operation_id: str
    columns: list[OutputField]
    rows: list[dict[str, JsonValue]] = Field(default_factory=list)
    total_rows: int
    stored_rows: int
    storage: Literal["inline", "preview", "metadata_only"]
    complete: bool
    warnings: list[str] = Field(default_factory=list)
    rows_processed: int | None = None
    rows_filtered: int = 0
    exclusions: list[ValueExclusion] = Field(default_factory=list)
    exclusions_complete: bool = True
    # No location a execution.operations: los datos efímeros no son recursos HTTP.


class StoredGeneration(GenerationRecord):
    context: None = None
    context_record: dict[str, JsonValue] | None = None


class StoredQueryResponse(QueryResponse):
    results: list[StoredTable] = Field(default_factory=list)
    generation: StoredGeneration


class AssistantSnapshot(BaseModel):
    schema_version: Literal[1] = 1
    kind: Literal["assistant_response"] = "assistant_response"
    response: StoredQueryResponse
    operations: list[dict[str, JsonValue]] = Field(default_factory=list)
    context_policy: dict[str, JsonValue]
    storage_warnings: list[str] = Field(default_factory=list)
    omitted_visualizations: list[dict[str, JsonValue]] = Field(default_factory=list)
    validated_at: datetime


@dataclass(frozen=True)
class TurnReservation:
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    history: list[ConversationMessage]
    history_message_ids: list[UUID]
    question: str


@dataclass(frozen=True)
class SourceAssociation:
    source_id: UUID
    dataset_id: UUID
    reference_order: int
    details: dict


@dataclass(frozen=True)
class AssistantRecord:
    content: str
    content_data: dict
    sources: list[SourceAssociation]
    status_name: Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True)
class SendMessageOutcome:
    message: MessageView
    outcome: Literal["completed", "generation_failed", "execution_failed", "timed_out"]
