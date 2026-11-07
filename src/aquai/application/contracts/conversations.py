from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.contracts.answering import (
    ConversationMessage,
    GenerationRecord,
    Language,
    QueryResponse,
)
from aquai.application.contracts.execution import ValueExclusion
from aquai.application.contracts.planning import OutputField
from aquai.application.contracts.routing import HistoryMessage


#
# Cuerpo de la solicitud para crear una conversación: título opcional.
#
class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str | None = Field(default=None, min_length=1, max_length=200)


#
# Cuerpo de la solicitud para renombrar una conversación: nuevo título.
#
class RenameConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=200, pattern=r"^[^\p{Cc}]+$")


#
# Cuerpo de la solicitud para enviar un mensaje: pregunta e idioma de respuesta.
#
class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(min_length=1, max_length=6000)
    response_language: Language | None = None


#
# Respuesta con los datos de una conversación.
#
class ConversationView(BaseModel):
    id: UUID
    title: str
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime | None


#
# Respuesta con una página de conversaciones y sus totales.
#
class ConversationPage(BaseModel):
    items: list[ConversationView]
    total: int
    page: int
    page_size: int


#
# Respuesta con el estado de un mensaje.
#
class MessageStatusView(BaseModel):
    id: int
    name: str


#
# Respuesta con la fuente asociada a un mensaje.
#
class MessageSourceView(BaseModel):
    id: UUID
    source_id: UUID
    dataset_id: UUID | None
    reference_order: int
    details: dict[str, JsonValue]


#
# Variable leída y su ubicación en la planilla original.
#
class CitationVariable(BaseModel):
    """Variable leída y su ubicación en la planilla original (columna G, celdas G2:G1113)."""

    name: str
    column: str | None = None
    row: int | None = None
    cells: str | None = None


#
# Respuesta con la fuente citada de una afirmación: archivo, hoja, filas y variables.
#
class CitationView(BaseModel):
    """Fuente legible de una afirmación: archivo, hoja, filas de datos y variables utilizadas."""

    id: str
    kind: Literal["dataset", "document"] = "dataset"
    source_id: UUID
    dataset_id: UUID | None = None
    version: int | None = None
    file_name: str
    file_type: str
    sheet: str | None = None
    dataset: str = ""
    first_row: int | None = None
    last_row: int | None = None
    variables: list[CitationVariable] = Field(default_factory=list)
    title: str | None = None
    pages: str | None = None
    section: str | None = None
    excerpt: str | None = None
    doubtful: bool = False


#
# Respuesta con un fragmento del mensaje, sus citas y su tabla o gráfico asociado.
#
class SegmentView(BaseModel):
    kind: Literal["paragraph", "note"] = "paragraph"
    role: Literal["result", "documented", "relation", "hypothesis"] | None = None
    text: str
    items: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    table: str | None = None
    chart: str | None = None


#
# Respuesta con los datos de una columna de una tabla de resultados.
#
class ColumnView(BaseModel):
    name: str
    label: str
    unit: str | None = None
    unit_documented: bool = False
    numeric: bool = False
    time_grain: Literal["day", "month", "year"] | None = None


#
# Respuesta con una tabla de resultados y sus citas.
#
class TableView(BaseModel):
    evidence_id: str
    columns: list[ColumnView]
    citations: list[str] = Field(default_factory=list)


#
# Respuesta con un gráfico, sus citas y sus títulos legibles.
#
class ChartReferenceLine(BaseModel):
    axis: Literal["x", "y"] = "y"
    value: float
    label: str
    citations: list[str] = Field(default_factory=list)


class ChartView(BaseModel):
    operation_id: str
    citations: list[str] = Field(default_factory=list)
    x_title: str | None = None
    y_title: str | None = None
    legend_title: str | None = None
    reference_lines: list[ChartReferenceLine] = Field(default_factory=list)


#
# Respuesta con la presentación estructurada de un mensaje.
#
class MessagePresentation(BaseModel):
    """Lectura del mensaje sin interpretar texto en el cliente; el detalle técnico queda en content_data."""

    segments: list[SegmentView] = Field(default_factory=list)
    citations: list[CitationView] = Field(default_factory=list)
    tables: list[TableView] = Field(default_factory=list)
    charts: list[ChartView] = Field(default_factory=list)
    trailing_tables: list[str] = Field(default_factory=list)
    trailing_charts: list[str] = Field(default_factory=list)
    duration_ms: int | None = None


#
# Respuesta con los datos de un mensaje de la conversación.
#
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
    presentation: MessagePresentation | None = None


#
# Respuesta con una página de mensajes de una conversación.
#
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
    validated_at: datetime | None


@dataclass(frozen=True)
class TurnReservation:
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    history: list[ConversationMessage]
    history_message_ids: list[UUID]
    question: str
    user_sequence_number: int = 0
    structured_history: list[HistoryMessage] = field(default_factory=list)


@dataclass(frozen=True)
class SourceAssociation:
    source_id: UUID
    dataset_id: UUID | None
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
