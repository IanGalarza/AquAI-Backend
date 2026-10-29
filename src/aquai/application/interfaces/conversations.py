"""Puertos de historial y ejecución de una consulta; sin sesiones ni LangGraph."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.contracts.conversations import AssistantRecord, MessageView, TurnReservation


class ConversationQueries(Protocol):
    def list(self, *, page: int, page_size: int) -> dict: ...
    def get(self, conversation_id: UUID) -> dict: ...
    def messages(self, conversation_id: UUID, *, page: int, page_size: int) -> dict: ...
    def context(self, conversation_id: UUID, *, before_sequence: int) -> list[dict]: ...


class ConversationCommands(Protocol):
    def create(self, title: str | None) -> dict: ...
    def begin_turn(
        self, conversation_id: UUID, content: str, language: str | None,
    ) -> TurnReservation: ...
    def finish_turn(self, turn: TurnReservation, record: AssistantRecord) -> MessageView: ...


class ConversationQueryRunner(Protocol):
    async def run(self, query: QueryResponseInput) -> QueryResponseState: ...
