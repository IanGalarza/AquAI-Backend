from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.contracts.conversations import (
    AssistantRecord,
    MessageView,
    TurnReservation,
    TurnStage,
)

TurnProgress = Callable[[TurnStage], None]


class ConversationQueries(Protocol):
    #
    # Devuelve una página de conversaciones.
    #
    def list(self, *, page: int, page_size: int) -> dict: ...

    #
    # Devuelve los datos de una conversación.
    #
    def get(self, conversation_id: UUID) -> dict: ...

    #
    # Devuelve una página de mensajes de una conversación.
    #
    def messages(self, conversation_id: UUID, *, page: int, page_size: int) -> dict: ...

    #
    # Devuelve el historial de mensajes previo a una posición de la conversación.
    #
    def context(self, conversation_id: UUID, *, before_sequence: int) -> list[dict]: ...


class ConversationCommands(Protocol):
    #
    # Crea una conversación con título opcional.
    #
    def create(self, title: str | None) -> dict: ...

    #
    # Cambia el título de una conversación.
    #
    def rename(self, conversation_id: UUID, title: str) -> dict: ...

    #
    # Elimina una conversación.
    #
    def delete(self, conversation_id: UUID) -> None: ...

    #
    # Registra la pregunta del usuario y reserva el turno de respuesta en la conversación.
    #
    def begin_turn(
        self,
        conversation_id: UUID,
        content: str,
        language: str | None,
    ) -> TurnReservation: ...

    #
    # Extiende la reserva de un turno en curso; devuelve False si el turno ya se cerró.
    #
    def renew_turn(self, turn: TurnReservation) -> bool: ...

    #
    # Guarda la respuesta del asistente y cierra el turno reservado.
    #
    def finish_turn(self, turn: TurnReservation, record: AssistantRecord) -> MessageView: ...


class ConversationQueryRunner(Protocol):
    #
    # Ejecuta el flujo de análisis para una pregunta y devuelve el estado con la respuesta;
    # informa cada etapa que comienza a quien siga el progreso.
    #
    async def run(
        self, query: QueryResponseInput, progress: TurnProgress | None = None
    ) -> QueryResponseState: ...
