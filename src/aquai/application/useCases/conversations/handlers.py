"""Casos de uso del historial y envío; ninguna transacción abarca el grafo."""

import asyncio
import logging
import traceback

from aquai.application.contracts.answering import QueryResponseInput
from aquai.application.contracts.conversations import ConversationPage, ConversationView, MessagePage, SendMessageOutcome
from aquai.application.conversations.snapshot import failure_record
from aquai.application.conversations.snapshot import ConversationSnapshotBuilder
from aquai.application.errors import ConversationConflictError, ConversationStorageError
from aquai.application.interfaces.conversations import ConversationQueryRunner, ConversationRepository


_logger = logging.getLogger(__name__)


class ConversationHistoryHandler:
    def __init__(self, repository: ConversationRepository):
        self._repository = repository

    async def create(self, request):
        return ConversationView.model_validate(await asyncio.to_thread(self._repository.create, request.title))

    async def list(self, *, page, page_size):
        return ConversationPage.model_validate(await asyncio.to_thread(self._repository.list, page=page, page_size=page_size))

    async def get(self, conversation_id):
        return ConversationView.model_validate(await asyncio.to_thread(self._repository.get, conversation_id))

    async def messages(self, conversation_id, *, page, page_size):
        return MessagePage.model_validate(await asyncio.to_thread(self._repository.messages, conversation_id, page=page, page_size=page_size))


class SendMessageHandler:
    def __init__(self, repository: ConversationRepository, runner: ConversationQueryRunner,
                 snapshots: ConversationSnapshotBuilder, *, timeout_seconds: float):
        self._repository, self._runner, self._snapshots = repository, runner, snapshots
        self._timeout_seconds = timeout_seconds

    async def handle(self, conversation_id, request):
        # La reserva se confirma antes de cualquier inferencia.
        reservation = asyncio.create_task(asyncio.to_thread(
            self._repository.begin_turn, conversation_id, request.content, request.response_language))
        try:
            turn = await asyncio.shield(reservation)
        except asyncio.CancelledError:
            turn = await reservation
            await asyncio.shield(asyncio.to_thread(self._repository.finish_turn, turn,
                failure_record(turn, "generation_cancelled", "La generación fue cancelada.", cancelled=True)))
            raise
        try:
            async with asyncio.timeout(self._timeout_seconds):
                state = await self._runner.run(QueryResponseInput(question=request.content,
                    conversation_context=turn.history, response_language=request.response_language))
            record = await asyncio.to_thread(self._snapshots.build, state, turn)
            message = await asyncio.to_thread(self._repository.finish_turn, turn, record)
            return SendMessageOutcome(message, "generation_failed" if record.status_name == "failed" else "completed")
        except asyncio.CancelledError:
            try:
                await asyncio.shield(asyncio.to_thread(self._repository.finish_turn, turn,
                    failure_record(turn, "generation_cancelled", "La generación fue cancelada.", cancelled=True)))
            except ConversationConflictError:
                pass  # Una recuperación por vencimiento ya cerró este turno.
            raise
        except ConversationConflictError:
            raise
        except TimeoutError:
            failure = failure_record(turn, "generation_timeout", "La generación superó el tiempo permitido. El mensaje del usuario se conservó.")
            outcome = "timed_out"
        except ConversationStorageError:
            failure = failure_record(turn, "response_persistence_failed", "No se pudo conservar una respuesta consistente dentro de los límites del historial.")
            outcome = "execution_failed"
        except Exception as error:
            frames = traceback.extract_tb(error.__traceback__)
            frame = frames[-1] if frames else None
            _logger.error("Falló la consulta %s: %s en %s:%s", turn.assistant_message_id,
                type(error).__name__, frame.name if frame else "unknown", frame.lineno if frame else 0)
            failure = failure_record(turn, "query_execution_failed", "No se pudo completar la consulta. El mensaje del usuario se conservó.")
            outcome = "execution_failed"
        message = await asyncio.to_thread(self._repository.finish_turn, turn, failure)
        return SendMessageOutcome(message, outcome)
