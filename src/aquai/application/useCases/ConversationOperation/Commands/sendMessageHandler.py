import asyncio
import logging
import traceback
from dataclasses import replace

from aquai.application.contracts.answering import (
    ConversationMessage,
    QueryResponseInput,
)
from aquai.application.contracts.conversations import SendMessageOutcome
from aquai.application.conversations.context import build_history
from aquai.application.conversations.snapshot import (
    ConversationSnapshotBuilder,
    failure_record,
)
from aquai.application.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
    ConversationStorageError,
)
from aquai.application.interfaces.conversations import (
    ConversationCommands,
    ConversationQueries,
    ConversationQueryRunner,
)

_logger = logging.getLogger(__name__)


class SendMessageHandler:
    def __init__(
        self,
        commands: ConversationCommands,
        queries: ConversationQueries,
        runner: ConversationQueryRunner,
        snapshots: ConversationSnapshotBuilder,
        *,
        timeout_seconds: float,
    ):
        self._commands = commands
        self._queries = queries
        self._runner = runner
        self._snapshots = snapshots
        self._timeout_seconds = timeout_seconds

    async def handle(self, conversation_id, request):
        turn = await self._reserve_turn(conversation_id, request)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                history = await self._load_history(turn)
                turn = _with_history(turn, history)
                state = await self._runner.run(
                    QueryResponseInput(
                        question=request.content,
                        conversation_context=turn.history,
                        conversation_history=history,
                        response_language=request.response_language,
                    )
                )

            record = await asyncio.to_thread(self._snapshots.build, state, turn)
            message = await asyncio.to_thread(self._commands.finish_turn, turn, record)
            outcome = "generation_failed" if record.status_name == "failed" else "completed"

            return SendMessageOutcome(message, outcome)
        except asyncio.CancelledError:
            await self._cancel(turn)
            raise
        except ConversationConflictError:
            raise
        except TimeoutError:
            failure = failure_record(
                turn,
                "generation_timeout",
                "La generación superó el tiempo permitido. El mensaje del usuario se conservó.",
            )
            outcome = "timed_out"
        except ConversationStorageError:
            failure = failure_record(
                turn,
                "response_persistence_failed",
                "No se pudo conservar una respuesta consistente dentro de los límites del historial.",
            )
            outcome = "execution_failed"
        except Exception as error:
            _log_unexpected_failure(turn, error)
            failure = failure_record(
                turn,
                "query_execution_failed",
                "No se pudo completar la consulta. El mensaje del usuario se conservó.",
            )
            outcome = "execution_failed"

        message = await asyncio.to_thread(self._commands.finish_turn, turn, failure)

        return SendMessageOutcome(message, outcome)

    async def _reserve_turn(self, conversation_id, request):
        # La reserva se protege con shield: si la petición se cancela, el turno ya reservado
        # se cierra como cancelado en lugar de quedar abierto.
        reservation = asyncio.create_task(
            asyncio.to_thread(
                self._commands.begin_turn,
                conversation_id,
                request.content,
                request.response_language,
            )
        )

        try:
            return await asyncio.shield(reservation)
        except asyncio.CancelledError:
            turn = await reservation
            await self._cancel(turn)
            raise

    async def _load_history(self, turn):
        rows = await asyncio.to_thread(
            self._queries.context,
            turn.conversation_id,
            before_sequence=turn.user_sequence_number,
        )

        return build_history(rows)

    async def _cancel(self, turn):
        try:
            record = failure_record(
                turn, "generation_cancelled", "La generación fue cancelada.", cancelled=True
            )
            await asyncio.shield(asyncio.to_thread(self._commands.finish_turn, turn, record))
        except (ConversationConflictError, ConversationNotFoundError):
            pass  # Una recuperación de la reserva ya cerró el turno, o la conversación se eliminó.


def _with_history(turn, history):
    visible_history = [item for item in history if item.text.strip()]

    return replace(
        turn,
        structured_history=history,
        history=[ConversationMessage(role=item.role, text=item.text) for item in visible_history],
        history_message_ids=[item.message_id for item in visible_history],
    )


def _log_unexpected_failure(turn, error):
    frames = traceback.extract_tb(error.__traceback__)
    frame = frames[-1] if frames else None

    _logger.error(
        "Falló la consulta %s: %s en %s:%s",
        turn.assistant_message_id,
        type(error).__name__,
        frame.name if frame else "unknown",
        frame.lineno if frame else 0,
    )
