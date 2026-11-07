import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import replace

from aquai.application.contracts.answering import (
    ConversationMessage,
    QueryResponseInput,
)
from aquai.application.contracts.conversations import SendMessageOutcome
from aquai.application.conversations.context import build_history
from aquai.application.conversations.snapshot import (
    INTERRUPTED_GENERATION,
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
    TurnProgress,
)

_logger = logging.getLogger(__name__)
_RENEW_RETRY_SECONDS = 2


class AnswerTurnService:
    #
    # Guarda los comandos, consultas, ejecutor, plazo y reserva que usan los turnos de conversación.
    #
    def __init__(
        self,
        commands: ConversationCommands,
        queries: ConversationQueries,
        runner: ConversationQueryRunner,
        snapshots: ConversationSnapshotBuilder,
        *,
        timeout_seconds: float,
        lease_seconds: float,
    ):
        self._commands = commands
        self._queries = queries
        self._runner = runner
        self._snapshots = snapshots
        self._timeout_seconds = timeout_seconds
        self._renew_seconds = lease_seconds / 3

    #
    # Reserva el turno del mensaje y lo cierra como interrumpido si la petición se cancela.
    #
    async def reserve(self, conversation_id, request):
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
            await self._close_stopped(turn, by_user=False)
            raise

    #
    # Responde un turno reservado y lo cierra con la respuesta o con un fallo registrado. Mientras tanto
    # renueva su reserva, para que un reinicio la deje vencer pronto, e informa cada etapa que comienza.
    # Si se detiene, cancelled_by_user indica si lo pidió el usuario o si se interrumpió el servidor.
    #
    async def answer(
        self,
        turn,
        request,
        progress: TurnProgress | None = None,
        cancelled_by_user: Callable[[], bool] | None = None,
    ):
        lease = asyncio.create_task(self._keep_lease(turn))
        try:
            return await self._respond(turn, request, progress, cancelled_by_user)
        finally:
            lease.cancel()

    #
    # Ejecuta el análisis del turno y guarda su respuesta o el fallo que corresponda.
    #
    async def _respond(self, turn, request, progress, cancelled_by_user):
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
                    ),
                    progress,
                )

            record = await asyncio.to_thread(self._snapshots.build, state, turn)
            message = await asyncio.to_thread(self._commands.finish_turn, turn, record)
            outcome = "generation_failed" if record.status_name == "failed" else "completed"

            return SendMessageOutcome(message, outcome)
        except asyncio.CancelledError:
            by_user = cancelled_by_user is not None and cancelled_by_user()
            await self._close_stopped(turn, by_user=by_user)
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

    #
    # Carga el historial previo de la conversación para contextualizar la pregunta.
    #
    async def _load_history(self, turn):
        rows = await asyncio.to_thread(
            self._queries.context,
            turn.conversation_id,
            before_sequence=turn.user_sequence_number,
        )

        return build_history(rows, turn.question)

    #
    # Renueva la reserva del turno cada tercio de su duración hasta que el turno se cierre; si una
    # renovación falla, reintenta enseguida para no perder la reserva de un turno que sigue vivo.
    #
    async def _keep_lease(self, turn):
        delay = self._renew_seconds
        while True:
            await asyncio.sleep(delay)
            try:
                if not await asyncio.to_thread(self._commands.renew_turn, turn):
                    return
            except ConversationNotFoundError:
                return
            except Exception as error:
                if delay == self._renew_seconds:
                    _logger.warning(
                        "No se pudo renovar la reserva del turno %s: %s",
                        turn.assistant_message_id,
                        type(error).__name__,
                    )
                delay = _RENEW_RETRY_SECONDS
                continue

            delay = self._renew_seconds

    #
    # Cierra un turno detenido, como cancelado si lo pidió el usuario o como interrumpido si no,
    # ignorando conflictos si ya se había cerrado.
    #
    async def _close_stopped(self, turn, *, by_user):
        record = (
            failure_record(
                turn, "generation_cancelled", "La generación fue cancelada.", cancelled=True
            )
            if by_user
            else failure_record(turn, "interrupted_generation", INTERRUPTED_GENERATION)
        )
        try:
            await asyncio.shield(asyncio.to_thread(self._commands.finish_turn, turn, record))
        except (ConversationConflictError, ConversationNotFoundError):
            pass


#
# Agrega al turno el historial estructurado y la ventana visible para el redactor.
#
def _with_history(turn, history):
    visible_history = [item for item in history if item.text.strip() and not item.recalled]

    return replace(
        turn,
        structured_history=history,
        history=[ConversationMessage(role=item.role, text=item.text) for item in visible_history],
        history_message_ids=[item.message_id for item in visible_history],
    )


#
# Registra en el log un fallo inesperado de la consulta sin exponer su detalle.
#
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
