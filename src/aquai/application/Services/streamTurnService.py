import asyncio
import logging
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from aquai.application.contracts.conversations import (
    SendMessageOutcome,
    TurnDoneEvent,
    TurnReservation,
    TurnStage,
    TurnStageEvent,
    TurnStartedEvent,
)
from aquai.application.errors import GenerationNotFoundError
from aquai.application.interfaces.conversations import TurnProgress

_logger = logging.getLogger(__name__)


#
# Evento de un turno transmitido: su tipo y su contenido, o el error que impidió cerrarlo.
#
@dataclass(frozen=True)
class TurnEvent:
    kind: Literal["turn", "stage", "done", "error"]
    data: BaseModel | Exception


class TurnEvents:
    #
    # Publica el inicio del turno y prepara el historial de eventos para quienes lo sigan.
    #
    def __init__(self, turn: TurnReservation):
        self._started = perf_counter()
        self._history: list[TurnEvent] = []
        self._followers: set[asyncio.Queue] = set()
        self._stage: TurnStage | None = None
        self._closed = False

        self._publish(
            "turn",
            TurnStartedEvent(
                conversation_id=turn.conversation_id,
                user_message_id=turn.user_message_id,
                assistant_message_id=turn.assistant_message_id,
                started_at=datetime.now(timezone.utc),
            ),
        )

    #
    # Publica el comienzo de una etapa del análisis, salvo que continúe la anterior.
    #
    def stage(self, stage: TurnStage) -> None:
        if stage == self._stage:
            return

        self._stage = stage
        elapsed_ms = round((perf_counter() - self._started) * 1000)

        self._publish("stage", TurnStageEvent(stage=stage, elapsed_ms=elapsed_ms))

    #
    # Publica el cierre del turno y termina el seguimiento de todos sus seguidores.
    #
    def close(self, kind: Literal["done", "error"], data: BaseModel | Exception) -> None:
        self._publish(kind, data)
        self._closed = True

        for follower in self._followers:
            follower.put_nowait(None)

    #
    # Entrega los eventos ya publicados y los siguientes, hasta el cierre del turno.
    #
    async def follow(self) -> AsyncIterator[TurnEvent]:
        queue = asyncio.Queue()
        for event in self._history:
            queue.put_nowait(event)

        if self._closed:
            queue.put_nowait(None)
        else:
            self._followers.add(queue)

        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            self._followers.discard(queue)

    #
    # Guarda un evento en el historial y lo entrega a los seguidores actuales.
    #
    def _publish(self, kind, data) -> None:
        event = TurnEvent(kind, data)
        self._history.append(event)

        for follower in self._followers:
            follower.put_nowait(event)


@dataclass
class _LiveTurn:
    conversation_id: UUID
    events: TurnEvents
    task: asyncio.Task | None = None
    cancelled: bool = False


class StreamTurnService:
    #
    # Prepara el registro de los turnos que este proceso está respondiendo.
    #
    def __init__(self):
        self._turns: dict[UUID, _LiveTurn] = {}

    #
    # Responde en segundo plano un turno reservado, sin depender de la conexión de quien lo pidió,
    # y devuelve sus eventos para seguirlo. La respuesta recibe el progreso y si el usuario la canceló.
    #
    def start(
        self,
        turn: TurnReservation,
        answer: Callable[[TurnProgress, Callable[[], bool]], Awaitable[SendMessageOutcome]],
    ) -> TurnEvents:
        live = _LiveTurn(turn.conversation_id, TurnEvents(turn))
        live.task = asyncio.create_task(self._answer(turn, live, answer))
        self._turns[turn.assistant_message_id] = live

        return live.events

    #
    # Devuelve los eventos de una respuesta que este proceso sigue generando en la conversación.
    #
    def follow(self, conversation_id: UUID, message_id: UUID) -> TurnEvents:
        return self._live(conversation_id, message_id).events

    #
    # Pide cancelar una respuesta en curso una sola vez; su cierre llega a quienes la siguen. La cancelación
    # se agenda detrás del primer paso de la tarea para que el turno siempre se cierre en la base.
    #
    def cancel(self, conversation_id: UUID, message_id: UUID) -> None:
        turn = self._live(conversation_id, message_id)
        if turn.cancelled:
            return

        turn.cancelled = True
        asyncio.get_running_loop().call_soon(turn.task.cancel)

    #
    # Devuelve el turno en curso de un mensaje de la conversación, o falla si ya no se genera aquí.
    #
    def _live(self, conversation_id: UUID, message_id: UUID) -> _LiveTurn:
        turn = self._turns.get(message_id)
        if turn is None or turn.conversation_id != conversation_id:
            raise GenerationNotFoundError("La respuesta ya no se está generando.")

        return turn

    #
    # Responde el turno y publica su cierre: el mensaje guardado, la cancelación o interrupción,
    # o el error que impidió cerrarlo.
    #
    async def _answer(self, turn, live, answer) -> None:
        events = live.events
        try:
            outcome = await answer(events.stage, lambda: live.cancelled)
            events.close("done", TurnDoneEvent(outcome=outcome.outcome, message=outcome.message))
        except asyncio.CancelledError:
            stopped = "cancelled" if live.cancelled else "execution_failed"
            events.close("done", TurnDoneEvent(outcome=stopped))
        except Exception as error:
            _log_unclosed_turn(turn, error)
            events.close("error", error)
        finally:
            self._turns.pop(turn.assistant_message_id, None)


#
# Registra en el log el error que impidió cerrar un turno, con su ubicación y sin exponer su detalle.
#
def _log_unclosed_turn(turn, error):
    frames = traceback.extract_tb(error.__traceback__)
    frame = frames[-1] if frames else None

    _logger.error(
        "No se pudo cerrar el turno %s: %s en %s:%s",
        turn.assistant_message_id,
        type(error).__name__,
        frame.name if frame else "unknown",
        frame.lineno if frame else 0,
    )
