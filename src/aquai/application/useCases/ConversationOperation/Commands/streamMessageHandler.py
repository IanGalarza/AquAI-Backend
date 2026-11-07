from aquai.application.Services.answerTurnService import AnswerTurnService
from aquai.application.Services.streamTurnService import StreamTurnService, TurnEvents


class StreamMessageHandler:
    #
    # Guarda el servicio que reserva y responde los turnos y el que los transmite.
    #
    def __init__(self, turns: AnswerTurnService, streams: StreamTurnService):
        self._turns = turns
        self._streams = streams

    #
    # Reserva el turno de un mensaje, lo responde en segundo plano y devuelve sus eventos para seguirlo.
    #
    async def handle(self, conversation_id, request) -> TurnEvents:
        turn = await self._turns.reserve(conversation_id, request)

        return self._streams.start(
            turn,
            lambda progress, cancelled: self._turns.answer(turn, request, progress, cancelled),
        )
