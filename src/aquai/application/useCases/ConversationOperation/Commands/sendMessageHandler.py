from aquai.application.Services.answerTurnService import AnswerTurnService


class SendMessageHandler:
    #
    # Guarda el servicio que reserva y responde los turnos.
    #
    def __init__(self, turns: AnswerTurnService):
        self._turns = turns

    #
    # Procesa un mensaje del usuario y devuelve la respuesta, o un fallo registrado.
    #
    async def handle(self, conversation_id, request):
        turn = await self._turns.reserve(conversation_id, request)

        return await self._turns.answer(turn, request)
