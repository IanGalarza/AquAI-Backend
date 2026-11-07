from uuid import UUID

from aquai.application.Services.streamTurnService import StreamTurnService, TurnEvents


class FollowGenerationHandler:
    #
    # Guarda el servicio que transmite los turnos en curso.
    #
    def __init__(self, streams: StreamTurnService):
        self._streams = streams

    #
    # Devuelve los eventos de una respuesta en curso de la conversación, desde su inicio.
    #
    async def handle(self, conversation_id: UUID, message_id: UUID) -> TurnEvents:
        return self._streams.follow(conversation_id, message_id)
