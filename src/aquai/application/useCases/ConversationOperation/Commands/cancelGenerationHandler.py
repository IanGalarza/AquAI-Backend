from uuid import UUID

from aquai.application.Services.streamTurnService import StreamTurnService


class CancelGenerationHandler:
    #
    # Guarda el servicio que transmite los turnos en curso.
    #
    def __init__(self, streams: StreamTurnService):
        self._streams = streams

    #
    # Cancela la respuesta que se está generando para un mensaje de la conversación.
    #
    async def handle(self, conversation_id: UUID, message_id: UUID) -> None:
        self._streams.cancel(conversation_id, message_id)
