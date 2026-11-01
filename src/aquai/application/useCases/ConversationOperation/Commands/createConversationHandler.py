import asyncio

from aquai.application.contracts.conversations import ConversationView
from aquai.application.interfaces.conversations import ConversationCommands


class CreateConversationHandler:
    #
    # Guarda los comandos de conversaciones que usa el manejador.
    #
    def __init__(self, commands: ConversationCommands):
        self._commands = commands

    #
    # Crea una conversación nueva con el título solicitado.
    #
    async def handle(self, request):
        result = await asyncio.to_thread(self._commands.create, request.title)

        return ConversationView.model_validate(result)
