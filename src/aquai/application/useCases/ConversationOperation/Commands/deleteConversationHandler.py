import asyncio

from aquai.application.interfaces.conversations import ConversationCommands


class DeleteConversationHandler:
    #
    # Guarda los comandos de conversaciones que usa el manejador.
    #
    def __init__(self, commands: ConversationCommands):
        self._commands = commands

    #
    # Elimina una conversación y sus mensajes.
    #
    async def handle(self, conversation_id):
        await asyncio.to_thread(self._commands.delete, conversation_id)
