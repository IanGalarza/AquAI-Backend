import asyncio

from aquai.application.contracts.conversations import ConversationView
from aquai.application.interfaces.conversations import ConversationCommands


class RenameConversationHandler:
    def __init__(self, commands: ConversationCommands):
        self._commands = commands

    async def handle(self, conversation_id, request):
        result = await asyncio.to_thread(self._commands.rename, conversation_id, request.title)

        return ConversationView.model_validate(result)
