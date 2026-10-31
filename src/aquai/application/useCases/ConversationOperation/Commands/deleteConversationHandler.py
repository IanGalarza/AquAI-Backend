import asyncio

from aquai.application.interfaces.conversations import ConversationCommands


class DeleteConversationHandler:
    def __init__(self, commands: ConversationCommands):
        self._commands = commands

    async def handle(self, conversation_id):
        await asyncio.to_thread(self._commands.delete, conversation_id)
