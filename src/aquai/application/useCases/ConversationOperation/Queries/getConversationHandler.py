import asyncio
from uuid import UUID

from aquai.application.contracts.conversations import ConversationView
from aquai.application.interfaces.conversations import ConversationQueries


class GetConversationHandler:
    def __init__(self, queries: ConversationQueries):
        self._queries = queries

    async def handle(self, conversation_id: UUID) -> ConversationView:
        result = await asyncio.to_thread(self._queries.get, conversation_id)

        return ConversationView.model_validate(result)
