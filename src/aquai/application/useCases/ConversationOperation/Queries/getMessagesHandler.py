import asyncio
from uuid import UUID

from aquai.application.contracts.conversations import MessagePage
from aquai.application.interfaces.conversations import ConversationQueries


class GetMessagesHandler:
    def __init__(self, queries: ConversationQueries):
        self._queries = queries

    async def handle(self, conversation_id: UUID, *, page: int, page_size: int) -> MessagePage:
        result = await asyncio.to_thread(
            self._queries.messages, conversation_id, page=page, page_size=page_size
        )

        return MessagePage.model_validate(result)
