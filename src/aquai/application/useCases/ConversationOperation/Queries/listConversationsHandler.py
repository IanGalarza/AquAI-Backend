import asyncio

from aquai.application.contracts.conversations import ConversationPage
from aquai.application.interfaces.conversations import ConversationQueries


class ListConversationsHandler:
    def __init__(self, queries: ConversationQueries):
        self._queries = queries

    async def handle(self, *, page: int, page_size: int) -> ConversationPage:
        result = await asyncio.to_thread(self._queries.list, page=page, page_size=page_size)

        return ConversationPage.model_validate(result)
