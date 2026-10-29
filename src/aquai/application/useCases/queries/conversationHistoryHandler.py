"""Consultas del historial; no crea ni modifica conversaciones."""

import asyncio

from aquai.application.contracts.conversations import (
    ConversationPage, ConversationView, MessagePage,
)
from aquai.application.interfaces.conversations import ConversationQueries


class ConversationHistoryHandler:
    def __init__(self, queries: ConversationQueries):
        self._queries = queries

    async def list(self, *, page, page_size):
        result = await asyncio.to_thread(
            self._queries.list, page=page, page_size=page_size
        )
        return ConversationPage.model_validate(result)

    async def get(self, conversation_id):
        result = await asyncio.to_thread(self._queries.get, conversation_id)
        return ConversationView.model_validate(result)

    async def messages(self, conversation_id, *, page, page_size):
        result = await asyncio.to_thread(
            self._queries.messages, conversation_id, page=page, page_size=page_size
        )
        return MessagePage.model_validate(result)
