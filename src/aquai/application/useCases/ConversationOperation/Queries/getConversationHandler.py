import asyncio
from uuid import UUID

from aquai.application.contracts.conversations import ConversationView
from aquai.application.interfaces.conversations import ConversationQueries


class GetConversationHandler:
    #
    # Guarda las consultas de conversaciones que usa el manejador.
    #
    def __init__(self, queries: ConversationQueries):
        self._queries = queries

    #
    # Devuelve la vista de una conversación.
    #
    async def handle(self, conversation_id: UUID) -> ConversationView:
        result = await asyncio.to_thread(self._queries.get, conversation_id)

        return ConversationView.model_validate(result)
