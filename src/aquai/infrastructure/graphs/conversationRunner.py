import asyncio

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.infrastructure.providers.planning import create_query_planning_graph


class LangGraphConversationRunner:
    #
    # Prepara la creación diferida del grafo de consulta y su lock.
    #
    def __init__(self):
        self._graph = None
        self._lock = asyncio.Lock()

    #
    # Ejecuta el grafo de consulta con la pregunta y devuelve el estado final.
    #
    async def run(self, query: QueryResponseInput) -> QueryResponseState:
        async with self._lock:
            if self._graph is None:
                self._graph = await asyncio.to_thread(create_query_planning_graph)

        result = await self._graph.ainvoke(query.model_dump())

        return QueryResponseState.model_validate(result)
