import asyncio

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.interfaces.conversations import TurnProgress
from aquai.infrastructure.graphs.queryPlanning import NODE_STAGES
from aquai.infrastructure.providers.planning import create_query_planning_graph


class LangGraphConversationRunner:
    #
    # Prepara la creación diferida del grafo de consulta y su lock.
    #
    def __init__(self):
        self._graph = None
        self._lock = asyncio.Lock()

    #
    # Ejecuta el grafo de consulta con la pregunta, informa la etapa de cada nodo que comienza
    # y devuelve el estado final, el mismo que daría ainvoke.
    #
    async def run(
        self, query: QueryResponseInput, progress: TurnProgress | None = None
    ) -> QueryResponseState:
        async with self._lock:
            if self._graph is None:
                self._graph = await asyncio.to_thread(create_query_planning_graph)

        result = None
        async for mode, data in self._graph.astream(
            query.model_dump(), stream_mode=["tasks", "values"]
        ):
            if mode == "values":
                result = data
            elif progress is not None and _starts_stage(data):
                progress(NODE_STAGES[data["name"]])

        return QueryResponseState.model_validate(result)


#
# Indica si el evento de tarea anuncia el comienzo de un nodo con etapa visible.
#
def _starts_stage(task: dict) -> bool:
    return "input" in task and task["name"] in NODE_STAGES
