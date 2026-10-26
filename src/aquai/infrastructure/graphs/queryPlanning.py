"""START → Retriever → Planner → END; los nodos sólo coordinan casos de uso."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.planning import PlanningInput, QueryPlanningState
from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.application.useCases.queries.retrieveDatasetsHandler import RetrieveDatasetsHandler


def build_query_planning_graph(retriever: RetrieveDatasetsHandler, planner: PlanQueryHandler) -> CompiledStateGraph:
    async def retrieve(state: QueryPlanningState):
        return {"retrieval": await retriever.handle(state.question, limit=state.candidate_limit),
                "planning": None, "status": None}

    async def plan(state: QueryPlanningState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")
        result = await planner.handle(state.question, state.retrieval)
        return {"planning": result, "status": result.status}

    builder = StateGraph(QueryPlanningState, input_schema=PlanningInput)
    builder.add_node("retriever", retrieve)
    builder.add_node("planner", plan)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_edge("planner", END)
    return builder.compile()
