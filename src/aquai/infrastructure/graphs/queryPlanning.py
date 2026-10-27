"""Retriever → Planner → SQL Agent condicional; nodos sin lógica de ejecución."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.planning import PlanningInput
from aquai.application.contracts.execution import QueryExecutionState
from aquai.application.useCases.queries.executeSqlPlanHandler import ExecuteSqlPlanHandler
from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.application.useCases.queries.retrieveDatasetsHandler import RetrieveDatasetsHandler


def next_after_planning(state: QueryExecutionState):
    if state.planning is not None and state.planning.status == "plannable" and any(
        step.tool == "sql" for step in state.planning.operations
    ):
        return "sql_agent"
    return END


def build_query_planning_graph(retriever: RetrieveDatasetsHandler, planner: PlanQueryHandler,
                               sql_agent: ExecuteSqlPlanHandler) -> CompiledStateGraph:
    async def retrieve(state: QueryExecutionState):
        return {"retrieval": await retriever.handle(state.question, limit=state.candidate_limit),
                "planning": None, "status": None, "execution": None,
                "execution_status": "not_executed", "pending_operations": []}

    async def plan(state: QueryExecutionState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")
        result = await planner.handle(state.question, state.retrieval)
        return {"planning": result, "status": result.status,
                "pending_operations": [step.id for step in result.operations if step.tool == "python"]}

    async def execute_sql(state: QueryExecutionState):
        if state.planning is None:
            raise ValueError("El SQL Agent requiere un plan verificado.")
        result = await sql_agent.handle(state.planning)
        return {"execution": result, "execution_status": result.status,
                "pending_operations": result.pending_operations}

    builder = StateGraph(QueryExecutionState, input_schema=PlanningInput)
    builder.add_node("retriever", retrieve)
    builder.add_node("planner", plan)
    builder.add_node("sql_agent", execute_sql)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_conditional_edges("planner", next_after_planning, {"sql_agent": "sql_agent", END: END})
    builder.add_edge("sql_agent", END)
    return builder.compile()
