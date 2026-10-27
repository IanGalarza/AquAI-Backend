"""Coordinación finita Retriever → Planner → SQL/Python, sin lógica de herramientas."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.planning import PlanningInput
from aquai.application.contracts.execution import QueryExecutionState
from aquai.application.execution import summarize_execution
from aquai.application.useCases.queries.executePythonPlanHandler import ExecutePythonPlanHandler
from aquai.application.useCases.queries.executeSqlPlanHandler import ExecuteSqlPlanHandler
from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.application.useCases.queries.retrieveDatasetsHandler import RetrieveDatasetsHandler


def next_after_planning(state: QueryExecutionState):
    if state.planning is not None and state.planning.status == "plannable":
        if any(step.tool == "sql" for step in state.planning.operations):
            return "sql_agent"
        if any(step.tool == "python" for step in state.planning.operations):
            return "python_agent"
    return END


def next_after_sql(state: QueryExecutionState):
    if (state.planning is not None and state.planning.status == "plannable"
            and any(step.tool == "python" for step in state.planning.operations)
            and not (state.execution and state.execution.error and state.execution.error.code == "invalid_plan")):
        return "python_agent"
    return END


def build_query_planning_graph(retriever: RetrieveDatasetsHandler, planner: PlanQueryHandler,
                               sql_agent: ExecuteSqlPlanHandler,
                               python_agent: ExecutePythonPlanHandler) -> CompiledStateGraph:
    async def retrieve(state: QueryExecutionState):
        return {"retrieval": await retriever.handle(state.question, limit=state.candidate_limit),
                "planning": None, "status": None, "execution": None,
                "python_execution": None, "visualizations": [], "execution_warnings": [],
                "execution_status": "not_executed", "pending_operations": []}

    async def plan(state: QueryExecutionState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")
        result = await planner.handle(state.question, state.retrieval)
        return {"planning": result, "status": result.status,
                "pending_operations": [step.id for step in result.operations]}

    async def execute_sql(state: QueryExecutionState):
        if state.planning is None:
            raise ValueError("El SQL Agent requiere un plan verificado.")
        result = await sql_agent.handle(state.planning)
        status, pending, warnings = summarize_execution(state.planning, result, None)
        return {"execution": result, "execution_status": status,
                "pending_operations": pending, "execution_warnings": warnings}

    async def execute_python(state: QueryExecutionState):
        if state.planning is None:
            raise ValueError("El Python Agent requiere un plan verificado.")
        result = await python_agent.handle(state.planning, state.execution, state.python_execution)
        status, pending, warnings = summarize_execution(state.planning, state.execution, result)
        return {"python_execution": result, "visualizations": result.visualizations,
                "execution_status": status, "pending_operations": pending, "execution_warnings": warnings}

    builder = StateGraph(QueryExecutionState, input_schema=PlanningInput)
    builder.add_node("retriever", retrieve)
    builder.add_node("planner", plan)
    builder.add_node("sql_agent", execute_sql)
    builder.add_node("python_agent", execute_python)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_conditional_edges("planner", next_after_planning,
        {"sql_agent": "sql_agent", "python_agent": "python_agent", END: END})
    builder.add_conditional_edges("sql_agent", next_after_sql, {"python_agent": "python_agent", END: END})
    builder.add_edge("python_agent", END)
    return builder.compile()
