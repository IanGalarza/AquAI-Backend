"""Coordinación finita de recuperación, ejecución, validación y respuesta."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from time import perf_counter

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.execution import summarize_execution
from aquai.application.useCases.queries.executePythonPlanHandler import ExecutePythonPlanHandler
from aquai.application.useCases.queries.executeSqlPlanHandler import ExecuteSqlPlanHandler
from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.application.useCases.queries.retrieveDatasetsHandler import RetrieveDatasetsHandler
from aquai.application.useCases.queries.validateEvidenceHandler import ValidateEvidenceHandler
from aquai.application.useCases.queries.generateResponseHandler import GenerateResponseHandler


def next_after_planning(state: QueryResponseState):
    if state.planning is not None and state.planning.status == "plannable":
        if any(step.tool == "sql" for step in state.planning.operations):
            return "sql_agent"
        if any(step.tool == "python" for step in state.planning.operations):
            return "python_agent"
    return "validator"


def next_after_sql(state: QueryResponseState):
    if (state.planning is not None and state.planning.status == "plannable"
            and any(step.tool == "python" for step in state.planning.operations)
            and not (state.execution and state.execution.error and state.execution.error.code == "invalid_plan")):
        return "python_agent"
    return "validator"


def build_query_planning_graph(retriever: RetrieveDatasetsHandler, planner: PlanQueryHandler,
                               sql_agent: ExecuteSqlPlanHandler,
                               python_agent: ExecutePythonPlanHandler,
                               validator: ValidateEvidenceHandler,
                               response_generator: GenerateResponseHandler) -> CompiledStateGraph:
    async def retrieve(state: QueryResponseState):
        started = perf_counter()
        result = await retriever.handle(state.question, limit=state.candidate_limit)
        return {"retrieval": result,
                "planning": None, "status": None, "execution": None,
                "python_execution": None, "visualizations": [], "execution_warnings": [],
                "execution_status": "not_executed", "pending_operations": [], "validation": None, "response": None,
                "stage_durations_ms": {"retriever": round((perf_counter() - started) * 1000, 3)}}

    async def plan(state: QueryResponseState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")
        started = perf_counter()
        result = await planner.handle(state.question, state.retrieval)
        return {"planning": result, "status": result.status,
                "pending_operations": [step.id for step in result.operations],
                "stage_durations_ms": {**state.stage_durations_ms, "planner": round((perf_counter() - started) * 1000, 3)}}

    async def execute_sql(state: QueryResponseState):
        if state.planning is None:
            raise ValueError("El SQL Agent requiere un plan verificado.")
        result = await sql_agent.handle(state.planning)
        status, pending, warnings = summarize_execution(state.planning, result, None)
        return {"execution": result, "execution_status": status,
                "pending_operations": pending, "execution_warnings": warnings}

    async def execute_python(state: QueryResponseState):
        if state.planning is None:
            raise ValueError("El Python Agent requiere un plan verificado.")
        result = await python_agent.handle(state.planning, state.execution, state.python_execution)
        status, pending, warnings = summarize_execution(state.planning, state.execution, result)
        return {"python_execution": result, "visualizations": result.visualizations,
                "execution_status": status, "pending_operations": pending, "execution_warnings": warnings}

    async def validate(state: QueryResponseState):
        result = await validator.handle(state)
        return {"validation": result, "pending_operations": result.pending_operations}

    async def respond(state: QueryResponseState):
        return {"response": await response_generator.handle(state)}

    builder = StateGraph(QueryResponseState, input_schema=QueryResponseInput)
    builder.add_node("retriever", retrieve)
    builder.add_node("planner", plan)
    builder.add_node("sql_agent", execute_sql)
    builder.add_node("python_agent", execute_python)
    builder.add_node("validator", validate)
    builder.add_node("response_generator", respond)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "planner")
    builder.add_conditional_edges("planner", next_after_planning,
        {"sql_agent": "sql_agent", "python_agent": "python_agent", "validator": "validator"})
    builder.add_conditional_edges("sql_agent", next_after_sql, {"python_agent": "python_agent", "validator": "validator"})
    builder.add_edge("python_agent", "validator")
    builder.add_edge("validator", "response_generator")
    builder.add_edge("response_generator", END)
    return builder.compile()
