"""Router conversacional y coordinación finita de los casos de uso existentes."""

import asyncio
from collections.abc import Callable
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.execution import summarize_execution
from aquai.application.useCases.queries.executePythonPlanHandler import ExecutePythonPlanHandler
from aquai.application.useCases.queries.executeSqlPlanHandler import ExecuteSqlPlanHandler
from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.application.useCases.queries.retrieveDatasetsHandler import RetrieveDatasetsHandler
from aquai.application.useCases.queries.validateEvidenceHandler import ValidateEvidenceHandler
from aquai.application.useCases.queries.generateResponseHandler import GenerateResponseHandler
from aquai.application.useCases.queries.generateDirectResponseHandler import (
    GenerateDirectResponseHandler,
)
from aquai.application.useCases.queries.routeQueryHandler import RouteQueryHandler


def analytical_state(state: QueryResponseState) -> QueryResponseState:
    """La pregunta original permanece intacta en el estado y en Message.content."""
    question = state.routing.contextualized_question if state.routing else state.question
    return state.model_copy(update={"question": question})


def next_after_routing(state: QueryResponseState):
    return "retriever" if state.routing.route == "data" else "direct_response"


def next_after_planning(state: QueryResponseState):
    if state.planning is not None and state.planning.status == "plannable":
        if any(step.tool == "sql" for step in state.planning.operations):
            return "sql_agent"
        if any(step.tool == "python" for step in state.planning.operations):
            return "python_agent"
    return "validator"


def next_after_sql(state: QueryResponseState):
    if (
        state.planning is not None and state.planning.status == "plannable"
        and any(step.tool == "python" for step in state.planning.operations)
        and not (
            state.execution and state.execution.error
            and state.execution.error.code == "invalid_plan"
        )
    ):
        return "python_agent"
    return "validator"


def build_query_planning_graph(
    retriever_factory: Callable[[], RetrieveDatasetsHandler],
    planner: PlanQueryHandler,
    sql_agent: ExecuteSqlPlanHandler,
    python_agent: ExecutePythonPlanHandler,
    validator: ValidateEvidenceHandler,
    response_generator: GenerateResponseHandler,
    router: RouteQueryHandler,
    direct_response: GenerateDirectResponseHandler,
) -> CompiledStateGraph:
    retriever = None
    retrieval_lock = asyncio.Lock()

    async def route(state: QueryResponseState):
        result = await router.handle(state)
        return {
            "routing": result, "response_language": result.language,
            "stage_durations_ms": {"router": result.duration_ms},
        }

    async def respond_directly(state: QueryResponseState):
        return {"response": await direct_response.handle(state)}

    async def retrieve(state: QueryResponseState):
        nonlocal retriever
        started = perf_counter()
        async with retrieval_lock:
            if retriever is None:
                retriever = await asyncio.to_thread(retriever_factory)
        query = analytical_state(state)
        result = await retriever.handle(
            query.question, limit=state.candidate_limit,
            dataset_bindings=state.routing.dataset_bindings,
        )
        return {
            "retrieval": result, "planning": None, "status": None, "execution": None,
            "python_execution": None, "visualizations": [], "execution_warnings": [],
            "execution_status": "not_executed", "pending_operations": [],
            "validation": None, "response": None,
            "stage_durations_ms": {
                **state.stage_durations_ms,
                "retriever": round((perf_counter() - started) * 1000, 3),
            },
        }

    async def plan(state: QueryResponseState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")
        started = perf_counter()
        result = await planner.handle(analytical_state(state).question, state.retrieval,
                                      original_question=state.question,
                                      data_mentions=[item for item in state.routing.data_mentions if not isinstance(item, str)])
        return {
            "planning": result, "status": result.status,
            "pending_operations": [step.id for step in result.operations],
            "stage_durations_ms": {
                **state.stage_durations_ms,
                "planner": round((perf_counter() - started) * 1000, 3),
            },
        }

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
        result = await python_agent.handle(
            state.planning, state.execution, state.python_execution
        )
        status, pending, warnings = summarize_execution(state.planning, state.execution, result)
        return {
            "python_execution": result, "visualizations": result.visualizations,
            "execution_status": status, "pending_operations": pending,
            "execution_warnings": warnings,
        }

    async def validate(state: QueryResponseState):
        result = await validator.handle(analytical_state(state))
        return {"validation": result, "pending_operations": result.pending_operations}

    async def respond(state: QueryResponseState):
        response = await response_generator.handle(analytical_state(state))
        response.routing = state.routing
        return {"response": response}

    builder = StateGraph(QueryResponseState, input_schema=QueryResponseInput)
    builder.add_node("router", route)
    builder.add_node("direct_response", respond_directly)
    builder.add_node("retriever", retrieve)
    builder.add_node("planner", plan)
    builder.add_node("sql_agent", execute_sql)
    builder.add_node("python_agent", execute_python)
    builder.add_node("validator", validate)
    builder.add_node("response_generator", respond)
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router", next_after_routing,
        {"retriever": "retriever", "direct_response": "direct_response"},
    )
    builder.add_edge("direct_response", END)
    builder.add_edge("retriever", "planner")
    builder.add_conditional_edges(
        "planner", next_after_planning,
        {"sql_agent": "sql_agent", "python_agent": "python_agent", "validator": "validator"},
    )
    builder.add_conditional_edges(
        "sql_agent", next_after_sql,
        {"python_agent": "python_agent", "validator": "validator"},
    )
    builder.add_edge("python_agent", "validator")
    builder.add_edge("validator", "response_generator")
    builder.add_edge("response_generator", END)
    return builder.compile()
