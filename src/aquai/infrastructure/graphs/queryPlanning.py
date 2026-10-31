import asyncio
import logging
from collections.abc import Callable
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.conversations.context import cited_analysis
from aquai.application.execution import summarize_execution
from aquai.application.Services.executePythonPlanService import ExecutePythonPlanService
from aquai.application.Services.executeSqlPlanService import ExecuteSqlPlanService
from aquai.application.Services.generateDirectResponseService import (
    GenerateDirectResponseService,
)
from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.Services.planQueryService import PlanQueryService
from aquai.application.Services.retrieveDatasetsService import RetrieveDatasetsService
from aquai.application.Services.routeQueryService import RouteQueryService
from aquai.application.Services.validateEvidenceService import ValidateEvidenceService

_logger = logging.getLogger(__name__)


def analytical_state(state: QueryResponseState) -> QueryResponseState:
    """La pregunta original permanece intacta en el estado y en Message.content."""
    question = state.routing.contextualized_question if state.routing else state.question

    return state.model_copy(update={"question": question})


def next_after_routing(state: QueryResponseState):
    return "retriever" if state.routing.route == "data" else "direct_response"


def next_after_planning(state: QueryResponseState):
    plan = state.planning
    if plan is None or plan.status != "plannable":
        return "validator"

    if any(step.tool == "sql" for step in plan.operations):
        return "sql_agent"

    if any(step.tool == "python" for step in plan.operations):
        return "python_agent"

    return "validator"


def next_after_sql(state: QueryResponseState):
    plan = state.planning
    python_pending = (
        plan is not None
        and plan.status == "plannable"
        and any(step.tool == "python" for step in plan.operations)
    )

    if python_pending and not _sql_rejected_plan(state.execution):
        return "python_agent"

    return "validator"


def build_query_planning_graph(
    retriever_factory: Callable[[], RetrieveDatasetsService],
    planner: PlanQueryService,
    sql_agent: ExecuteSqlPlanService,
    python_agent: ExecutePythonPlanService,
    validator: ValidateEvidenceService,
    response_generator: GenerateResponseService,
    router: RouteQueryService,
    direct_response: GenerateDirectResponseService,
) -> CompiledStateGraph:
    retriever = None
    retrieval_lock = asyncio.Lock()

    async def route(state: QueryResponseState):
        result = await router.route(state)

        return {
            "routing": result,
            "response_language": result.language,
            "stage_durations_ms": {"router": result.duration_ms},
        }

    async def respond_directly(state: QueryResponseState):
        response = await direct_response.generate(state)
        _logger.info(
            "Turno route=%s status=%s etapas_ms=%s",
            state.routing.route,
            response.status,
            response.timings_ms,
        )

        return {"response": response}

    async def retrieve(state: QueryResponseState):
        nonlocal retriever
        started = perf_counter()

        async with retrieval_lock:
            if retriever is None:
                retriever = await asyncio.to_thread(retriever_factory)

        query = analytical_state(state)
        result = await retriever.retrieve(
            query.question,
            limit=state.candidate_limit,
            dataset_bindings=state.routing.dataset_bindings,
            data_mentions=_structured_mentions(state.routing),
            tentative_values=state.routing.tentative_values,
        )

        return {
            "retrieval": result,
            **_cleared_analysis(),
            "stage_durations_ms": _stage_durations(state, "retriever", started),
        }

    async def plan(state: QueryResponseState):
        if state.retrieval is None:
            raise ValueError("El nodo Planner requiere el resultado de recuperación.")

        started = perf_counter()
        result = await planner.plan(
            analytical_state(state).question,
            state.retrieval,
            original_question=state.question,
            data_mentions=_structured_mentions(state.routing),
            request_history=state.routing.request_history,
            source_mentions=state.routing.source_mentions,
            previous_analysis=cited_analysis(
                state.conversation_history, state.routing.history_message_ids
            ),
            conserved_mentions=state.routing.conserved_mentions,
        )

        return {
            "planning": result,
            "status": result.status,
            "pending_operations": [step.id for step in result.operations],
            "stage_durations_ms": _stage_durations(state, "planner", started),
        }

    async def execute_sql(state: QueryResponseState):
        if state.planning is None:
            raise ValueError("El SQL Agent requiere un plan verificado.")

        result = await sql_agent.execute(state.planning)
        status, pending, warnings = summarize_execution(state.planning, result, None)

        return {
            "execution": result,
            "execution_status": status,
            "pending_operations": pending,
            "execution_warnings": warnings,
        }

    async def execute_python(state: QueryResponseState):
        if state.planning is None:
            raise ValueError("El Python Agent requiere un plan verificado.")

        result = await python_agent.execute(state.planning, state.execution, state.python_execution)
        status, pending, warnings = summarize_execution(state.planning, state.execution, result)

        return {
            "python_execution": result,
            "visualizations": result.visualizations,
            "execution_status": status,
            "pending_operations": pending,
            "execution_warnings": warnings,
        }

    async def validate(state: QueryResponseState):
        result = await validator.validate(analytical_state(state))

        return {"validation": result, "pending_operations": result.pending_operations}

    async def respond(state: QueryResponseState):
        response = await response_generator.generate(analytical_state(state))
        response.routing = state.routing
        _logger.info(
            "Turno route=data status=%s etapas_ms=%s", response.status, response.timings_ms
        )

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
    _add_edges(builder)

    return builder.compile()


def _add_edges(builder: StateGraph) -> None:
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        next_after_routing,
        {"retriever": "retriever", "direct_response": "direct_response"},
    )
    builder.add_edge("direct_response", END)
    builder.add_edge("retriever", "planner")
    builder.add_conditional_edges(
        "planner",
        next_after_planning,
        {"sql_agent": "sql_agent", "python_agent": "python_agent", "validator": "validator"},
    )
    builder.add_conditional_edges(
        "sql_agent",
        next_after_sql,
        {"python_agent": "python_agent", "validator": "validator"},
    )
    builder.add_edge("python_agent", "validator")
    builder.add_edge("validator", "response_generator")
    builder.add_edge("response_generator", END)


def _sql_rejected_plan(execution) -> bool:
    return execution and execution.error and execution.error.code == "invalid_plan"


def _structured_mentions(routing):
    return [item for item in routing.data_mentions if not isinstance(item, str)]


def _cleared_analysis() -> dict:
    return {
        "planning": None,
        "status": None,
        "execution": None,
        "python_execution": None,
        "visualizations": [],
        "execution_warnings": [],
        "execution_status": "not_executed",
        "pending_operations": [],
        "validation": None,
        "response": None,
    }


def _stage_durations(state: QueryResponseState, stage: str, started: float) -> dict:
    return {
        **state.stage_durations_ms,
        stage: round((perf_counter() - started) * 1000, 3),
    }
