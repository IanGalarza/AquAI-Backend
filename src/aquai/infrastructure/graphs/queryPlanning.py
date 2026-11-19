import asyncio
import logging
from collections.abc import Callable
from time import perf_counter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aquai.application.contracts.answering import QueryResponseInput, QueryResponseState
from aquai.application.contracts.retrieval import DocumentRetrievalResult
from aquai.application.conversations.context import cited_analysis
from aquai.application.execution import summarize_execution
from aquai.application.Services.contrastReferencesService import ContrastReferencesService
from aquai.application.Services.executePythonPlanService import ExecutePythonPlanService
from aquai.application.Services.executeSqlPlanService import ExecuteSqlPlanService
from aquai.application.Services.generateDirectResponseService import (
    GenerateDirectResponseService,
)
from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.Services.planQueryService import PlanQueryService
from aquai.application.Services.retrieveDatasetsService import RetrieveDatasetsService
from aquai.application.Services.retrieveDocumentsService import RetrieveDocumentsService
from aquai.application.Services.routeQueryService import RouteQueryService
from aquai.application.Services.validateEvidenceService import ValidateEvidenceService

_logger = logging.getLogger(__name__)

#
# Etapa que ve el usuario mientras se ejecuta cada nodo del grafo.
#
NODE_STAGES = {
    "router": "interpreting",
    "retriever": "retrieving",
    "reference_resolver": "referencing",
    "planner": "planning",
    "sql_agent": "analyzing",
    "python_agent": "analyzing",
    "validator": "validating",
    "document_retriever": "documents",
    "response_generator": "writing",
    "document_response": "writing",
    "direct_response": "writing",
}


#
# Devuelve el estado con la pregunta contextualizada para el análisis.
#
def analytical_state(state: QueryResponseState) -> QueryResponseState:
    question = state.routing.contextualized_question if state.routing else state.question

    return state.model_copy(update={"question": question})


#
# Elige el siguiente nodo tras el enrutado: datos, documentos o respuesta directa.
#
def next_after_routing(state: QueryResponseState):
    if state.routing.route == "data":
        return "retriever"

    return "document_retriever" if state.routing.route == "document" else "direct_response"


#
# Elige la respuesta tras buscar documentos: la analítica o la puramente documental.
#
def next_after_documents(state: QueryResponseState):
    return "response_generator" if state.routing.route == "data" else "document_response"


#
# Elige el siguiente nodo tras recuperar datasets: buscar antes la referencia documental pedida, o planificar.
#
def next_after_retrieval(state: QueryResponseState):
    return "reference_resolver" if state.routing.reference_comparison else "planner"


#
# Elige el siguiente nodo tras buscar la referencia: planificar con ella o presentar lo que dicen los documentos.
#
def next_after_reference(state: QueryResponseState):
    return "document_response" if state.routing.route == "document" else "planner"


#
# Elige el siguiente nodo tras planificar: SQL, Python o validación.
#
def next_after_planning(state: QueryResponseState):
    plan = state.planning
    if plan is None or plan.status != "plannable":
        return "validator"

    if any(step.tool == "sql" for step in plan.operations):
        return "sql_agent"

    if any(step.tool == "python" for step in plan.operations):
        return "python_agent"

    return "validator"


#
# Elige el siguiente nodo tras SQL: Python si hay pasos pendientes, si no la validación.
#
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


#
# Construye el grafo de consulta con enrutado, recuperación, planificación, ejecución y respuesta.
#
def build_query_planning_graph(
    retriever_factory: Callable[[], RetrieveDatasetsService],
    planner: PlanQueryService,
    sql_agent: ExecuteSqlPlanService,
    python_agent: ExecutePythonPlanService,
    validator: ValidateEvidenceService,
    response_generator: GenerateResponseService,
    router: RouteQueryService,
    direct_response: GenerateDirectResponseService,
    document_services_factory: Callable[
        [], tuple[RetrieveDocumentsService, ContrastReferencesService]
    ],
) -> CompiledStateGraph:
    retriever = None
    document_services = None
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
            reselect=state.routing.reselects_datasets,
        )

        return {
            "retrieval": result,
            **_cleared_analysis(),
            "stage_durations_ms": _stage_durations(state, "retriever", started),
        }

    async def retrieve_documents(state: QueryResponseState):
        nonlocal document_services
        started = perf_counter()
        analytical = state.routing.route == "data"
        try:
            async with retrieval_lock:
                if document_services is None:
                    document_services = await asyncio.to_thread(document_services_factory)

            documents, references = document_services
            result = _resolved_evidence(state) if analytical else None
            result = result or await documents.retrieve(state)
            current = state.model_copy(update={"document_retrieval": result})
            contrast = await references.contrast(current) if analytical else None
        except Exception as error:
            _logger.warning("Falló la evidencia documental: %s", type(error).__name__)
            purpose = "enrichment" if analytical else "question"
            result = DocumentRetrievalResult(purpose=purpose, status="unavailable")
            contrast = None

        return {
            "document_retrieval": result,
            "document_contrast": contrast,
            "stage_durations_ms": _stage_durations(state, "documents", started),
        }

    async def resolve_reference(state: QueryResponseState):
        nonlocal document_services
        started = perf_counter()
        try:
            async with retrieval_lock:
                if document_services is None:
                    document_services = await asyncio.to_thread(document_services_factory)

            documents, references = document_services
            retrieval = await documents.retrieve_reference(state)
            found = await references.documented_thresholds(state, retrieval)
            routing = references.resolved_routing(state, retrieval, found)
        except Exception as error:
            _logger.warning("Falló la búsqueda de la referencia documental: %s", type(error).__name__)
            return {"stage_durations_ms": _stage_durations(state, "reference", started)}

        documental = {"retrieval": None} if routing.route == "document" else {}

        return {
            **documental,
            "routing": routing,
            "document_retrieval": retrieval,
            "stage_durations_ms": _stage_durations(state, "reference", started),
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
            requested_period=state.routing.requested_period,
            reference_request=None if state.routing.reference_options else state.routing.reference_comparison,
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

    async def respond_from_documents(state: QueryResponseState):
        response = await response_generator.generate_documental(state)
        _logger.info(
            "Turno route=document status=%s etapas_ms=%s", response.status, response.timings_ms
        )

        return {"response": response}

    builder = StateGraph(QueryResponseState, input_schema=QueryResponseInput)
    builder.add_node("router", route)
    builder.add_node("direct_response", respond_directly)
    builder.add_node("retriever", retrieve)
    builder.add_node("reference_resolver", resolve_reference)
    builder.add_node("planner", plan)
    builder.add_node("sql_agent", execute_sql)
    builder.add_node("python_agent", execute_python)
    builder.add_node("validator", validate)
    builder.add_node("document_retriever", retrieve_documents)
    builder.add_node("response_generator", respond)
    builder.add_node("document_response", respond_from_documents)
    _add_edges(builder)

    return builder.compile()


#
# Agrega al grafo las conexiones entre nodos.
#
def _add_edges(builder: StateGraph) -> None:
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        next_after_routing,
        {
            "retriever": "retriever",
            "document_retriever": "document_retriever",
            "direct_response": "direct_response",
        },
    )
    builder.add_edge("direct_response", END)
    builder.add_conditional_edges(
        "retriever",
        next_after_retrieval,
        {"reference_resolver": "reference_resolver", "planner": "planner"},
    )
    builder.add_conditional_edges(
        "reference_resolver",
        next_after_reference,
        {"document_response": "document_response", "planner": "planner"},
    )
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
    builder.add_edge("validator", "document_retriever")
    builder.add_conditional_edges(
        "document_retriever",
        next_after_documents,
        {"response_generator": "response_generator", "document_response": "document_response"},
    )
    builder.add_edge("response_generator", END)
    builder.add_edge("document_response", END)


#
# Devuelve los fragmentos de la referencia que se resolvió antes de planificar, para citarlos al responder.
#
def _resolved_evidence(state):
    retrieval = state.document_retrieval
    if not state.routing.reference_comparison or retrieval is None or retrieval.status != "found":
        return None

    return retrieval.model_copy(update={"purpose": "enrichment"})


#
# Indica si el SQL Agent rechazó el plan por inválido.
#
def _sql_rejected_plan(execution) -> bool:
    return execution and execution.error and execution.error.code == "invalid_plan"


#
# Devuelve las menciones de datos estructuradas, sin las de texto simple.
#
def _structured_mentions(routing):
    return [item for item in routing.data_mentions if not isinstance(item, str)]


#
# Devuelve el estado de análisis vacío para reiniciar planificación, ejecución y respuesta.
#
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


#
# Devuelve las duraciones por etapa añadiendo la de la etapa indicada.
#
def _stage_durations(state: QueryResponseState, stage: str, started: float) -> dict:
    return {
        **state.stage_durations_ms,
        stage: round((perf_counter() - started) * 1000, 3),
    }
