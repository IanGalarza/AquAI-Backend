from langgraph.graph.state import CompiledStateGraph

from aquai.application.Services.generateDirectResponseService import (
    GenerateDirectResponseService,
)
from aquai.application.Services.planQueryService import PlanQueryService
from aquai.application.Services.routeQueryService import RouteQueryService
from aquai.infrastructure.graphs.queryPlanning import build_query_planning_graph
from aquai.infrastructure.providers.answering import (
    create_evidence_validator,
    create_response_generator,
)
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.llm.planner import OllamaPlannerProvider
from aquai.infrastructure.providers.llm.routing import (
    OllamaGeneralResponseProvider,
    OllamaRoutingProvider,
)
from aquai.infrastructure.providers.pythonExecution import create_python_agent
from aquai.infrastructure.providers.retrieval import create_dataset_retriever
from aquai.infrastructure.providers.sqlExecution import create_sql_agent
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


def create_query_planning_graph() -> CompiledStateGraph:
    settings = get_tabular_settings()
    sources = SqlAlchemySourceQueries(get_db_context())
    planner = PlanQueryService(OllamaPlannerProvider(settings), sources)
    router = RouteQueryService(OllamaRoutingProvider(settings), sources)
    direct = GenerateDirectResponseService(OllamaGeneralResponseProvider(settings), sources)

    return build_query_planning_graph(
        create_dataset_retriever,
        planner,
        create_sql_agent(),
        create_python_agent(),
        create_evidence_validator(),
        create_response_generator(),
        router,
        direct,
    )
