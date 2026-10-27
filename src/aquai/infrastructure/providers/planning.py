"""Composición del flujo completo de consulta, sin endpoints conversacionales."""

from langgraph.graph.state import CompiledStateGraph

from aquai.application.useCases.queries.planQueryHandler import PlanQueryHandler
from aquai.infrastructure.graphs.queryPlanning import build_query_planning_graph
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.llm.planner import OllamaPlannerProvider
from aquai.infrastructure.providers.retrieval import create_dataset_retriever
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.providers.sqlExecution import create_sql_agent
from aquai.infrastructure.providers.pythonExecution import create_python_agent
from aquai.infrastructure.providers.answering import create_evidence_validator, create_response_generator
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


def create_query_planning_graph() -> CompiledStateGraph:
    retriever = create_dataset_retriever()
    planner = PlanQueryHandler(OllamaPlannerProvider(get_tabular_settings()), SqlAlchemySourceQueries(get_db_context()))
    return build_query_planning_graph(retriever, planner, create_sql_agent(), create_python_agent(),
                                      create_evidence_validator(), create_response_generator())
