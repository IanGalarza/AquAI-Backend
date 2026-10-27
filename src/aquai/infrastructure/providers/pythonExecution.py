"""Composición del Python Agent; no crea clientes LLM ni conexiones nuevas."""

from functools import lru_cache

from aquai.application.useCases.queries.executePythonPlanHandler import ExecutePythonPlanHandler
from aquai.infrastructure.analysis.pythonPlanExecutor import DeterministicPythonPlanExecutor
from aquai.infrastructure.config.pythonExecution import PythonExecutionSettings
from aquai.infrastructure.providers.sqlExecution import get_sql_execution_context
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


@lru_cache
def get_python_execution_settings() -> PythonExecutionSettings:
    return PythonExecutionSettings()


def create_python_agent() -> ExecutePythonPlanHandler:
    return ExecutePythonPlanHandler(DeterministicPythonPlanExecutor(get_python_execution_settings()),
        SqlAlchemySourceQueries(get_sql_execution_context()))
