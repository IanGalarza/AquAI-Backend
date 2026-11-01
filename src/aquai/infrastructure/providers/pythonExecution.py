from functools import lru_cache

from aquai.application.Services.executePythonPlanService import ExecutePythonPlanService
from aquai.infrastructure.analysis.pythonPlanExecutor import (
    DeterministicPythonPlanExecutor,
)
from aquai.infrastructure.config.pythonExecution import PythonExecutionSettings
from aquai.infrastructure.providers.sqlExecution import get_sql_execution_context
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


#
# Devuelve la configuración de ejecución de planes en Python, cacheada.
#
@lru_cache
def get_python_execution_settings() -> PythonExecutionSettings:
    return PythonExecutionSettings()


#
# Crea el servicio que ejecuta planes con el ejecutor determinista en Python.
#
def create_python_agent() -> ExecutePythonPlanService:
    return ExecutePythonPlanService(
        DeterministicPythonPlanExecutor(get_python_execution_settings()),
        SqlAlchemySourceQueries(get_sql_execution_context()),
    )
