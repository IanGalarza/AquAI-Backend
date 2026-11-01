from typing import Protocol

from aquai.application.contracts.execution import (
    PythonExecutionResult,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult


class SqlPlanExecutor(Protocol):
    #
    # Ejecuta las operaciones SQL de un plan y devuelve sus resultados.
    #
    def execute(self, plan: PlanningResult) -> SqlExecutionResult: ...


class PythonPlanExecutor(Protocol):
    #
    # Ejecuta las operaciones Python de un plan a partir de los resultados SQL y previos.
    #
    def execute(
        self,
        plan: PlanningResult,
        sql_result: SqlExecutionResult | None,
        previous: PythonExecutionResult | None = None,
    ) -> PythonExecutionResult: ...
