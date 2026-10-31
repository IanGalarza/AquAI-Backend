from typing import Protocol

from aquai.application.contracts.execution import (
    PythonExecutionResult,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult


class SqlPlanExecutor(Protocol):
    def execute(self, plan: PlanningResult) -> SqlExecutionResult: ...


class PythonPlanExecutor(Protocol):
    def execute(
        self,
        plan: PlanningResult,
        sql_result: SqlExecutionResult | None,
        previous: PythonExecutionResult | None = None,
    ) -> PythonExecutionResult: ...
