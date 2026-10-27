"""Puerto de ejecución de planes, independiente de SQLAlchemy y PostgreSQL."""

from typing import Protocol

from aquai.application.contracts.execution import SqlExecutionResult
from aquai.application.contracts.planning import PlanningResult


class SqlPlanExecutor(Protocol):
    def execute(self, plan: PlanningResult) -> SqlExecutionResult: ...
