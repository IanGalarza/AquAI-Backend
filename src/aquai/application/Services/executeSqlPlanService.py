import asyncio
from datetime import datetime, timezone

from aquai.application.contracts.execution import (
    ExecutionError,
    OperationExecution,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.interfaces.execution import SqlPlanExecutor


class ExecuteSqlPlanService:
    def __init__(self, executor: SqlPlanExecutor) -> None:
        self._executor = executor

    async def execute(self, plan: PlanningResult) -> SqlExecutionResult:
        if plan.status != "plannable" or not any(step.tool == "sql" for step in plan.operations):
            return _not_executed(plan)

        return await asyncio.to_thread(self._executor.execute, plan)


def _not_executed(plan: PlanningResult) -> SqlExecutionResult:
    now = datetime.now(timezone.utc)
    pending = _pending_python_operations(plan)

    return SqlExecutionResult(
        status="not_executed",
        started_at=now,
        finished_at=now,
        pending_operations=pending,
        operations=[
            OperationExecution(operation_id=step_id, status="pending_python") for step_id in pending
        ],
        error=ExecutionError(
            code="plan_not_executable",
            message="El plan no requiere o no autoriza ejecución SQL.",
        ),
    )


def _pending_python_operations(plan: PlanningResult) -> list[str]:
    if plan.status != "plannable":
        return []

    return [step.id for step in plan.operations if step.tool == "python"]
