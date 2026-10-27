"""Python Agent: revalida el catálogo y delega herramientas determinísticas."""

import asyncio
from datetime import datetime, timezone
from time import perf_counter

from aquai.application.contracts.execution import ExecutionError, PythonExecutionResult, SqlExecutionResult
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import PlannerValidationError
from aquai.application.execution import validate_sql_transfer
from aquai.application.interfaces.execution import PythonPlanExecutor
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.useCases.queries.executeSqlPlanHandler import validate_execution_plan


class ExecutePythonPlanHandler:
    def __init__(self, executor: PythonPlanExecutor, queries: SourceQueries) -> None:
        self._executor = executor
        self._queries = queries

    async def handle(self, plan: PlanningResult, sql_result: SqlExecutionResult | None,
                     previous: PythonExecutionResult | None = None) -> PythonExecutionResult:
        now = datetime.now(timezone.utc)
        if plan.status != "plannable" or not any(step.tool == "python" for step in plan.operations):
            return PythonExecutionResult(status="not_executed", started_at=now, finished_at=now,
                error=ExecutionError(code="plan_not_executable", message="El plan no autoriza operaciones Python."))
        return await asyncio.to_thread(self._execute, plan, sql_result, previous)

    def _execute(self, plan, sql_result, previous):
        now, started = datetime.now(timezone.utc), perf_counter()
        try:
            validate_execution_plan(plan, plan.selected_datasets)
            validate_sql_transfer(plan, sql_result)
            versions = {item.dataset_id: item.verified_version for item in plan.selected_datasets}
            current = self._queries.verified_datasets(list(versions), versions=versions)
            validate_execution_plan(plan, current)
        except PlannerValidationError:
            return PythonExecutionResult(status="not_executed", started_at=now, finished_at=datetime.now(timezone.utc),
                pending_operations=[step.id for step in plan.operations if step.tool == "python"],
                duration_ms=round((perf_counter() - started) * 1000, 3),
                error=ExecutionError(code="invalid_python_input", message="El plan, las versiones o la transferencia SQL no superaron la revalidación."))
        except Exception:
            return PythonExecutionResult(status="not_executed", started_at=now, finished_at=datetime.now(timezone.utc),
                pending_operations=[step.id for step in plan.operations if step.tool == "python"],
                duration_ms=round((perf_counter() - started) * 1000, 3),
                error=ExecutionError(code="catalog_unavailable", message="No se pudo verificar la disponibilidad del catálogo antes del análisis Python."))
        result = self._executor.execute(plan, sql_result, previous)
        result.started_at = now
        result.duration_ms = round((perf_counter() - started) * 1000, 3)
        return result
