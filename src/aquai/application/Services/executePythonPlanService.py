import asyncio
from datetime import datetime, timezone
from time import perf_counter

from aquai.application.contracts.execution import (
    ExecutionError,
    PythonExecutionResult,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import PlannerValidationError
from aquai.application.execution import validate_sql_transfer
from aquai.application.interfaces.execution import PythonPlanExecutor
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.planning.validation import validate_execution_plan


class ExecutePythonPlanService:
    #
    # Guarda el ejecutor de Python y las consultas de fuentes que usa el servicio.
    #
    def __init__(self, executor: PythonPlanExecutor, queries: SourceQueries) -> None:
        self._executor = executor
        self._queries = queries

    #
    # Ejecuta las operaciones Python autorizadas del plan, o informa que no son ejecutables.
    #
    async def execute(
        self,
        plan: PlanningResult,
        sql_result: SqlExecutionResult | None,
        previous: PythonExecutionResult | None = None,
    ) -> PythonExecutionResult:
        now = datetime.now(timezone.utc)

        if plan.status != "plannable" or not any(step.tool == "python" for step in plan.operations):
            return PythonExecutionResult(
                status="not_executed",
                started_at=now,
                finished_at=now,
                error=ExecutionError(
                    code="plan_not_executable", message="El plan no autoriza operaciones Python."
                ),
            )

        return await asyncio.to_thread(self._execute, plan, sql_result, previous)

    #
    # Revalida el plan y ejecuta el análisis Python registrando su inicio y su duración.
    #
    def _execute(self, plan, sql_result, previous):
        now = datetime.now(timezone.utc)
        started = perf_counter()

        try:
            self._revalidate(plan, sql_result)
        except PlannerValidationError:
            return _not_executed(
                plan,
                now,
                started,
                code="invalid_python_input",
                message="El plan, las versiones o la transferencia SQL no superaron la revalidación.",
            )
        except Exception:
            return _not_executed(
                plan,
                now,
                started,
                code="catalog_unavailable",
                message="No se pudo verificar la disponibilidad del catálogo antes del análisis Python.",
            )

        result = self._executor.execute(plan, sql_result, previous)
        result.started_at = now
        result.duration_ms = round((perf_counter() - started) * 1000, 3)

        return result

    #
    # Revalida el plan, la transferencia SQL y las versiones de los datasets contra el catálogo vigente.
    #
    def _revalidate(self, plan, sql_result) -> None:
        validate_execution_plan(plan, plan.selected_datasets)
        validate_sql_transfer(plan, sql_result)

        versions = {item.dataset_id: item.verified_version for item in plan.selected_datasets}
        current = self._queries.verified_datasets(list(versions), versions=versions)
        validate_execution_plan(plan, current)


#
# Construye el resultado Python no ejecutado con las operaciones pendientes y el error indicado.
#
def _not_executed(plan, now, started, *, code, message) -> PythonExecutionResult:
    return PythonExecutionResult(
        status="not_executed",
        started_at=now,
        finished_at=datetime.now(timezone.utc),
        pending_operations=[step.id for step in plan.operations if step.tool == "python"],
        duration_ms=round((perf_counter() - started) * 1000, 3),
        error=ExecutionError(code=code, message=message),
    )
