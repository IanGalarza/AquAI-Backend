"""SQL Agent determinístico: autoriza el contrato y delega ejecución al puerto."""

import asyncio
from datetime import datetime, timezone

from aquai.application.contracts.execution import ExecutionError, OperationExecution, SqlExecutionResult
from aquai.application.contracts.planning import DatasetSelection, PlannableDecision, PlanningResult
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.errors import PlannerValidationError
from aquai.application.interfaces.execution import SqlPlanExecutor
from aquai.application.planning.validation import validate_operations, validate_selections, validate_temporal_scope


def validate_execution_plan(plan: PlanningResult, catalog: list[VerifiedDataset]):
    """Reutiliza las verificaciones del Planner con un catálogo fresco y confiable."""
    if plan.status != "plannable" or any(issue.severity == "clarification" for issue in plan.validation_issues):
        raise PlannerValidationError("El plan no está autorizado para ejecución.")
    decision = PlannableDecision(
        status="plannable", reason="Revalidación previa a ejecución.",
        datasets=[DatasetSelection(dataset_id=item.dataset_id, version=item.verified_version,
                  columns=[column.sql_name for column in item.columns_metadata], reason=item.selection_reason)
                  for item in plan.selected_datasets],
        steps=[step.model_dump(exclude={"output_fields"}) for step in plan.operations],
        output_steps=plan.output_steps,
    )
    snapshots = [VerifiedDataset.model_validate(item.model_dump()) for item in catalog]
    selected = validate_selections(decision, snapshots)
    for proposed, actual in zip(plan.selected_datasets, selected, strict=True):
        if any(getattr(proposed, field) != getattr(actual, field) for field in (
            "source_id", "semantic_point_id", "physical_table_name", "row_count", "columns_metadata", "provenance", "warnings",
        )):
            raise PlannerValidationError("Los metadatos autorizados cambiaron; hay que volver a planificar.")
    operations, issues = validate_operations(decision, selected, plan.temporal_text_columns)
    issues.extend(validate_temporal_scope(operations, plan.output_steps,
        require_latest=plan.context.evidence.get("temporal_requirements", {}).get("operation") == "latest",
        requested_values=[item["text"] for item in plan.context.evidence.get("query_mentions", []) if item["kind"] == "value"]))
    if any(issue.severity == "clarification" for issue in issues):
        raise PlannerValidationError("La operación requiere resolver una aclaración antes de ejecutarse.")
    if any(old.output_fields != new.output_fields for old, new in zip(plan.operations, operations, strict=True)):
        raise PlannerValidationError("El esquema del plan no coincide con el derivado del catálogo.")
    return selected, operations


class ExecuteSqlPlanHandler:
    def __init__(self, executor: SqlPlanExecutor) -> None:
        self._executor = executor

    async def handle(self, plan: PlanningResult) -> SqlExecutionResult:
        if plan.status != "plannable" or not any(step.tool == "sql" for step in plan.operations):
            now = datetime.now(timezone.utc)
            pending = [step.id for step in plan.operations if step.tool == "python"] if plan.status == "plannable" else []
            return SqlExecutionResult(
                status="not_executed", started_at=now, finished_at=now, pending_operations=pending,
                operations=[OperationExecution(operation_id=step_id, status="pending_python") for step_id in pending],
                error=ExecutionError(code="plan_not_executable", message="El plan no requiere o no autoriza ejecución SQL."),
            )
        return await asyncio.to_thread(self._executor.execute, plan)
