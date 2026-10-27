"""Verificación de transferencias y resumen del plan, sin frameworks ni motores."""

from hashlib import sha256

from aquai.application.contracts.execution import PythonExecutionResult, SqlExecutionResult
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import PlannerValidationError


def plan_fingerprint(plan: PlanningResult) -> str:
    return sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def execution_fingerprint(plan: PlanningResult, sql: SqlExecutionResult | None, policy: str = "") -> str:
    digest = sha256(plan.model_dump_json().encode("utf-8"))
    if sql is not None:
        digest.update(sql.model_dump_json().encode("utf-8"))
    digest.update(policy.encode("utf-8"))
    return digest.hexdigest()


def validate_sql_transfer(plan: PlanningResult, sql: SqlExecutionResult | None) -> None:
    if sql is None:
        return
    if any(record.materialized for record in sql.operations) and sql.plan_fingerprint != plan_fingerprint(plan):
        raise PlannerValidationError("Los resultados SQL no pertenecen al plan recibido (operaciones, filtros y versiones).")
    expected = {step.id: step for step in plan.operations}
    seen = set()
    for record in sql.operations:
        step = expected.get(record.operation_id)
        if step is None or record.operation_id in seen:
            raise PlannerValidationError("La transferencia SQL contiene operaciones ajenas o repetidas.")
        seen.add(record.operation_id)
        if record.materialized:
            if (step.tool != "sql" or record.status != "executed" or record.columns != step.output_fields
                    or record.rows_returned != len(record.rows) or not sql.read_only):
                raise PlannerValidationError("El resultado materializado no coincide con la operación SQL autorizada.")
    if any(record.materialized for record in sql.operations):
        actual = {item.dataset_id: item for item in sql.datasets}
        if len(actual) != len(sql.datasets) or set(actual) != {item.dataset_id for item in plan.selected_datasets}:
            raise PlannerValidationError("Faltan las versiones de origen de los resultados SQL.")
        for selected in plan.selected_datasets:
            item = actual[selected.dataset_id]
            if (item.version != selected.verified_version or item.source_id != selected.source_id
                    or item.provenance != selected.provenance or item.columns != selected.columns_metadata):
                raise PlannerValidationError("La procedencia SQL no coincide con el catálogo autorizado.")


def summarize_execution(plan: PlanningResult | None, sql: SqlExecutionResult | None,
                        python: PythonExecutionResult | None):
    if plan is None or plan.status != "plannable":
        return "not_executed", [], []
    records = {item.operation_id: item for item in sql.operations} if sql else {}
    if python:
        records.update({item.operation_id: item for item in python.operations})
    completed = {key for key, value in records.items()
                 if value.status in {"executed", "executed_as_dependency"} and not value.truncated}
    pending = [step.id for step in plan.operations if step.id not in completed]
    errors = [item.error for item in (sql, python) if item is not None and item.error is not None]
    warnings = list(dict.fromkeys(warning for item in (sql, python) if item is not None for warning in item.warnings))
    if not pending and not errors:
        return "succeeded", [], warnings
    if completed:
        return "partial", pending, warnings
    return ("failed" if errors else "not_executed"), pending, warnings
