import re
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256

from aquai.application.contracts.execution import (
    PythonExecutionResult,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import AnalysisBlocked, PlannerValidationError

_COMPLETED_STATUSES = {"executed", "executed_as_dependency"}
_CANONICAL_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_MAX_NUMERIC_TEXT_LENGTH = 1100
_MAX_NUMERIC_DIGITS = 1000
_MAX_NUMERIC_EXPONENT = 1000


def plan_fingerprint(plan: PlanningResult) -> str:
    return sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def sql_result_fingerprint(sql: SqlExecutionResult | None) -> str:
    return sha256(
        (sql.model_dump_json() if sql is not None else "null").encode("utf-8")
    ).hexdigest()


def evidence_fingerprint(plan, sql, python) -> str:
    digest = sha256()

    for item in (plan, sql, python):
        digest.update((item.model_dump_json() if item is not None else "null").encode("utf-8"))
        digest.update(b"\x00")

    return digest.hexdigest()


def execution_fingerprint(
    plan: PlanningResult, sql: SqlExecutionResult | None, policy: str = ""
) -> str:
    digest = sha256(plan.model_dump_json().encode("utf-8"))
    if sql is not None:
        digest.update(sql.model_dump_json().encode("utf-8"))
    digest.update(policy.encode("utf-8"))

    return digest.hexdigest()


def validate_sql_transfer(plan: PlanningResult, sql: SqlExecutionResult | None) -> None:
    if sql is None:
        return

    materialized = any(record.materialized for record in sql.operations)
    if materialized and sql.plan_fingerprint != plan_fingerprint(plan):
        raise PlannerValidationError(
            "Los resultados SQL no pertenecen al plan recibido (operaciones, filtros y versiones)."
        )

    _check_transferred_operations(plan, sql)

    if materialized:
        _check_transferred_datasets(plan, sql)


def _check_transferred_operations(plan: PlanningResult, sql: SqlExecutionResult) -> None:
    expected = {step.id: step for step in plan.operations}
    seen = set()

    for record in sql.operations:
        step = expected.get(record.operation_id)
        if step is None or record.operation_id in seen:
            raise PlannerValidationError(
                "La transferencia SQL contiene operaciones ajenas o repetidas."
            )
        seen.add(record.operation_id)

        if record.materialized and _differs_from_sql_step(record, step, sql.read_only):
            raise PlannerValidationError(
                "El resultado materializado no coincide con la operación SQL autorizada."
            )


def _differs_from_sql_step(record, step, read_only: bool) -> bool:
    return (
        step.tool != "sql"
        or record.status != "executed"
        or record.columns != step.output_fields
        or record.rows_returned != len(record.rows)
        or not read_only
    )


def _check_transferred_datasets(plan: PlanningResult, sql: SqlExecutionResult) -> None:
    actual = {item.dataset_id: item for item in sql.datasets}
    expected_ids = {item.dataset_id for item in plan.selected_datasets}
    if len(actual) != len(sql.datasets) or set(actual) != expected_ids:
        raise PlannerValidationError("Faltan las versiones de origen de los resultados SQL.")

    for selected in plan.selected_datasets:
        if _differs_from_catalog(actual[selected.dataset_id], selected):
            raise PlannerValidationError(
                "La procedencia SQL no coincide con el catálogo autorizado."
            )


def _differs_from_catalog(item, selected) -> bool:
    return (
        item.version != selected.verified_version
        or item.source_id != selected.source_id
        or item.provenance != selected.provenance
        or item.columns != selected.columns_metadata
    )


def summarize_execution(
    plan: PlanningResult | None,
    sql: SqlExecutionResult | None,
    python: PythonExecutionResult | None,
):
    if plan is None or plan.status != "plannable":
        return "not_executed", [], []

    records = _operation_records(sql, python)
    completed = {
        key
        for key, value in records.items()
        if value.status in _COMPLETED_STATUSES and not value.truncated
    }
    pending = [step.id for step in plan.operations if step.id not in completed]
    errors = [item.error for item in (sql, python) if item is not None and item.error is not None]
    warnings = _batch_warnings(sql, python)

    if not pending and not errors:
        return "succeeded", [], warnings

    if completed:
        return "partial", pending, warnings

    return ("failed" if errors else "not_executed"), pending, warnings


def _operation_records(sql, python) -> dict:
    records = {item.operation_id: item for item in sql.operations} if sql else {}
    if python:
        records.update({item.operation_id: item for item in python.operations})

    return records


def _batch_warnings(*batches) -> list[str]:
    warnings = []

    for batch in batches:
        if batch is not None:
            warnings.extend(batch.warnings)

    return list(dict.fromkeys(warnings))


def execution_records(state):
    """Python prevalece si SQL y Python informan la misma operación."""
    records = {}

    for batch in (state.execution, state.python_execution):
        if batch is not None:
            records.update({item.operation_id: item for item in batch.operations})

    return records


def decode_rows(record, fields, budget=None):
    """Sólo rehidrata el tipo del esquema verificado, nunca interpreta un TEXT."""
    rows = []
    names = {field.name for field in fields}

    for row in record.rows:
        if budget is not None:
            budget.check()
        if set(row) != names:
            raise AnalysisBlocked(
                "invalid_schema", "Una fila no coincide con el esquema de su resultado autorizado."
            )
        rows.append(_decode_row(row, fields))

    return rows


def _decode_row(row, fields) -> dict:
    decoded = {}

    for field in fields:
        decoded[field.name] = _decode_value(field.sql_type, row[field.name])

    return decoded


def _decode_value(sql_type, value):
    if value is None:
        return None

    decoder = _DECODERS.get(sql_type)
    if decoder is None:
        raise AnalysisBlocked(
            "unsupported_type", "El tipo de entrada no está habilitado para Python."
        )

    return decoder(value)


def _decode_numeric(value):
    if type(value) is str:
        value = _parse_canonical_decimal(value)
    elif type(value) is int:
        value = Decimal(value)

    if not isinstance(value, Decimal) or not value.is_finite():
        raise AnalysisBlocked(
            "invalid_numeric",
            "NUMERIC exige decimales finitos; no se convierte float silenciosamente.",
        )

    parts = value.as_tuple()
    if len(parts.digits) > _MAX_NUMERIC_DIGITS or abs(parts.exponent) > _MAX_NUMERIC_EXPONENT:
        raise AnalysisBlocked(
            "numeric_limit",
            "El número excede la precisión/tamaño de entrada permitidos.",
        )

    return value


def _parse_canonical_decimal(text: str) -> Decimal:
    if len(text) > _MAX_NUMERIC_TEXT_LENGTH or not _CANONICAL_DECIMAL.fullmatch(text):
        raise AnalysisBlocked(
            "invalid_numeric", "La representación NUMERIC no es decimal canónica."
        )

    return Decimal(text)


def _decode_bigint(value):
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise AnalysisBlocked(
            "invalid_integer", "BIGINT exige enteros dentro de su rango original."
        )

    return value


def _decode_date(value):
    value = date.fromisoformat(value) if type(value) is str else value
    if type(value) is not date:
        raise AnalysisBlocked("invalid_date", "La fecha no coincide con DATE.")

    return value


def _decode_timestamp(value):
    value = datetime.fromisoformat(value) if type(value) is str else value
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise AnalysisBlocked(
            "invalid_date",
            "TIMESTAMP requiere su fecha completa sin zona horaria implícita.",
        )

    return value


def _decode_boolean(value):
    if type(value) is not bool:
        raise AnalysisBlocked("invalid_boolean", "La entrada no coincide con BOOLEAN.")

    return value


def _decode_text(value):
    if type(value) is not str:
        raise AnalysisBlocked("invalid_text", "La entrada no coincide con TEXT.")

    return value


_DECODERS = {
    "NUMERIC": _decode_numeric,
    "BIGINT": _decode_bigint,
    "DATE": _decode_date,
    "TIMESTAMP": _decode_timestamp,
    "BOOLEAN": _decode_boolean,
    "TEXT": _decode_text,
}
