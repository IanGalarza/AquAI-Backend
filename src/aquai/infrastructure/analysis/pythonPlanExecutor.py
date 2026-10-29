"""Ejecutor de herramientas Python autorizadas, acotado y sin acceso a datasets."""

from datetime import datetime, timezone
from decimal import DecimalException
from time import perf_counter

from pydantic import ValidationError

from aquai.application.contracts.execution import (
    ExecutionDataset, ExecutionError, OperationExecution, PythonExecutionResult, SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.execution import (
    execution_fingerprint, plan_fingerprint, sql_result_fingerprint, summarize_execution, validate_sql_transfer,
)
from aquai.application.errors import PlannerValidationError
from aquai.application.useCases.queries.executeSqlPlanHandler import validate_execution_plan
from aquai.infrastructure.analysis.operations import AnalysisBlocked, decode_rows, tabular_operation
from aquai.infrastructure.analysis.visualizations import create_visualization
from aquai.infrastructure.analysis.plotlyFigures import build_plotly_figure
from aquai.infrastructure.config.pythonExecution import PythonExecutionSettings


class _Budget:
    def __init__(self, settings, started):
        self.settings, self.started = settings, started

    def check(self):
        if (perf_counter() - self.started) * 1000 > self.settings.max_duration_ms:
            raise AnalysisBlocked("python_timeout", "Se superó el presupuesto de tiempo de las herramientas Python.")

    def input(self, record):
        self.check()
        settings = self.settings
        if len(record.rows) > settings.max_input_rows or len(record.rows) * len(record.columns) > settings.max_cells:
            raise AnalysisBlocked("input_limit", "La entrada completa supera los límites de filas/celdas de Python; reducila explícitamente mediante SQL.")
        size = len(record.model_dump_json().encode("utf-8"))
        if size > settings.max_input_bytes or size * 8 > settings.max_memory_bytes:
            raise AnalysisBlocked("input_limit", "La entrada completa supera el presupuesto de bytes/memoria; no se analiza un prefijo.")


class DeterministicPythonPlanExecutor:
    def __init__(self, settings: PythonExecutionSettings) -> None:
        self._settings = settings

    def execute(self, plan: PlanningResult, sql_result: SqlExecutionResult | None,
                previous: PythonExecutionResult | None = None) -> PythonExecutionResult:
        started, now = perf_counter(), datetime.now(timezone.utc)
        settings = self._settings
        budget = _Budget(settings, started)
        result = PythonExecutionResult(status="not_executed", started_at=now, finished_at=now,
            plan_fingerprint=plan_fingerprint(plan), sql_input_fingerprint=sql_result_fingerprint(sql_result),
            output_operations=list(plan.output_steps), decimal_precision=settings.decimal_precision,
            warnings=list(dict.fromkeys([*plan.warnings,
                f"Cálculos NUMERIC con Decimal y precisión de {settings.decimal_precision} dígitos significativos; no se convierten silenciosamente a float."])))
        try:
            validate_execution_plan(plan, plan.selected_datasets)
            validate_sql_transfer(plan, sql_result)
            result.input_fingerprint = execution_fingerprint(plan, sql_result, settings.model_dump_json())
        except (PlannerValidationError, ValidationError):
            result.error = ExecutionError(code="invalid_python_input", message="El plan o la transferencia no están autorizados para ejecución Python.")
            return self._finish(result, plan, sql_result, started)
        result.datasets = [ExecutionDataset(dataset_id=item.dataset_id, source_id=item.source_id,
            version=item.verified_version, provenance=item.provenance, columns=item.columns_metadata,
            warnings=item.warnings) for item in plan.selected_datasets]
        try:
            self._output_budget(result)
        except AnalysisBlocked as error:
            result.datasets = []
            result.warnings = ["Los metadatos completos permanecen en el plan y superan el presupuesto del resultado Python."]
            result.error = ExecutionError(code=error.code, message=str(error))
            return self._finish(result, plan, sql_result, started)
        records = {item.operation_id: item for item in sql_result.operations} if sql_result else {}
        steps = {step.id: step for step in plan.operations}
        cached = {}
        charts = {}
        if previous is not None and previous.input_fingerprint == result.input_fingerprint:
            cached = {item.operation_id: item for item in previous.operations if item.status == "executed" and not item.truncated}
            charts = {chart.operation_id: chart for chart in previous.visualizations}
        for step in plan.operations:
            if step.tool != "python":
                continue
            record = OperationExecution(operation_id=step.id, status="not_executed", columns=step.output_fields,
                input_operation_ids=list(step.depends_on),
                warnings=[issue.message for issue in plan.validation_issues if issue.step_id == step.id])
            record.dataset_ids = sorted({ref.dataset_id for field in step.output_fields for ref in field.sources} |
                {dataset_id for parent in step.depends_on if parent in records for dataset_id in records[parent].dataset_ids}, key=str)
            result.operations.append(record)
            records[step.id] = record
            operation_started = perf_counter()
            rows = None
            try:
                budget.check()
                if step.operation not in {"aggregate", "calculate", "sort", "latest", "visualize"}:
                    raise AnalysisBlocked("unsupported_operation", "La operación todavía no está habilitada en Python.")
                parents = [records.get(parent) for parent in step.depends_on]
                if len(parents) != 1 or parents[0] is None:
                    raise AnalysisBlocked("missing_dependency", "Falta el resultado autorizado de la dependencia; Python no vuelve a consultar PostgreSQL.")
                parent = parents[0]
                if parent.truncated:
                    raise AnalysisBlocked("incomplete_input", "La entrada está truncada; no se calculan estadísticas ni gráficos de una muestra accidental.")
                if parent.status != "executed" or not parent.materialized:
                    raise AnalysisBlocked("missing_dependency", "La dependencia no está completa/materializada; permanece pendiente.")
                if parent.columns != steps[step.depends_on[0]].output_fields or parent.rows_returned != len(parent.rows):
                    raise AnalysisBlocked("invalid_schema", "La dependencia no coincide con el esquema o la cantidad de filas autorizados.")
                budget.input(parent)
                if step.id in cached and cached[step.id].columns == step.output_fields:
                    reused = cached[step.id]
                    if not reused.materialized or reused.rows_returned != len(reused.rows):
                        raise AnalysisBlocked("invalid_cached_result", "El resultado anterior no está completo.")
                    budget.input(reused)
                    decode_rows(reused, step.output_fields, budget)
                    if step.operation == "visualize" and step.id not in charts:
                        raise AnalysisBlocked("missing_chart", "Falta la especificación del gráfico previamente ejecutado.")
                    record = reused.model_copy(deep=True)
                    message = "Se reutilizó el resultado completo del mismo plan, entradas y configuración; no se repitió el cálculo."
                    if message not in record.warnings:
                        record.warnings.append(message)
                    result.operations[-1] = record
                    records[step.id] = record
                    if step.id in charts:
                        chart = charts[step.id].model_copy(deep=True)
                        # Resultados previos conservan sus datos; sólo se construye
                        # o actualiza su figura, sin repetir SQL ni análisis.
                        chart.figure = build_plotly_figure(chart)
                        budget.check()
                        result.visualizations.append(chart)
                    result.rows_returned += record.rows_returned
                    try:
                        self._output_budget(result)
                    except AnalysisBlocked:
                        result.rows_returned -= record.rows_returned
                        if step.id in charts:
                            result.visualizations.pop()
                        raise
                    continue
                rows = decode_rows(parent, parent.columns, budget)
                record.rows_processed = len(rows)
                record.warnings.extend(parent.warnings)
                chart = None
                if step.operation == "visualize":
                    chart = create_visualization(step, rows, parent.columns,
                        [item for item in result.datasets if item.dataset_id in record.dataset_ids], settings, budget,
                        contextual_warnings=[*result.warnings, *record.warnings])
                    record.warnings.extend(chart.warnings)
                    record.rows = []
                else:
                    record.rows, record.exclusions, messages, filtered = tabular_operation(step, rows, parent.columns, settings, budget)
                    record.rows_filtered = filtered
                    record.warnings.extend(messages)
                    if filtered:
                        record.warnings.append(f"Los filtros explícitos del paso excluyeron {filtered} filas.")
                budget.check()
                record.rows_returned = len(record.rows)
                # Valida los tipos de salida sin perder Decimal mediante JSON.
                validated = OperationExecution.model_validate(record.model_dump())
                decode_rows(validated, step.output_fields, budget)
                record = validated
                record.materialized, record.status = True, "executed"
                record.warnings = list(dict.fromkeys(record.warnings))
                result.operations[-1] = record
                records[step.id] = record
                if chart is not None:
                    result.visualizations.append(chart)
                result.rows_returned += record.rows_returned
                try:
                    self._output_budget(result)
                except AnalysisBlocked:
                    result.rows_returned -= record.rows_returned
                    record.rows, record.rows_returned, record.materialized = [], 0, False
                    if chart is not None:
                        result.visualizations.pop()
                    raise
            except AnalysisBlocked as error:
                record.status = "unsupported" if error.code == "unsupported_operation" else "blocked"
                record.error = ExecutionError(code=error.code, message=str(error))
                record.materialized = False
                record.rows, record.rows_returned = [], 0
            except (DecimalException, ValidationError, ValueError, TypeError, OverflowError):
                record.status, record.materialized = "error", False
                record.rows, record.rows_returned = [], 0
                record.error = ExecutionError(code="invalid_python_result", message="Los datos no permiten completar el cálculo autorizado; no se generaron valores sustitutos.")
            except Exception:
                record.status, record.materialized = "error", False
                record.rows, record.rows_returned = [], 0
                record.error = ExecutionError(code="python_error", message="La herramienta Python no pudo completar la operación; no se generaron resultados ficticios.")
            finally:
                record.duration_ms = round((perf_counter() - operation_started) * 1000, 3)
                rows = None  # No conserva tablas de trabajo entre pasos.
        return self._finish(result, plan, sql_result, started)

    def _output_budget(self, result):
        settings = self._settings
        if (result.rows_returned > settings.max_result_rows or
                len(result.model_dump_json().encode("utf-8")) + 4096 > settings.max_result_bytes):
            raise AnalysisBlocked("result_limit", "El resultado completo excede el límite de filas/bytes de Python; no se publica un resultado recortado como completo.")

    def _finish(self, result, plan, sql_result, started):
        if len(result.model_dump_json().encode("utf-8")) + 4096 > self._settings.max_result_bytes:
            result.error = ExecutionError(code="result_limit", message="Los resultados y sus metadatos exceden el presupuesto; el catálogo completo permanece en PlanningResult.")
            result.datasets, result.visualizations, result.rows_returned = [], [], 0
            result.warnings = [result.error.message]
            for record in result.operations:
                record.rows, record.columns, record.exclusions, record.warnings = [], [], [], []
                record.rows_returned, record.materialized = 0, False
                if record.status == "executed":
                    record.status, record.error = "blocked", result.error
        failures = [item.error for item in result.operations if item.error]
        if failures and not result.error:
            result.error = failures[0]
        result.status, result.pending_operations, _ = summarize_execution(plan, sql_result, result)
        result.finished_at = datetime.now(timezone.utc)
        result.duration_ms = round((perf_counter() - started) * 1000, 3)
        if len(result.model_dump_json().encode("utf-8")) > self._settings.max_result_bytes:
            result.operations, result.datasets, result.visualizations = [], [], []
            result.rows_returned = 0
            result.error = ExecutionError(code="result_limit", message="El detalle excede el presupuesto; consultar el plan para las operaciones pendientes.")
            result.warnings = [result.error.message]
            result.status, result.pending_operations, _ = summarize_execution(plan, sql_result, result)
        return result
