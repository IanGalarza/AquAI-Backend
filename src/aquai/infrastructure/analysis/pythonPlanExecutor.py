from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import DecimalException
from time import perf_counter

from pydantic import ValidationError

from aquai.application.contracts.execution import (
    ExecutionDataset,
    ExecutionError,
    OperationExecution,
    PythonExecutionResult,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import AnalysisBlocked, PlannerValidationError
from aquai.application.execution import (
    decode_rows,
    execution_fingerprint,
    plan_fingerprint,
    sql_result_fingerprint,
    summarize_execution,
    validate_sql_transfer,
)
from aquai.application.planning.temporalText import period_bounds
from aquai.application.planning.validation import validate_execution_plan
from aquai.infrastructure.analysis.operations import tabular_operation
from aquai.infrastructure.analysis.plotlyFigures import build_plotly_figure
from aquai.infrastructure.analysis.visualizations import create_visualization
from aquai.infrastructure.config.pythonExecution import PythonExecutionSettings

_PYTHON_OPERATIONS = {"aggregate", "calculate", "sort", "latest", "visualize"}
_REUSED_RESULT_WARNING = (
    "Se reutilizó el resultado completo del mismo plan, entradas y configuración; "
    "no se repitió el cálculo."
)

# Margen para los metadatos que se agregan al cerrar el resultado.
_ENVELOPE_RESERVE_BYTES = 4096


def _json_size(model) -> int:
    return len(model.model_dump_json().encode("utf-8"))


class _Budget:
    def __init__(self, settings, started):
        self.settings = settings
        self.started = started

    def check(self):
        elapsed_ms = (perf_counter() - self.started) * 1000

        if elapsed_ms > self.settings.max_duration_ms:
            raise AnalysisBlocked(
                "python_timeout", "Se superó el presupuesto de tiempo de las herramientas Python."
            )

    def input(self, record):
        self.check()
        settings = self.settings
        oversized = (
            len(record.rows) > settings.max_input_rows
            or len(record.rows) * len(record.columns) > settings.max_cells
        )

        if oversized:
            raise AnalysisBlocked(
                "input_limit",
                "La entrada completa supera los límites de filas/celdas de Python; reducila explícitamente mediante SQL.",
            )

        size = _json_size(record)

        if size > settings.max_input_bytes or size * 8 > settings.max_memory_bytes:
            raise AnalysisBlocked(
                "input_limit",
                "La entrada completa supera el presupuesto de bytes/memoria; no se analiza un prefijo.",
            )


@dataclass
class _Run:
    plan: PlanningResult
    result: PythonExecutionResult
    records: dict[str, OperationExecution]
    steps: dict
    cached: dict[str, OperationExecution]
    charts: dict
    budget: _Budget


@dataclass
class _StepState:
    """Registro vigente del paso: el cálculo puede reemplazarlo por una versión validada."""

    step: object
    record: OperationExecution


class DeterministicPythonPlanExecutor:
    def __init__(self, settings: PythonExecutionSettings) -> None:
        self._settings = settings

    def execute(
        self,
        plan: PlanningResult,
        sql_result: SqlExecutionResult | None,
        previous: PythonExecutionResult | None = None,
    ) -> PythonExecutionResult:
        started, now = perf_counter(), datetime.now(timezone.utc)
        budget = _Budget(self._settings, started)
        result = self._new_result(plan, sql_result, now)

        if not self._authorize_input(result, plan, sql_result):
            return self._finish(result, plan, sql_result, started)

        if not self._attach_datasets(result, plan):
            return self._finish(result, plan, sql_result, started)

        records = {item.operation_id: item for item in sql_result.operations} if sql_result else {}
        steps = {step.id: step for step in plan.operations}
        cached, charts = self._reusable_results(previous, result.input_fingerprint)
        run = _Run(plan, result, records, steps, cached, charts, budget)

        for step in plan.operations:
            if step.tool == "python":
                self._run_step(run, step)

        return self._finish(result, plan, sql_result, started)

    def _new_result(self, plan, sql_result, now):
        settings = self._settings
        precision_warning = f"Cálculos NUMERIC con Decimal y precisión de {settings.decimal_precision} dígitos significativos; no se convierten silenciosamente a float."

        return PythonExecutionResult(
            status="not_executed",
            started_at=now,
            finished_at=now,
            plan_fingerprint=plan_fingerprint(plan),
            sql_input_fingerprint=sql_result_fingerprint(sql_result),
            output_operations=list(plan.output_steps),
            decimal_precision=settings.decimal_precision,
            warnings=list(dict.fromkeys([*plan.warnings, precision_warning])),
        )

    def _authorize_input(self, result, plan, sql_result) -> bool:
        try:
            validate_execution_plan(plan, plan.selected_datasets)
            validate_sql_transfer(plan, sql_result)
            result.input_fingerprint = execution_fingerprint(
                plan, sql_result, self._settings.model_dump_json()
            )
        except (PlannerValidationError, ValidationError):
            result.error = ExecutionError(
                code="invalid_python_input",
                message="El plan o la transferencia no están autorizados para ejecución Python.",
            )
            return False

        return True

    def _attach_datasets(self, result, plan) -> bool:
        result.datasets = [
            ExecutionDataset(
                dataset_id=item.dataset_id,
                source_id=item.source_id,
                version=item.verified_version,
                provenance=item.provenance,
                columns=item.columns_metadata,
                warnings=item.warnings,
            )
            for item in plan.selected_datasets
        ]

        try:
            self._output_budget(result)
        except AnalysisBlocked as error:
            result.datasets = []
            result.warnings = [
                "Los metadatos completos permanecen en el plan y superan el presupuesto del resultado Python."
            ]
            result.error = ExecutionError(code=error.code, message=str(error))
            return False

        return True

    @staticmethod
    def _reusable_results(previous, input_fingerprint):
        if previous is None or previous.input_fingerprint != input_fingerprint:
            return {}, {}

        cached = {
            item.operation_id: item
            for item in previous.operations
            if item.status == "executed" and not item.truncated
        }
        charts = {chart.operation_id: chart for chart in previous.visualizations}

        return cached, charts

    def _run_step(self, run, step):
        state = _StepState(step, self._new_record(step, run.plan, run.records))
        run.result.operations.append(state.record)
        run.records[step.id] = state.record
        operation_started = perf_counter()

        try:
            self._execute_step(run, state)
        except AnalysisBlocked as error:
            self._mark_blocked(state.record, error)
        except (DecimalException, ValidationError, ValueError, TypeError, OverflowError):
            self._mark_failed(
                state.record,
                "invalid_python_result",
                "Los datos no permiten completar el cálculo autorizado; no se generaron valores sustitutos.",
            )
        except Exception:
            self._mark_failed(
                state.record,
                "python_error",
                "La herramienta Python no pudo completar la operación; no se generaron resultados ficticios.",
            )
        finally:
            state.record.duration_ms = round((perf_counter() - operation_started) * 1000, 3)

    @staticmethod
    def _new_record(step, plan, records):
        record = OperationExecution(
            operation_id=step.id,
            status="not_executed",
            columns=step.output_fields,
            input_operation_ids=list(step.depends_on),
            warnings=[
                issue.message for issue in plan.validation_issues if issue.step_id == step.id
            ],
        )
        declared = {ref.dataset_id for field in step.output_fields for ref in field.sources}
        inherited = {
            dataset_id
            for parent in step.depends_on
            if parent in records
            for dataset_id in records[parent].dataset_ids
        }
        record.dataset_ids = sorted(declared | inherited, key=str)

        return record

    @staticmethod
    def _mark_blocked(record, error):
        record.status = "unsupported" if error.code == "unsupported_operation" else "blocked"
        record.error = ExecutionError(code=error.code, message=str(error))
        record.materialized = False
        record.rows, record.rows_returned = [], 0

    @staticmethod
    def _mark_failed(record, code, message):
        record.status, record.materialized = "error", False
        record.rows, record.rows_returned = [], 0
        record.error = ExecutionError(code=code, message=message)

    def _execute_step(self, run, state):
        step = state.step
        run.budget.check()
        parent = self._authorized_parent(run, step)
        run.budget.input(parent)
        reusable = step.id in run.cached and run.cached[step.id].columns == step.output_fields

        if reusable:
            self._reuse_cached(run, state, run.cached[step.id])
        else:
            self._compute(run, state, parent)

    @staticmethod
    def _authorized_parent(run, step):
        if step.operation not in _PYTHON_OPERATIONS:
            raise AnalysisBlocked(
                "unsupported_operation",
                "La operación todavía no está habilitada en Python.",
            )

        parents = [run.records.get(parent) for parent in step.depends_on]

        if len(parents) != 1 or parents[0] is None:
            raise AnalysisBlocked(
                "missing_dependency",
                "Falta el resultado autorizado de la dependencia; Python no vuelve a consultar PostgreSQL.",
            )

        parent = parents[0]

        if parent.truncated:
            raise AnalysisBlocked(
                "incomplete_input",
                "La entrada está truncada; no se calculan estadísticas ni gráficos de una muestra accidental.",
            )

        if parent.status != "executed" or not parent.materialized:
            raise AnalysisBlocked(
                "missing_dependency",
                "La dependencia no está completa/materializada; permanece pendiente.",
            )

        expected_columns = run.steps[step.depends_on[0]].output_fields

        if parent.columns != expected_columns or parent.rows_returned != len(parent.rows):
            raise AnalysisBlocked(
                "invalid_schema",
                "La dependencia no coincide con el esquema o la cantidad de filas autorizados.",
            )

        return parent

    def _reuse_cached(self, run, state, reused):
        step = state.step

        if not reused.materialized or reused.rows_returned != len(reused.rows):
            raise AnalysisBlocked(
                "invalid_cached_result", "El resultado anterior no está completo."
            )

        run.budget.input(reused)
        decode_rows(reused, step.output_fields, run.budget)

        if step.operation == "visualize" and step.id not in run.charts:
            raise AnalysisBlocked(
                "missing_chart",
                "Falta la especificación del gráfico previamente ejecutado.",
            )

        record = reused.model_copy(deep=True)
        state.record = record

        if _REUSED_RESULT_WARNING not in record.warnings:
            record.warnings.append(_REUSED_RESULT_WARNING)

        self._replace_record(run, step, record)
        chart = None

        if step.id in run.charts:
            chart = self._rebuilt_chart(run.charts[step.id], run.budget)

        self._publish(run.result, record, chart)

    @staticmethod
    def _rebuilt_chart(previous_chart, budget):
        chart = previous_chart.model_copy(deep=True)
        # Conserva los datos del gráfico previo; sólo reconstruye su figura, sin repetir SQL ni análisis.
        chart.figure = build_plotly_figure(chart)
        budget.check()

        return chart

    def _compute(self, run, state, parent):
        step, record = state.step, state.record
        rows = decode_rows(parent, parent.columns, run.budget)
        record.rows_processed = len(rows)
        record.warnings.extend(parent.warnings)
        chart = None

        if step.operation == "visualize":
            chart = self._visualize(run, state, rows, parent)
        else:
            self._tabulate(run, state, rows, parent)

        run.budget.check()
        record.rows_returned = len(record.rows)
        # Valida los tipos de salida sin perder Decimal mediante JSON.
        validated = OperationExecution.model_validate(record.model_dump())
        decode_rows(validated, step.output_fields, run.budget)

        state.record = validated
        validated.materialized, validated.status = True, "executed"
        validated.warnings = list(dict.fromkeys(validated.warnings))
        self._replace_record(run, step, validated)
        self._publish(run.result, validated, chart)

    def _visualize(self, run, state, rows, parent):
        record = state.record
        datasets = [item for item in run.result.datasets if item.dataset_id in record.dataset_ids]
        chart = create_visualization(
            state.step,
            rows,
            parent.columns,
            datasets,
            self._settings,
            run.budget,
            contextual_warnings=[*run.result.warnings, *record.warnings],
        )
        record.warnings.extend(chart.warnings)
        record.rows = []

        return chart

    def _tabulate(self, run, state, rows, parent):
        record = state.record
        period_key = self._period_key(run.plan, state.step, parent.columns)
        record.rows, record.exclusions, messages, filtered = tabular_operation(
            state.step, rows, parent.columns, self._settings, run.budget, period_key=period_key
        )
        record.rows_filtered = filtered
        record.warnings.extend(messages)

        if filtered:
            record.warnings.append(f"Los filtros explícitos del paso excluyeron {filtered} filas.")

    @staticmethod
    def _replace_record(run, step, record):
        run.result.operations[-1] = record
        run.records[step.id] = record

    def _publish(self, result, record, chart):
        if chart is not None:
            result.visualizations.append(chart)

        result.rows_returned += record.rows_returned

        try:
            self._output_budget(result)
        except AnalysisBlocked:
            result.rows_returned -= record.rows_returned

            if chart is not None:
                result.visualizations.pop()

            raise

    @staticmethod
    def _period_key(plan, step, fields):
        if step.operation != "latest":
            return None

        field = next((item for item in fields if item.name == step.order_by[0].field.field), None)

        if field is None or field.sql_type != "TEXT" or len(field.sources) != 1:
            return None

        source = field.sources[0]
        column = next(
            (
                item
                for item in plan.temporal_text_columns
                if (item.dataset_id, item.column) == (source.dataset_id, source.column)
            ),
            None,
        )

        if column is None:
            return None

        def period_end(value):
            bounds = period_bounds(value, column.day_first)
            return bounds[1] if bounds else None

        return period_end

    def _output_budget(self, result):
        settings = self._settings

        if (
            result.rows_returned > settings.max_result_rows
            or _json_size(result) + _ENVELOPE_RESERVE_BYTES > settings.max_result_bytes
        ):
            raise AnalysisBlocked(
                "result_limit",
                "El resultado completo excede el límite de filas/bytes de Python; no se publica un resultado recortado como completo.",
            )

    def _finish(self, result, plan, sql_result, started):
        if _json_size(result) + _ENVELOPE_RESERVE_BYTES > self._settings.max_result_bytes:
            self._discard_details(result)

        failures = [item.error for item in result.operations if item.error]

        if failures and not result.error:
            result.error = failures[0]

        result.status, result.pending_operations, _ = summarize_execution(plan, sql_result, result)
        result.finished_at = datetime.now(timezone.utc)
        result.duration_ms = round((perf_counter() - started) * 1000, 3)

        if _json_size(result) > self._settings.max_result_bytes:
            self._discard_operations(result, plan, sql_result)

        return result

    @staticmethod
    def _discard_details(result):
        result.error = ExecutionError(
            code="result_limit",
            message="Los resultados y sus metadatos exceden el presupuesto; el catálogo completo permanece en PlanningResult.",
        )
        result.datasets, result.visualizations, result.rows_returned = [], [], 0
        result.warnings = [result.error.message]

        for record in result.operations:
            record.rows, record.columns, record.exclusions, record.warnings = [], [], [], []
            record.rows_returned, record.materialized = 0, False

            if record.status == "executed":
                record.status, record.error = "blocked", result.error

    @staticmethod
    def _discard_operations(result, plan, sql_result):
        result.operations, result.datasets, result.visualizations = [], [], []
        result.rows_returned = 0
        result.error = ExecutionError(
            code="result_limit",
            message="El detalle excede el presupuesto; consultar el plan para las operaciones pendientes.",
        )
        result.warnings = [result.error.message]
        result.status, result.pending_operations, _ = summarize_execution(plan, sql_result, result)
