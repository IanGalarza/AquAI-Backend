import asyncio
from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace

from pydantic import ValidationError

from aquai.application.contracts.answering import (
    EvidenceIssue,
    EvidenceReference,
    EvidenceValidationResult,
    SourceReference,
)
from aquai.application.contracts.execution import QueryExecutionState
from aquai.application.errors import AnalysisBlocked, PlannerValidationError
from aquai.application.execution import (
    decode_rows,
    evidence_fingerprint,
    plan_fingerprint,
    sql_result_fingerprint,
    validate_sql_transfer,
)
from aquai.application.planning.validation import validate_execution_plan
from aquai.application.validation.visualizations import validate_visualization

EXECUTED_STATUSES = {"executed", "executed_as_dependency"}
INTERMEDIATE_EVIDENCE_OPERATIONS = {"aggregate", "calculate", "sort", "latest"}


class ValidateEvidenceService:
    async def validate(self, state: QueryExecutionState) -> EvidenceValidationResult:
        return await asyncio.to_thread(self._evaluate, state)

    def _evaluate(self, state):
        return _EvidenceEvaluation(state).run()


class _EvidenceEvaluation:
    def __init__(self, state: QueryExecutionState) -> None:
        self._started = perf_counter()
        self._state = state
        self._plan = state.planning
        self._sql = state.execution
        self._python = state.python_execution
        self._result = EvidenceValidationResult(
            status="failed",
            verified_at=datetime.now(timezone.utc),
            evidence_fingerprint=evidence_fingerprint(self._plan, self._sql, self._python),
            limitations=[
                "Se comprueba la coherencia y completitud de la evidencia, no la validez científica automática de cualquier interpretación."
            ],
        )

        self._sql_trusted = True
        self._python_trusted = True
        self._steps = {}
        self._records = {}
        self._ancestors = {}
        self._dataset_ids = {}
        self._normalized = {}
        self._valid = {}
        self._charts = {}
        self._complete = set()

    def run(self) -> EvidenceValidationResult:
        self._validate()

        return self._finish()

    def _validate(self) -> None:
        self._collect_retrieval_warnings()

        if self._plan is None:
            self._issue("missing_plan", "No existe un plan para contrastar la evidencia.")
            return

        self._collect_execution_diagnostics()

        if self._question_mismatch():
            self._issue("question_mismatch", "La evidencia no corresponde a la pregunta recibida.")
            return

        if self._plan.status != "plannable":
            self._resolve_without_execution()
            return

        if not self._plan_passes_deterministic_checks():
            self._issue(
                "invalid_plan", "El plan no supera las comprobaciones determinísticas existentes."
            )
            return

        self._check_sql_transfer()
        self._check_python_transfer()
        self._index_plan_operations()
        self._check_operation_records()
        self._collect_visualizations()
        self._check_dependencies()
        self._record_operation_progress()
        self._collect_usable_evidence()
        self._collect_sources()
        self._resolve_status()
        self._note_pending_operations()

    def _issue(self, code, message, operation_id=None) -> None:
        self._result.issues.append(
            EvidenceIssue(code=code, message=message, operation_id=operation_id)
        )

    def _finish(self) -> EvidenceValidationResult:
        self._result.warnings = list(dict.fromkeys(self._result.warnings))
        self._result.duration_ms = round((perf_counter() - self._started) * 1000, 3)

        return self._result

    def _collect_retrieval_warnings(self) -> None:
        if self._state.retrieval:
            self._result.warnings.extend(self._state.retrieval.warnings)

    def _collect_execution_diagnostics(self) -> None:
        self._result.warnings.extend(self._plan.warnings)

        for execution in (self._sql, self._python):
            if execution:
                self._collect_diagnostics(execution)

    def _collect_diagnostics(self, execution) -> None:
        self._result.warnings.extend(execution.warnings)

        if execution.error:
            self._result.errors.append(execution.error)

        for record in execution.operations:
            self._result.warnings.extend(record.warnings)
            if record.error and record.error not in self._result.errors:
                self._result.errors.append(record.error)

    def _question_mismatch(self) -> bool:
        question = self._state.question
        retrieval = self._state.retrieval

        if self._plan.context.evidence.get("question") != question:
            return True

        return retrieval is not None and retrieval.question != question

    def _resolve_without_execution(self) -> None:
        plan = self._plan
        execution_received = (
            self._sql is not None
            or self._python is not None
            or plan.operations
            or plan.output_steps
        )

        if execution_received:
            self._issue(
                "unexpected_execution",
                "Se recibieron herramientas para una planificación que no autoriza ejecución.",
            )
            return

        self._result.status = plan.status
        self._result.clarification_question = plan.clarification_question
        self._result.missing_information = list(plan.missing_information)

        clarification_missing = (
            plan.status == "needs_clarification" and not plan.clarification_question
        )
        if clarification_missing:
            self._result.status = "failed"
            self._issue("missing_clarification", "El estado requiere una pregunta de aclaración.")

    def _plan_passes_deterministic_checks(self) -> bool:
        try:
            validate_execution_plan(self._plan, self._plan.selected_datasets)
        except (PlannerValidationError, ValidationError, ValueError):
            return False

        return True

    def _check_sql_transfer(self) -> None:
        try:
            validate_sql_transfer(self._plan, self._sql)
        except (PlannerValidationError, ValidationError):
            self._sql_trusted = False
            self._issue(
                "invalid_sql_transfer",
                "El plan, esquema o procedencia de la transferencia SQL son incoherentes.",
            )

    def _check_python_transfer(self) -> None:
        if not self._python_executed():
            return

        if self._python_transfer_consistent():
            return

        self._python_trusted = False
        self._issue(
            "invalid_python_transfer",
            "Python no conserva el plan, las entradas SQL o sus versiones verificadas.",
        )

    def _python_executed(self) -> bool:
        python = self._python

        return python is not None and any(
            record.status == "executed" for record in python.operations
        )

    def _python_transfer_consistent(self) -> bool:
        python = self._python
        selected = {item.dataset_id: item for item in self._plan.selected_datasets}
        actual = {item.dataset_id: item for item in python.datasets}

        transfer_matches_plan = (
            python.plan_fingerprint == plan_fingerprint(self._plan)
            and python.sql_input_fingerprint == sql_result_fingerprint(self._sql)
            and set(actual) == set(selected)
            and len(actual) == len(python.datasets)
        )
        if not transfer_matches_plan:
            return False

        return all(
            _same_verified_dataset(dataset, selected.get(dataset_id))
            for dataset_id, dataset in actual.items()
        )

    def _index_plan_operations(self) -> None:
        operations = self._plan.operations
        self._steps = {step.id: step for step in operations}

        for step in operations:
            self._ancestors[step.id] = self._lineage_of(step)
            self._dataset_ids[step.id] = self._dataset_ids_of(step)

    def _lineage_of(self, step) -> set:
        lineage = {step.id}

        for parent in step.depends_on:
            lineage |= self._ancestors[parent]

        return lineage

    def _dataset_ids_of(self, step) -> set:
        dataset_ids = {ref.dataset_id for ref in step.columns}

        for parent in step.depends_on:
            dataset_ids |= self._dataset_ids[parent]

        return dataset_ids

    def _check_operation_records(self) -> None:
        transfers = (
            (self._sql, "sql", self._sql_trusted),
            (self._python, "python", self._python_trusted),
        )

        for batch, tool, trusted in transfers:
            if batch is not None:
                self._check_batch_records(batch, tool, trusted)

    def _check_batch_records(self, batch, tool, trusted) -> None:
        seen = set()

        for record in batch.operations:
            step = self._steps.get(record.operation_id)

            if _is_pending_python_marker(record, step, tool):
                continue

            if step is None or step.tool != tool or record.operation_id in seen:
                self._reject_operation_reference(record)
                continue

            seen.add(record.operation_id)
            self._records[step.id] = record
            self._valid[step.id] = False

            inputs_untrusted = not trusted or (tool == "python" and not self._sql_trusted)
            if inputs_untrusted:
                continue

            self._check_operation_result(step, record)

    def _reject_operation_reference(self, record) -> None:
        self._issue(
            "invalid_operation_reference",
            "Una transferencia repite operaciones o contiene referencias ajenas.",
            record.operation_id,
        )

        if record.operation_id in self._steps:
            self._valid[record.operation_id] = False

    def _check_operation_result(self, step, record) -> None:
        if record.status not in EXECUTED_STATUSES or record.error:
            self._issue("operation_incomplete", "La operación no terminó correctamente.", step.id)
            return

        if self._schema_or_sources_differ(step, record):
            self._issue(
                "invalid_schema_or_sources",
                "El esquema, las unidades o los datasets de la operación no coinciden con el plan.",
                step.id,
            )
            return

        if _fully_materialized(record):
            self._normalize_rows(step, record)
            return

        if _dependency_without_rows(record):
            return  # Su completitud se contrasta con consultas finales realmente ejecutadas.

        self._issue(
            "incomplete_materialization",
            "El resultado está truncado, incompleto o no conserva su cantidad de filas.",
            step.id,
        )

    def _schema_or_sources_differ(self, step, record) -> bool:
        expected_ids = self._dataset_ids[step.id]

        return (
            record.columns != step.output_fields
            or set(record.dataset_ids) != expected_ids
            or len(record.dataset_ids) != len(expected_ids)
        )

    def _normalize_rows(self, step, record) -> None:
        try:
            self._normalized[step.id] = decode_rows(record, step.output_fields)
            self._valid[step.id] = True
        except (AnalysisBlocked, ValueError, TypeError):
            self._issue(
                "invalid_values",
                "Los valores no corresponden a los tipos verificados.",
                step.id,
            )

    def _collect_visualizations(self) -> None:
        if not self._python:
            return

        for chart in self._python.visualizations:
            if chart.operation_id in self._charts:
                self._valid[chart.operation_id] = False
                self._issue(
                    "duplicate_visualization",
                    "Se repitió una visualización.",
                    chart.operation_id,
                )
                continue

            self._charts[chart.operation_id] = chart

    def _check_dependencies(self) -> None:
        for step in self._plan.operations:
            if not self._valid.get(step.id):
                continue

            if step.tool == "sql":
                self._check_sql_lineage(step)
            elif step.tool == "python":
                self._check_python_dependencies(step)

    def _check_sql_lineage(self, step) -> None:
        lineage = self._ancestors[step.id]
        lineage_traced = all(self._ancestor_traced(key, step.id) for key in lineage)

        if not lineage_traced:
            self._valid[step.id] = False
            self._issue(
                "invalid_sql_lineage",
                "Falta la traza de las dependencias de la consulta SQL completa.",
                step.id,
            )
            return

        self._complete.update(lineage)

    def _ancestor_traced(self, key, query_id) -> bool:
        ancestor = self._steps[key]

        if ancestor.tool != "sql" or key not in self._records:
            return False

        record = self._records[key]

        return (
            query_id in record.query_operation_ids
            and record.columns == ancestor.output_fields
            and set(record.dataset_ids) == self._dataset_ids[key]
        )

    def _check_python_dependencies(self, step) -> None:
        record = self._records.get(step.id)

        if self._dependencies_unusable(step, record):
            self._valid[step.id] = False
            self._issue(
                "incomplete_dependency",
                "Python depende de un resultado ausente, bloqueado o truncado.",
                step.id,
            )
            return

        try:
            self._validate_python_evidence(step, record)
            self._complete.add(step.id)
        except (PlannerValidationError, ValueError, TypeError, KeyError, OverflowError):
            self._valid[step.id] = False
            self._issue(
                "invalid_python_evidence",
                "Los contadores, exclusiones o la figura no conservan la evidencia de entrada.",
                step.id,
            )

    def _dependencies_unusable(self, step, record) -> bool:
        if record.input_operation_ids != step.depends_on:
            return True

        return any(self._dependency_unusable(parent) for parent in step.depends_on)

    def _dependency_unusable(self, parent) -> bool:
        return (
            parent not in self._complete
            or not self._valid.get(parent)
            or not self._records[parent].materialized
            or self._records[parent].truncated
        )

    def _validate_python_evidence(self, step, record) -> None:
        parent = self._records[step.depends_on[0]]

        _validate_input_counters(record, parent)
        _validate_exclusions(step, record, self._normalized.get(step.id, []))

        if step.operation == "visualize":
            self._validate_chart(step, parent)

    def _validate_chart(self, step, parent) -> None:
        chart = self._charts.get(step.id)
        if chart is None:
            raise PlannerValidationError("Falta la visualización de la operación ejecutada.")

        sources = [
            item for item in self._python.datasets if item.dataset_id in self._dataset_ids[step.id]
        ]
        validate_visualization(chart, step, parent, self._normalized[step.depends_on[0]], sources)

        self._result.warnings.extend(chart.warnings)

    def _record_operation_progress(self) -> None:
        operations = self._plan.operations
        complete = self._complete

        self._result.completed_operations = [step.id for step in operations if step.id in complete]
        self._result.pending_operations = [
            step.id for step in operations if step.id not in complete
        ]

    def _collect_usable_evidence(self) -> None:
        requested_ancestors = self._requested_ancestors()

        for step in self._plan.operations:
            if self._usable_as_evidence(step, requested_ancestors):
                self._result.usable_evidence.append(self._evidence_reference(step))

    def _requested_ancestors(self) -> set:
        requested = set()

        for output in self._plan.output_steps:
            requested |= self._ancestors[output]

        return requested

    def _usable_as_evidence(self, step, requested_ancestors) -> bool:
        materialized = (
            step.id in self._complete
            and self._valid.get(step.id)
            and self._records[step.id].materialized
        )
        if not materialized:
            return False

        if step.id in self._plan.output_steps:
            return True

        return step.id in requested_ancestors and step.operation in INTERMEDIATE_EVIDENCE_OPERATIONS

    def _evidence_reference(self, step) -> EvidenceReference:
        return EvidenceReference(
            evidence_id=f"e{len(self._result.usable_evidence) + 1}",
            operation_id=step.id,
            tool=step.tool,
            kind="visualization" if step.operation == "visualize" else "table",
            dataset_ids=sorted(self._dataset_ids[step.id], key=str),
            rows=self._records[step.id].rows_returned,
            is_requested_output=step.id in self._plan.output_steps,
        )

    def _collect_sources(self) -> None:
        used = set()
        for evidence in self._result.usable_evidence:
            used.update(evidence.dataset_ids)

        self._result.sources = [
            _source_reference(dataset)
            for dataset in self._plan.selected_datasets
            if dataset.dataset_id in used
        ]

    def _resolve_status(self) -> None:
        if not self._result.usable_evidence:
            return

        self._result.status = "validated" if self._fully_validated() else "partial"

    def _fully_validated(self) -> bool:
        result = self._result

        if result.pending_operations or result.errors or result.issues:
            return False

        return all(self._has_evidence_for(output) for output in self._plan.output_steps)

    def _has_evidence_for(self, operation_id) -> bool:
        return any(item.operation_id == operation_id for item in self._result.usable_evidence)

    def _note_pending_operations(self) -> None:
        if self._result.pending_operations:
            self._result.limitations.append(
                "Hay operaciones pendientes o no utilizables; no se presenta la consulta completa como resuelta."
            )


def _same_verified_dataset(dataset, expected) -> bool:
    if expected is None:
        return False

    return (
        dataset.source_id == expected.source_id
        and dataset.version == expected.verified_version
        and dataset.provenance == expected.provenance
        and dataset.columns == expected.columns_metadata
    )


def _is_pending_python_marker(record, step, tool) -> bool:
    # Marcador del SQL Agent, no resultado Python.
    return (
        tool == "sql"
        and step is not None
        and step.tool == "python"
        and record.status == "pending_python"
    )


def _fully_materialized(record) -> bool:
    return record.materialized and not record.truncated and record.rows_returned == len(record.rows)


def _dependency_without_rows(record) -> bool:
    return record.status == "executed_as_dependency" and not record.rows and not record.materialized


def _source_reference(dataset) -> SourceReference:
    return SourceReference(
        source_id=dataset.source_id,
        dataset_id=dataset.dataset_id,
        dataset_name=dataset.name,
        version=dataset.verified_version,
        provenance=dataset.provenance,
    )


def _validate_input_counters(record, parent) -> None:
    counters_differ = (
        record.rows_processed != parent.rows_returned
        or not 0 <= record.rows_filtered <= record.rows_processed
    )
    if counters_differ:
        raise PlannerValidationError("Los contadores de entrada Python no son coherentes.")


def _validate_exclusions(step, record, rows) -> None:
    calculations = {item.output_name: item for item in step.calculations}
    totals = {key: 0 for key in calculations}
    group_names = [group.output_name for group in step.group_by]
    rows_by_group = _group_rows(rows, group_names)
    seen = set()

    for exclusion in record.exclusions:
        calculation = calculations.get(exclusion.output_name)
        if _exclusion_unauthorized(exclusion, calculation, step):
            raise PlannerValidationError(
                "Las exclusiones no corresponden a los cálculos autorizados."
            )

        totals[exclusion.output_name] += exclusion.input_rows

        if _requires_group_match(step, calculation):
            _validate_exclusion_group(exclusion, record, group_names, rows_by_group, seen)

    if not calculations:
        return

    filtered_rows = record.rows_processed - record.rows_filtered
    if any(total != filtered_rows for total in totals.values()):
        raise PlannerValidationError(
            "Faltan exclusiones o se repiten observaciones en sus contadores."
        )


def _group_rows(rows, group_names) -> dict:
    grouped = {}

    for row in rows:
        group_key = tuple(row[name] for name in group_names)
        grouped.setdefault(group_key, []).append(row)

    return grouped


def _exclusion_unauthorized(exclusion, calculation, step) -> bool:
    return (
        calculation is None
        or exclusion.policy != calculation.null_policy
        or exclusion.input_rows != exclusion.valid_rows + exclusion.null_rows
        or exclusion.zero_divisions > exclusion.valid_rows
        or set(exclusion.group) != {group.output_name for group in step.group_by}
    )


def _requires_group_match(step, calculation) -> bool:
    if step.operation == "aggregate":
        return True

    return step.operation == "calculate" and calculation.function == "correlation"


def _validate_exclusion_group(exclusion, record, group_names, rows_by_group, seen) -> None:
    group_fields = [field for field in record.columns if field.name in exclusion.group]
    group = decode_rows(SimpleNamespace(rows=[exclusion.group]), group_fields)[0]
    group_key = tuple(group[name] for name in group_names)

    counter_key = (exclusion.output_name, group_key)
    if counter_key in seen:
        raise PlannerValidationError("Un contador de exclusiones repite su grupo.")
    seen.add(counter_key)

    matches = rows_by_group.get(group_key, [])
    if len(matches) != 1 or _preserved_nulls_lost(exclusion, matches[0]):
        raise PlannerValidationError("El resultado no conserva su grupo o su política de nulos.")


def _preserved_nulls_lost(exclusion, row) -> bool:
    return (
        exclusion.policy == "preserve"
        and exclusion.null_rows > 0
        and row[exclusion.output_name] is not None
    )
