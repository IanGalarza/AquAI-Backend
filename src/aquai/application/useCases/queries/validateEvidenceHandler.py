"""Validator determinístico de evidencia: sin LLM, SQL ni recálculo estadístico."""

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from types import SimpleNamespace

from pydantic import ValidationError

from aquai.application.contracts.answering import EvidenceIssue, EvidenceReference, EvidenceValidationResult, SourceReference
from aquai.application.contracts.execution import QueryExecutionState
from aquai.application.errors import AnalysisBlocked, PlannerValidationError
from aquai.application.execution import decode_rows, evidence_fingerprint, plan_fingerprint, sql_result_fingerprint, validate_sql_transfer
from aquai.application.useCases.queries.executeSqlPlanHandler import validate_execution_plan
from aquai.application.validation.visualizations import validate_visualization


class ValidateEvidenceHandler:
    async def handle(self, state: QueryExecutionState) -> EvidenceValidationResult:
        return await asyncio.to_thread(self._validate, state)

    def _validate(self, state):
        started = perf_counter()
        plan, sql, python = state.planning, state.execution, state.python_execution
        result = EvidenceValidationResult(status="failed", verified_at=datetime.now(timezone.utc),
            evidence_fingerprint=evidence_fingerprint(plan, sql, python),
            limitations=["Se comprueba la coherencia y completitud de la evidencia, no la validez científica automática de cualquier interpretación."])

        def issue(code, message, operation_id=None):
            result.issues.append(EvidenceIssue(code=code, message=message, operation_id=operation_id))

        def finish():
            result.warnings = list(dict.fromkeys(result.warnings))
            result.duration_ms = round((perf_counter() - started) * 1000, 3)
            return result

        if state.retrieval:
            result.warnings.extend(state.retrieval.warnings)
        if plan is None:
            issue("missing_plan", "No existe un plan para contrastar la evidencia.")
            return finish()
        result.warnings.extend(plan.warnings)
        for item in (sql, python):
            if item:
                result.warnings.extend(item.warnings)
                if item.error:
                    result.errors.append(item.error)
                for record in item.operations:
                    result.warnings.extend(record.warnings)
                    if record.error and record.error not in result.errors:
                        result.errors.append(record.error)
        if (plan.context.evidence.get("question") != state.question or
                (state.retrieval is not None and state.retrieval.question != state.question)):
            issue("question_mismatch", "La evidencia no corresponde a la pregunta recibida.")
            return finish()
        if plan.status != "plannable":
            if sql is not None or python is not None or plan.operations or plan.output_steps:
                issue("unexpected_execution", "Se recibieron herramientas para una planificación que no autoriza ejecución.")
                return finish()
            result.status = plan.status
            result.clarification_question = plan.clarification_question
            result.missing_information = list(plan.missing_information)
            if plan.status == "needs_clarification" and not plan.clarification_question:
                result.status = "failed"
                issue("missing_clarification", "El estado requiere una pregunta de aclaración.")
            return finish()
        try:
            validate_execution_plan(plan, plan.selected_datasets)
        except (PlannerValidationError, ValidationError, ValueError):
            issue("invalid_plan", "El plan no supera las comprobaciones determinísticas existentes.")
            return finish()

        sql_trusted, python_trusted = True, True
        try:
            validate_sql_transfer(plan, sql)
        except (PlannerValidationError, ValidationError):
            sql_trusted = False
            issue("invalid_sql_transfer", "El plan, esquema o procedencia de la transferencia SQL son incoherentes.")
        selected = {item.dataset_id: item for item in plan.selected_datasets}
        if python is not None and any(record.status == "executed" for record in python.operations):
            actual = {item.dataset_id: item for item in python.datasets}
            python_trusted = (python.plan_fingerprint == plan_fingerprint(plan)
                and python.sql_input_fingerprint == sql_result_fingerprint(sql)
                and set(actual) == set(selected) and len(actual) == len(python.datasets))
            for key, item in actual.items():
                expected = selected.get(key)
                python_trusted = python_trusted and expected is not None and (
                    item.source_id == expected.source_id and item.version == expected.verified_version
                    and item.provenance == expected.provenance and item.columns == expected.columns_metadata)
            if not python_trusted:
                issue("invalid_python_transfer", "Python no conserva el plan, las entradas SQL o sus versiones verificadas.")

        steps = {step.id: step for step in plan.operations}
        records, ancestors, ids, normalized, valid = {}, {}, {}, {}, {}
        for step in plan.operations:
            ancestors[step.id] = {step.id} | {item for parent in step.depends_on for item in ancestors[parent]}
            ids[step.id] = {ref.dataset_id for ref in step.columns} | {key for parent in step.depends_on for key in ids[parent]}
        for batch, tool, trusted in ((sql, "sql", sql_trusted), (python, "python", python_trusted)):
            if batch is None:
                continue
            seen = set()
            for record in batch.operations:
                step = steps.get(record.operation_id)
                if tool == "sql" and step is not None and step.tool == "python" and record.status == "pending_python":
                    continue  # Marcador del SQL Agent, no resultado Python.
                if step is None or step.tool != tool or record.operation_id in seen:
                    issue("invalid_operation_reference", "Una transferencia repite operaciones o contiene referencias ajenas.", record.operation_id)
                    if record.operation_id in steps:
                        valid[record.operation_id] = False
                    continue
                seen.add(record.operation_id)
                records[step.id] = record
                valid[step.id] = False
                if not trusted or (tool == "python" and not sql_trusted):
                    continue
                if record.status not in {"executed", "executed_as_dependency"} or record.error:
                    issue("operation_incomplete", "La operación no terminó correctamente.", step.id)
                    continue
                if record.columns != step.output_fields or set(record.dataset_ids) != ids[step.id] or len(record.dataset_ids) != len(ids[step.id]):
                    issue("invalid_schema_or_sources", "El esquema, las unidades o los datasets de la operación no coinciden con el plan.", step.id)
                    continue
                if record.materialized and not record.truncated and record.rows_returned == len(record.rows):
                    try:
                        normalized[step.id] = decode_rows(record, step.output_fields)
                        valid[step.id] = True
                    except (AnalysisBlocked, ValueError, TypeError):
                        issue("invalid_values", "Los valores no corresponden a los tipos verificados.", step.id)
                elif record.status == "executed_as_dependency" and not record.rows and not record.materialized:
                    pass  # Su completitud se contrasta con consultas finales realmente ejecutadas.
                else:
                    issue("incomplete_materialization", "El resultado está truncado, incompleto o no conserva su cantidad de filas.", step.id)

        charts = {}
        if python:
            for chart in python.visualizations:
                if chart.operation_id in charts:
                    valid[chart.operation_id] = False
                    issue("duplicate_visualization", "Se repitió una visualización.", chart.operation_id)
                else:
                    charts[chart.operation_id] = chart
        complete = set()
        for step in plan.operations:
            record = records.get(step.id)
            if step.tool == "sql" and valid.get(step.id):
                lineage = ancestors[step.id]
                if all(steps[key].tool == "sql" and key in records and step.id in records[key].query_operation_ids
                       and records[key].columns == steps[key].output_fields
                       and set(records[key].dataset_ids) == ids[key] for key in lineage):
                    complete.update(lineage)
                else:
                    valid[step.id] = False
                    issue("invalid_sql_lineage", "Falta la traza de las dependencias de la consulta SQL completa.", step.id)
            elif step.tool == "python" and valid.get(step.id):
                if (record.input_operation_ids != step.depends_on or
                        any(parent not in complete or not valid.get(parent) or not records[parent].materialized
                            or records[parent].truncated for parent in step.depends_on)):
                    valid[step.id] = False
                    issue("incomplete_dependency", "Python depende de un resultado ausente, bloqueado o truncado.", step.id)
                    continue
                try:
                    parent = records[step.depends_on[0]]
                    if record.rows_processed != parent.rows_returned or not 0 <= record.rows_filtered <= record.rows_processed:
                        raise PlannerValidationError("Los contadores de entrada Python no son coherentes.")
                    self._validate_exclusions(step, record, normalized.get(step.id, []))
                    if step.operation == "visualize":
                        chart = charts.get(step.id)
                        if chart is None:
                            raise PlannerValidationError("Falta la visualización de la operación ejecutada.")
                        sources = [item for item in python.datasets if item.dataset_id in ids[step.id]]
                        validate_visualization(chart, step, parent, normalized[step.depends_on[0]], sources)
                        result.warnings.extend(chart.warnings)
                    complete.add(step.id)
                except (PlannerValidationError, ValueError, TypeError, KeyError, OverflowError):
                    valid[step.id] = False
                    issue("invalid_python_evidence", "Los contadores, exclusiones o la figura no conservan la evidencia de entrada.", step.id)
        result.completed_operations = [step.id for step in plan.operations if step.id in complete]
        result.pending_operations = [step.id for step in plan.operations if step.id not in complete]
        requested_ancestors = {key for output in plan.output_steps for key in ancestors[output]}
        for step in plan.operations:
            if (step.id in complete and valid.get(step.id) and records[step.id].materialized
                    and (step.id in plan.output_steps or (step.id in requested_ancestors and step.operation in {"aggregate", "calculate", "sort", "latest"}))):
                result.usable_evidence.append(EvidenceReference(evidence_id=f"e{len(result.usable_evidence) + 1}",
                    operation_id=step.id, tool=step.tool, kind="visualization" if step.operation == "visualize" else "table",
                    dataset_ids=sorted(ids[step.id], key=str), rows=records[step.id].rows_returned,
                    is_requested_output=step.id in plan.output_steps))
        used = {key for item in result.usable_evidence for key in item.dataset_ids}
        result.sources = [SourceReference(source_id=item.source_id, dataset_id=item.dataset_id,
            dataset_name=item.name, version=item.verified_version, provenance=item.provenance)
            for item in plan.selected_datasets if item.dataset_id in used]
        if result.usable_evidence:
            result.status = "validated" if not result.pending_operations and not result.errors and not result.issues and all(
                any(item.operation_id == key for item in result.usable_evidence) for key in plan.output_steps) else "partial"
        if result.pending_operations:
            result.limitations.append("Hay operaciones pendientes o no utilizables; no se presenta la consulta completa como resuelta.")
        return finish()

    @staticmethod
    def _validate_exclusions(step, record, rows):
        calculations = {item.output_name: item for item in step.calculations}
        totals = {key: 0 for key in calculations}
        grouped = {}
        names = [group.output_name for group in step.group_by]
        for row in rows:
            grouped.setdefault(tuple(row[name] for name in names), []).append(row)
        seen = set()
        for item in record.exclusions:
            calculation = calculations.get(item.output_name)
            if (calculation is None or item.policy != calculation.null_policy
                    or item.input_rows != item.valid_rows + item.null_rows or item.zero_divisions > item.valid_rows
                    or set(item.group) != {group.output_name for group in step.group_by}):
                raise PlannerValidationError("Las exclusiones no corresponden a los cálculos autorizados.")
            totals[item.output_name] += item.input_rows
            if step.operation in {"aggregate", "calculate"} and (step.operation == "aggregate" or calculation.function == "correlation"):
                group_fields = [field for field in record.columns if field.name in item.group]
                group = decode_rows(SimpleNamespace(rows=[item.group]), group_fields)[0]
                key = (item.output_name, tuple(group[name] for name in names))
                if key in seen:
                    raise PlannerValidationError("Un contador de exclusiones repite su grupo.")
                seen.add(key)
                matches = grouped.get(tuple(group[name] for name in names), [])
                if len(matches) != 1 or (item.policy == "preserve" and item.null_rows > 0 and matches[0][item.output_name] is not None):
                    raise PlannerValidationError("El resultado no conserva su grupo o su política de nulos.")
        if calculations and any(total != record.rows_processed - record.rows_filtered for total in totals.values()):
            raise PlannerValidationError("Faltan exclusiones o se repiten observaciones en sus contadores.")
