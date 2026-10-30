"""Planificación semántica con validaciones determinísticas y catálogo vigente."""

import asyncio
import logging
import re
from datetime import datetime, timezone

from aquai.application.contracts.planning import (
    ColumnRef, OrderingSpec, PlanningInput, PlanningResult, ResultRef, SortPlanStep, TemporalTextColumn,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import DatasetRetrievalResult
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import LLMProviderError, PlannerValidationError
from aquai.application.interfaces.planning import PlannerProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.planning.context import build_planning_context, latest_requested
from aquai.application.planning.filterValues import resolve_filter_values
from aquai.application.planning.temporalText import profile_temporal_text
from aquai.application.planning.validation import validate_operations, validate_selections, validate_temporal_scope


_logger = logging.getLogger(__name__)
# Sólo estas incidencias requieren una decisión del usuario; el resto son defectos del plan.
_USER_DECISIONS = {"ambiguous_filter_value"}
_MAX_ATTEMPTS = 2


class PlanQueryHandler:
    def __init__(self, provider: PlannerProvider, queries: SourceQueries) -> None:
        self._provider = provider
        self._queries = queries

    async def handle(self, question: str, retrieval: DatasetRetrievalResult, *, original_question: str | None = None,
                     data_mentions=(), request_history=(), source_mentions=()) -> PlanningResult:
        question = PlanningInput(question=question).question
        # Las menciones que el Retriever identificó como columnas no son filtros obligatorios.
        data_mentions = [item for item in data_mentions
                         if not (item.kind == "value" and item.text in retrieval.dimension_mentions)]
        data_mentions += [DataMention(text=value, kind="value") for value in retrieval.confirmed_values]
        if question != retrieval.question.strip():
            raise PlannerValidationError("La pregunta no coincide con la utilizada por el Retriever.")
        versions = {item.dataset_id: item.verified_version for item in retrieval.candidates}
        verified = await asyncio.to_thread(self._queries.verified_datasets, list(versions), versions=versions) if versions else []
        by_id = {item.dataset_id: item for item in verified}
        # Mantiene el orden recuperado sin trasladar scores al LLM.
        catalog = [by_id[item.dataset_id] for item in retrieval.candidates if item.dataset_id in by_id]
        require_latest = latest_requested(original_question or question)
        # Un seguimiento («¿Y de qué año es?») hereda la selección temporal sin exigirla.
        temporal = require_latest or any(latest_requested(item) for item in request_history)
        excluded, periods = [], []
        if temporal:
            for item in catalog:
                if any(column.sql_type in {"DATE", "TIMESTAMP"} for column in item.columns_metadata):
                    continue
                found = await self._temporal_text_columns(item)
                if found:
                    periods.extend(found)
                else:
                    excluded.append(str(item.dataset_id))
            if require_latest:
                catalog = [item for item in catalog if str(item.dataset_id) not in excluded]
            else:
                excluded = []
        evidence = [item for item in retrieval.evidence if item.dataset_id in {dataset.dataset_id for dataset in catalog}]
        context = build_planning_context(question, catalog, self._provider.context_budget_bytes,
                                         temporal_candidates_excluded=excluded, require_latest=require_latest,
                                         query_mentions=data_mentions, temporal_text_columns=periods,
                                         dataset_evidence=evidence, unmatched_values=retrieval.unmatched_values,
                                         source_mentions=source_mentions)
        common = {"context": context, "verified_at": datetime.now(timezone.utc), "temporal_text_columns": periods}
        if not catalog and excluded:
            return PlanningResult(
                **common, status="insufficient_coverage", llm_status=None,
                reason="Los candidatos recuperados carecen de fechas DATE/TIMESTAMP o de períodos TEXT interpretables; no se elige un registro arbitrario.",
                missing_information=["Una columna de fecha o período interpretable para determinar la medición más reciente."],
                warnings=list(retrieval.warnings),
            )
        if not catalog:
            return PlanningResult(
                **common, status="insufficient_coverage", llm_status=None,
                reason="No hay candidatos disponibles para fundamentar una planificación.",
                missing_information=["Datasets recuperados disponibles y vigentes."],
                warnings=[*retrieval.warnings, "El alcance es el conjunto recuperado; no se evaluó todo el catálogo."],
            )
        if context.size_bytes > context.budget_bytes:
            return PlanningResult(
                **common, status="needs_clarification", llm_status=None,
                reason="El catálogo completo de columnas supera el presupuesto del modelo; no se descartaron columnas para forzar un plan.",
                clarification_question="La consulta abarca demasiadas fuentes a la vez. ¿Podés indicar qué variable o qué fuente querés analizar?",
                missing_information=["Un alcance que permita enviar todos sus metadatos al Planner."],
            )
        feedback, attempt, model, tokens = [], 0, None, [None, None]
        while True:
            attempt += 1
            if feedback:
                context.evidence["previous_attempt_feedback"] = feedback
            try:
                reply = await self._provider.propose(context)
                context.sent_to_model = True
                model, tokens = reply.model, [reply.input_tokens, reply.output_tokens]
                decision, selected, operations, issues, resolutions = await self._verify(
                    reply.decision, catalog, data_mentions, evidence, periods, require_latest, source_mentions)
            except (LLMProviderError, PlannerValidationError) as error:
                context.sent_to_model = True
                _logger.warning("Planner intento %s rechazado: %s", attempt, error)
                if attempt < _MAX_ATTEMPTS:
                    feedback = [f"El plan anterior fue rechazado: {error} Corregilo respetando la gramática."]
                    continue
                return PlanningResult(
                    **common, status="failed", llm_status=None, attempts=attempt, model=model,
                    reason=f"El modelo no produjo un plan verificable en {attempt} intentos: {error}",
                    missing_information=[], warnings=list(retrieval.warnings),
                )
            defects = [issue for issue in issues if issue.severity == "clarification" and issue.code not in _USER_DECISIONS]
            if defects and attempt < _MAX_ATTEMPTS:
                _logger.warning("Planner intento %s con defectos: %s", attempt, [issue.message for issue in defects])
                feedback = [issue.message for issue in defects]
                continue
            break
        context.evidence.pop("previous_attempt_feedback", None)
        decisions = [issue.message for issue in issues if issue.severity == "clarification" and issue.code in _USER_DECISIONS]
        defects = [issue.message for issue in issues if issue.severity == "clarification" and issue.code not in _USER_DECISIONS]
        status = "failed" if defects else "needs_clarification" if decisions else decision.status
        absent = [] if not defects or not retrieval.unmatched_values or any(
            issue.code not in {"filter_value_not_found", "missing_requested_filter"}
            for issue in issues if issue.severity == "clarification") else retrieval.unmatched_values
        if absent:
            # El plan no pudo acotarse porque el valor pedido no existe en los registros.
            status, defects = "insufficient_coverage", []
            decision = decision.model_copy(update={"missing_information": [
                f"No hay registros de «{value}» en los datasets consultados." for value in absent]})
        warnings = list(dict.fromkeys([
            "La cobertura se evaluó sólo sobre los candidatos recuperados, no sobre todo el catálogo.",
            *([f"No se evaluó recencia en {len(excluded)} candidatos sin fechas ni períodos interpretables."] if excluded else []),
            *(f"La columna {item.column} guarda fechas como texto; se interpretaron como períodos explícitos ({', '.join(item.precisions)})."
              for item in periods if any(step.operation == "latest" for step in operations)),
            *decision.warnings, *(warning for item in selected for warning in item.warnings),
            *(issue.message for issue in issues if issue.severity == "warning"),
        ]))
        if defects:
            reason = "La propuesta no superó las verificaciones determinísticas: " + " ".join(defects)
        elif decisions:
            reason = "La consulta coincide con varios valores almacenados; hace falta que el usuario elija."
        elif status == "plannable":
            reason = f"Propuesta de {len(selected)} datasets y {len(operations)} operaciones contrastada con el catálogo; la relevancia fue evaluada por el modelo."
        elif status == "needs_clarification":
            reason = "El modelo requiere aclaración para completar el plan."
        else:
            reason = "El modelo no encontró cobertura suficiente entre los candidatos recuperados."
        return PlanningResult(
            status=status, llm_status=decision.status, attempts=attempt,
            reason=reason, llm_reason=decision.reason,
            selected_datasets=selected, operations=operations if status == "plannable" else [],
            output_steps=decision.output_steps if status == "plannable" else [],
            required_tools=sorted({step.tool for step in operations}) if status == "plannable" else [],
            missing_information=[] if defects else [*decision.missing_information],
            clarification_question=(" ".join(decisions) if decisions else
                                    None if defects else decision.clarification_question),
            warnings=warnings, validation_issues=issues, context=context,
            filter_value_resolutions=resolutions, temporal_text_columns=periods,
            model=model, input_tokens=tokens[0], output_tokens=tokens[1],
            verified_at=datetime.now(timezone.utc),
        )

    async def _verify(self, decision, catalog, data_mentions, evidence, periods, require_latest, source_mentions=()):
        decision = self._latest_in_python(self._extremes_as_sort(self._result_references(self._publication_steps(
            self._dependency_order(self._merged_selections(decision)))), catalog), periods)
        selected = validate_selections(decision, catalog)
        operations, issues = validate_operations(decision, selected, periods)
        resolutions = []
        if decision.status == "plannable" and not any(issue.severity == "clarification" for issue in issues):
            decision, resolutions, value_issues = await resolve_filter_values(
                decision, selected, operations, data_mentions, self._queries, evidence, source_mentions)
            if resolutions or any(issue.code == "source_name_filter_removed" for issue in value_issues):
                operations, issues = validate_operations(decision, selected, periods)
            issues.extend(value_issues)
        if decision.status == "plannable":
            issues.extend(validate_temporal_scope(operations, decision.output_steps, require_latest=require_latest,
                requested_values=[mention.text for mention in data_mentions if mention.kind == "value"]))
        if selected:
            current = await asyncio.to_thread(
                self._queries.verified_datasets, [item.dataset_id for item in selected],
                versions={item.dataset_id: item.verified_version for item in selected},
            )
            current_by_id = {item.dataset_id: item for item in current}
            old_by_id = {item.dataset_id: item for item in catalog}
            for item in selected:
                latest = current_by_id.get(item.dataset_id)
                if latest is None or latest.model_dump() != old_by_id[item.dataset_id].model_dump():
                    issues.append(ValidationIssue(code="catalog_changed", severity="clarification",
                        message="El catálogo elegido cambió durante la inferencia; hay que recuperar y planificar de nuevo."))
        return decision, selected, operations, issues, resolutions

    @staticmethod
    def _merged_selections(decision):
        """Un dataset seleccionado dos veces se une en una selección con todas sus columnas."""
        ids = [item.dataset_id for item in decision.datasets]
        if len(ids) == len(set(ids)):
            return decision
        changed = decision.model_copy(deep=True)
        merged = {}
        for item in changed.datasets:
            if item.dataset_id in merged and merged[item.dataset_id].version == item.version:
                merged[item.dataset_id].columns = list(dict.fromkeys([*merged[item.dataset_id].columns, *item.columns]))
            elif item.dataset_id not in merged:
                merged[item.dataset_id] = item
            else:
                return decision  # Versiones distintas: la validación lo rechaza.
        changed.datasets = list(merged.values())
        return changed

    @staticmethod
    def _result_references(decision):
        """Una columna citada después del read es el campo homónimo que ese read expone."""
        steps = {step.id: step for step in decision.steps}
        changed = decision.model_copy(deep=True)
        modified = False
        for step in changed.steps:
            if step.operation == "read" or len(step.depends_on) != 1 or step.depends_on[0] not in steps:
                continue
            parent = steps[step.depends_on[0]]
            origin = parent
            while origin.operation in {"sort", "latest"} and origin.depends_on and origin.depends_on[0] in steps:
                origin = steps[origin.depends_on[0]]  # sort y latest conservan los campos del read.
            if origin.operation != "read":
                continue

            def convert(ref):
                nonlocal modified
                if isinstance(ref, ColumnRef) and ref in origin.columns:
                    modified = True
                    return ResultRef(kind="result", step_id=parent.id, field=ref.column)
                return ref

            for item in [*step.filters, *step.group_by, *step.order_by]:
                item.field = convert(item.field)
            for calculation in step.calculations:
                calculation.inputs = [convert(ref) for ref in calculation.inputs]
            if step.visualization is not None:
                for name in ("x", "y", "series"):
                    if getattr(step.visualization, name, None) is not None:
                        setattr(step.visualization, name, convert(getattr(step.visualization, name)))
        return changed if modified else decision

    @staticmethod
    def _extremes_as_sort(decision, catalog):
        """latest sobre una variable numérica pide su máximo: equivale a ordenar y conservar empates."""
        if decision.status != "plannable":
            return decision
        types = {(dataset.dataset_id, column.sql_name): column.sql_type
                 for dataset in catalog for column in dataset.columns_metadata}
        steps = {step.id: step for step in decision.steps}
        changed, modified = decision.model_copy(deep=True), False
        for index, step in enumerate(changed.steps):
            if step.operation != "latest" or not step.order_by or not step.depends_on:
                continue
            origin = steps.get(step.depends_on[0])
            while origin is not None and origin.operation in {"sort", "latest"} and origin.depends_on:
                origin = steps.get(origin.depends_on[0])
            if origin is None or origin.operation != "read":
                continue
            field = step.order_by[0].field.field
            kinds = {types.get((ref.dataset_id, ref.column)) for ref in origin.columns if ref.column == field}
            if kinds and kinds <= {"NUMERIC", "BIGINT"}:
                changed.steps[index] = SortPlanStep(
                    id=step.id, tool=step.tool, operation="sort", depends_on=list(step.depends_on),
                    filters=step.filters, order_by=[OrderingSpec(field=step.order_by[0].field, direction="desc")],
                    limit=1, purpose=step.purpose)
                modified = True
        return changed if modified else decision

    @staticmethod
    def _publication_steps(decision):
        """Pasos llamados como el campo output_steps sólo republican su dependencia.

        El modelo a veces confunde la lista de salidas con un paso; se publica directamente
        la dependencia y se elimina el paso, siempre que ningún otro paso lo use.
        """
        if decision.status != "plannable":
            return decision
        used = {key for step in decision.steps for key in step.depends_on}
        replace = {step.id: step.depends_on[0] for step in decision.steps
                   if re.fullmatch(r"output_steps?(?:_[a-z0-9]+)?", step.id) and len(step.depends_on) == 1
                   and step.id not in used and step.operation in {"aggregate", "calculate", "sort"}}
        if not replace:
            return decision
        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        changed.output_steps = list(dict.fromkeys(replace.get(key, key) for key in changed.output_steps))
        return changed

    @staticmethod
    def _dependency_order(decision):
        """Ordena los pasos por dependencias sin alterarlos; ciclos e ids inexistentes se rechazan luego."""
        steps = {step.id: step for step in decision.steps}
        if len(steps) != len(decision.steps) or any(
                key not in steps for step in decision.steps for key in step.depends_on):
            return decision
        ordered, done = [], set()
        while len(ordered) < len(steps):
            ready = [step for step in decision.steps if step.id not in done and set(step.depends_on) <= done]
            if not ready:
                return decision
            ordered.extend(ready)
            done.update(step.id for step in ready)
        if [step.id for step in ordered] == [step.id for step in decision.steps]:
            return decision
        changed = decision.model_copy(deep=True)
        changed.steps = [next(item for item in changed.steps if item.id == step.id) for step in ordered]
        return changed

    @staticmethod
    def _latest_in_python(decision, periods):
        """Un latest sobre períodos TEXT se ejecuta en Python, que interpreta cada valor."""
        columns = {(item.dataset_id, item.column) for item in periods}
        if not columns or decision.status != "plannable":
            return decision
        steps = {step.id: step for step in decision.steps}
        changed = decision.model_copy(deep=True)
        for step in changed.steps:
            if step.operation != "latest" or step.tool == "python" or not step.order_by:
                continue
            ref, parent = step.order_by[0].field, steps.get(step.depends_on[0]) if step.depends_on else None
            while parent is not None and parent.operation != "read":
                parent = steps.get(parent.depends_on[0]) if parent.depends_on else None
            if parent is not None and any((column.dataset_id, column.column) in columns and column.column == ref.field
                                          for column in parent.columns):
                step.tool = "python"
        return changed

    async def _temporal_text_columns(self, dataset):
        names = [column.sql_name for column in dataset.columns_metadata if column.sql_type == "TEXT"]
        if not names:
            return []
        try:
            values = await asyncio.to_thread(self._queries.distinct_text_values, dataset, names)
        except Exception as error:
            _logger.warning("No se pudieron perfilar fechas TEXT: %s", type(error).__name__)
            return []
        found = []
        for name, items in values.items():
            profile = profile_temporal_text(items)
            if profile:
                found.append(TemporalTextColumn(dataset_id=dataset.dataset_id, column=name, **profile))
        return found
