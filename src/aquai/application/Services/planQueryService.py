import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import NamedTuple

from aquai.application.contracts.planning import (
    ColumnRef,
    GroupingSpec,
    OrderingSpec,
    PlanningInput,
    PlanningResult,
    ResultRef,
    SortPlanStep,
    TemporalTextColumn,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import DatasetRetrievalResult
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import LLMProviderError, PlannerValidationError
from aquai.application.interfaces.planning import PlannerProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.planning.context import build_planning_context, latest_requested
from aquai.application.planning.filterValues import resolve_filter_values
from aquai.application.planning.limits import (
    conserved_limits,
    countable_mentions,
    requested_limit,
    result_limits,
)
from aquai.application.planning.temporalText import profile_temporal_text
from aquai.application.planning.validation import (
    validate_operations,
    validate_selections,
    validate_temporal_scope,
)

_logger = logging.getLogger(__name__)
# Sólo estas incidencias requieren una decisión del usuario; el resto son defectos del plan.
_USER_DECISIONS = {"ambiguous_filter_value"}
_MAX_ATTEMPTS = 2
_CATEGORY_LENGTH = 80
# Errores frecuentes del modelo y cómo corregirlos; el texto de la validación sólo describe el síntoma.
_HINTS = {
    "resultado inexistente": "Cada paso expone sólo las salidas de su dependencia: para un valor por categoría, "
    "agregá esa columna como group_by del aggregate y usá su output_name después.",
    "columna inexistente": "Usá sólo sql_name que figuren en column_catalogs para ese mismo dataset; no elijas "
    "datasets que no tengan la variable pedida (revisá variable_columns).",
}


class _VerifiedPlan(NamedTuple):
    decision: object
    selected: list
    operations: list
    issues: list
    resolutions: list


@dataclass
class _Proposal:
    attempts: int
    model: str | None
    tokens: list
    verified: _VerifiedPlan | None = None
    error: Exception | None = None


def _passthrough_origin(step, steps):
    """sort y latest conservan los campos de su entrada: el paso que los produjo es el primero que no lo es."""
    while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
        step = steps.get(step.depends_on[0])
    return step


def _column_types(catalog):
    return {
        (dataset.dataset_id, column.sql_name): column.sql_type
        for dataset in catalog
        for column in dataset.columns_metadata
    }


class PlanQueryService:
    def __init__(self, provider: PlannerProvider, queries: SourceQueries) -> None:
        self._provider = provider
        self._queries = queries

    async def plan(
        self,
        question: str,
        retrieval: DatasetRetrievalResult,
        *,
        original_question: str | None = None,
        data_mentions=(),
        request_history=(),
        source_mentions=(),
        previous_analysis=None,
    ) -> PlanningResult:
        question = PlanningInput(question=question).question
        data_mentions = self._planning_mentions(data_mentions, retrieval)
        limits = requested_limit(data_mentions), conserved_limits(data_mentions, previous_analysis)
        if question != retrieval.question.strip():
            raise PlannerValidationError(
                "La pregunta no coincide con la utilizada por el Retriever."
            )

        catalog = await self._verified_catalog(retrieval.candidates)
        require_latest = latest_requested(original_question or question)
        # Un seguimiento («¿Y de qué año es?») hereda la selección temporal sin exigirla.
        temporal = require_latest or any(latest_requested(item) for item in request_history)
        catalog, excluded, periods = await self._temporal_scope(catalog, temporal, require_latest)
        evidence = await self._with_categories(
            self._catalog_evidence(retrieval.evidence, catalog), catalog
        )

        context = build_planning_context(
            question,
            catalog,
            self._provider.context_budget_bytes,
            temporal_candidates_excluded=excluded,
            require_latest=require_latest,
            query_mentions=countable_mentions(data_mentions),
            temporal_text_columns=periods,
            dataset_evidence=evidence,
            unmatched_values=retrieval.unmatched_values,
            source_mentions=source_mentions,
        )
        common = {
            "context": context,
            "verified_at": datetime.now(timezone.utc),
            "temporal_text_columns": periods,
        }

        if not catalog and excluded:
            return self._without_dated_candidates(common, retrieval.warnings)
        if not catalog:
            return self._without_candidates(common, retrieval.warnings)
        if context.size_bytes > context.budget_bytes:
            return self._over_budget(common)

        verify = partial(
            self._verify,
            catalog=catalog,
            data_mentions=data_mentions,
            evidence=evidence,
            periods=periods,
            require_latest=require_latest,
            source_mentions=source_mentions,
            limits=limits,
        )
        proposal = await self._propose(context, verify)
        if proposal.error is not None:
            return self._rejected_result(common, proposal, retrieval.warnings)

        context.evidence.pop("previous_attempt_feedback", None)
        return self._planning_result(
            proposal, retrieval.unmatched_values, context, excluded, periods
        )

    @staticmethod
    def _planning_mentions(data_mentions, retrieval):
        # Las menciones que el Retriever identificó como columnas no son filtros obligatorios.
        mentions = [
            item
            for item in data_mentions
            if not (item.kind == "value" and item.text in retrieval.dimension_mentions)
        ]
        mentions += [DataMention(text=value, kind="value") for value in retrieval.confirmed_values]
        return mentions

    async def _verified_catalog(self, candidates):
        versions = {item.dataset_id: item.verified_version for item in candidates}
        if not versions:
            return []

        verified = await asyncio.to_thread(
            self._queries.verified_datasets, list(versions), versions=versions
        )
        by_id = {item.dataset_id: item for item in verified}
        # Mantiene el orden recuperado sin trasladar scores al LLM.
        return [by_id[item.dataset_id] for item in candidates if item.dataset_id in by_id]

    async def _temporal_scope(self, catalog, temporal, require_latest):
        if not temporal:
            return catalog, [], []

        excluded, periods = await self._text_periods(catalog)
        if not require_latest:
            return catalog, [], periods

        dated = [item for item in catalog if str(item.dataset_id) not in excluded]
        return dated, excluded, periods

    async def _text_periods(self, catalog):
        excluded, periods = [], []
        for item in catalog:
            if any(column.sql_type in {"DATE", "TIMESTAMP"} for column in item.columns_metadata):
                continue

            found = await self._temporal_text_columns(item)
            if found:
                periods.extend(found)
            else:
                excluded.append(str(item.dataset_id))
        return excluded, periods

    @staticmethod
    def _catalog_evidence(evidence, catalog):
        catalog_ids = {dataset.dataset_id for dataset in catalog}
        return [item for item in evidence if item.dataset_id in catalog_ids]

    @staticmethod
    def _without_dated_candidates(common, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="Los candidatos recuperados carecen de fechas DATE/TIMESTAMP o de períodos TEXT interpretables; no se elige un registro arbitrario.",
            missing_information=[
                "Una columna de fecha o período interpretable para determinar la medición más reciente."
            ],
            warnings=list(retrieval_warnings),
        )

    @staticmethod
    def _without_candidates(common, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="No hay candidatos disponibles para fundamentar una planificación.",
            missing_information=["Datasets recuperados disponibles y vigentes."],
            warnings=[
                *retrieval_warnings,
                "El alcance es el conjunto recuperado; no se evaluó todo el catálogo.",
            ],
        )

    @staticmethod
    def _over_budget(common):
        return PlanningResult(
            **common,
            status="needs_clarification",
            llm_status=None,
            reason="El catálogo completo de columnas supera el presupuesto del modelo; no se descartaron columnas para forzar un plan.",
            clarification_question="La consulta abarca demasiadas fuentes a la vez. ¿Podés indicar qué variable o qué fuente querés analizar?",
            missing_information=["Un alcance que permita enviar todos sus metadatos al Planner."],
        )

    async def _propose(self, context, verify):
        feedback, attempt, model, tokens = [], 0, None, [None, None]
        while True:
            attempt += 1
            if feedback:
                context.evidence["previous_attempt_feedback"] = feedback

            try:
                reply = await self._provider.propose(context)
                context.sent_to_model = True
                model, tokens = reply.model, [reply.input_tokens, reply.output_tokens]
                verified = await verify(reply.decision)
            except (LLMProviderError, PlannerValidationError) as error:
                context.sent_to_model = True
                _logger.warning("Planner intento %s rechazado: %s", attempt, error)
                if attempt < _MAX_ATTEMPTS:
                    feedback = self._rejection_feedback(error)
                    continue
                return _Proposal(attempt, model, tokens, error=error)

            defects = self._plan_defects(verified.issues)
            if not defects or attempt >= _MAX_ATTEMPTS:
                return _Proposal(attempt, model, tokens, verified=verified)

            _logger.warning(
                "Planner intento %s con defectos: %s",
                attempt,
                [issue.message for issue in defects],
            )
            feedback = [issue.message for issue in defects]

    @staticmethod
    def _rejection_feedback(error):
        hints = [hint for symptom, hint in _HINTS.items() if symptom in str(error)]
        return [
            f"El plan anterior fue rechazado: {error} Corregilo respetando la gramática.",
            *hints,
        ]

    @staticmethod
    def _plan_defects(issues):
        return [
            issue
            for issue in issues
            if issue.severity == "clarification" and issue.code not in _USER_DECISIONS
        ]

    @staticmethod
    def _user_decisions(issues):
        return [
            issue
            for issue in issues
            if issue.severity == "clarification" and issue.code in _USER_DECISIONS
        ]

    @staticmethod
    def _rejected_result(common, proposal, retrieval_warnings):
        attempt, error = proposal.attempts, proposal.error
        return PlanningResult(
            **common,
            status="failed",
            llm_status=None,
            attempts=attempt,
            model=proposal.model,
            reason=f"El modelo no produjo un plan verificable en {attempt} intentos: {error}",
            missing_information=[],
            warnings=list(retrieval_warnings),
        )

    def _planning_result(self, proposal, unmatched_values, context, excluded, periods):
        decision, selected, operations, issues, resolutions = proposal.verified
        decisions = [issue.message for issue in self._user_decisions(issues)]
        defects = [issue.message for issue in self._plan_defects(issues)]
        if defects:
            _logger.warning(
                "Planner intento %s con defectos finales: %s", proposal.attempts, defects
            )
        status = self._final_status(defects, decisions, decision)

        absent = self._absent_values(issues, defects, unmatched_values)
        if absent:
            status, defects = "insufficient_coverage", []
            decision = decision.model_copy(
                update={
                    "missing_information": [
                        f"No hay registros de {value} en los datos consultados." for value in absent
                    ]
                }
            )

        warnings = self._result_warnings(excluded, periods, operations, decision, selected, issues)
        reason = self._result_reason(defects, decisions, status, selected, operations)
        plannable = status == "plannable"
        return PlanningResult(
            status=status,
            llm_status=decision.status,
            attempts=proposal.attempts,
            reason=reason,
            llm_reason=decision.reason,
            selected_datasets=selected,
            operations=operations if plannable else [],
            output_steps=decision.output_steps if plannable else [],
            required_tools=sorted({step.tool for step in operations}) if plannable else [],
            missing_information=[] if defects else [*decision.missing_information],
            clarification_question=self._clarification_question(decisions, defects, decision),
            warnings=warnings,
            validation_issues=issues,
            context=context,
            filter_value_resolutions=resolutions,
            temporal_text_columns=periods,
            model=proposal.model,
            input_tokens=proposal.tokens[0],
            output_tokens=proposal.tokens[1],
            verified_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _final_status(defects, decisions, decision):
        if defects:
            return "failed"
        if decisions:
            return "needs_clarification"
        return decision.status

    @staticmethod
    def _absent_values(issues, defects, unmatched_values):
        """Valores pedidos inexistentes en los registros, si son lo único que impidió acotar el plan."""
        if not defects or not unmatched_values:
            return []

        blocking = [issue.code for issue in issues if issue.severity == "clarification"]
        if any(
            code not in {"filter_value_not_found", "missing_requested_filter"} for code in blocking
        ):
            return []
        return unmatched_values

    @staticmethod
    def _result_warnings(excluded, periods, operations, decision, selected, issues):
        warnings = [
            "La cobertura se evaluó sólo sobre los candidatos recuperados, no sobre todo el catálogo."
        ]
        if excluded:
            warnings.append(
                f"No se evaluó recencia en {len(excluded)} candidatos sin fechas ni períodos interpretables."
            )
        if any(step.operation == "latest" for step in operations):
            warnings.extend(
                f"La columna {item.column} guarda fechas como texto; se interpretaron como períodos explícitos ({', '.join(item.precisions)})."
                for item in periods
            )
        warnings.extend(decision.warnings)
        warnings.extend(warning for item in selected for warning in item.warnings)
        warnings.extend(issue.message for issue in issues if issue.severity == "warning")
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _result_reason(defects, decisions, status, selected, operations):
        if defects:
            return "La propuesta no superó las verificaciones determinísticas: " + " ".join(defects)
        if decisions:
            return "La consulta coincide con varios valores almacenados; hace falta que el usuario elija."
        if status == "plannable":
            return f"Propuesta de {len(selected)} datasets y {len(operations)} operaciones contrastada con el catálogo; la relevancia fue evaluada por el modelo."
        if status == "needs_clarification":
            return "El modelo requiere aclaración para completar el plan."
        return "El modelo no encontró cobertura suficiente entre los candidatos recuperados."

    @staticmethod
    def _clarification_question(decisions, defects, decision):
        if decisions:
            return " ".join(decisions)
        if defects:
            return None
        return decision.clarification_question

    async def _verify(
        self,
        decision,
        catalog,
        data_mentions,
        evidence,
        periods,
        require_latest,
        source_mentions=(),
        limits=(None, {}),
    ):
        decision = self._repaired(decision, catalog, periods)
        selected = validate_selections(decision, catalog)
        operations, issues = validate_operations(decision, selected, periods)

        limited = self._result_limits(decision, operations, *limits)
        if limited is not decision:
            decision = limited
            operations, issues = validate_operations(decision, selected, periods)

        resolutions = []
        blocked = any(issue.severity == "clarification" for issue in issues)
        if decision.status == "plannable" and not blocked:
            decision, resolutions, value_issues = await resolve_filter_values(
                decision,
                selected,
                operations,
                data_mentions,
                self._queries,
                evidence,
                source_mentions,
            )
            if resolutions or any(
                issue.code == "source_name_filter_removed" for issue in value_issues
            ):
                operations, issues = validate_operations(decision, selected, periods)
            issues.extend(value_issues)

        if decision.status == "plannable":
            requested_values = [
                mention.text for mention in data_mentions if mention.kind == "value"
            ]
            issues.extend(
                validate_temporal_scope(
                    operations,
                    decision.output_steps,
                    require_latest=require_latest,
                    requested_values=requested_values,
                )
            )

        issues.extend(await self._catalog_changes(selected, catalog))
        return _VerifiedPlan(decision, selected, operations, issues, resolutions)

    @staticmethod
    def _repaired(decision, catalog, periods):
        decision = PlanQueryService._merged_selections(decision)
        decision = PlanQueryService._dependency_order(decision)
        decision = PlanQueryService._publication_steps(decision)
        decision = PlanQueryService._unpublished_branches(decision)
        decision = PlanQueryService._result_references(decision)
        decision = PlanQueryService._missing_groupings(decision, catalog)
        decision = PlanQueryService._extremes_as_sort(decision, catalog)
        return PlanQueryService._latest_in_python(decision, periods)

    async def _catalog_changes(self, selected, catalog):
        if not selected:
            return []

        current = await asyncio.to_thread(
            self._queries.verified_datasets,
            [item.dataset_id for item in selected],
            versions={item.dataset_id: item.verified_version for item in selected},
        )
        current_by_id = {item.dataset_id: item for item in current}
        old_by_id = {item.dataset_id: item for item in catalog}
        changes = []
        for item in selected:
            latest = current_by_id.get(item.dataset_id)
            if latest is None or latest.model_dump() != old_by_id[item.dataset_id].model_dump():
                changes.append(
                    ValidationIssue(
                        code="catalog_changed",
                        severity="clarification",
                        message="El catálogo elegido cambió durante la inferencia; hay que recuperar y planificar de nuevo.",
                    )
                )
        return changes

    async def _with_categories(self, evidence, catalog):
        datasets = {item.dataset_id: item for item in catalog}
        enriched = []
        for item in evidence:
            categories = await self._category_values(item, datasets)
            enriched.append(
                item.model_copy(update={"category_values": categories}) if categories else item
            )
        return enriched

    async def _category_values(self, item, datasets):
        measured = next((names[0] for names in item.variable_matches.values() if names), None)
        columns = dict.fromkeys(
            match.column for matches in item.value_matches.values() for match in matches
        )
        selected = list(columns)[:3]
        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self._queries.frequent_text_values,
                    datasets[item.dataset_id],
                    column,
                    measured=measured,
                )
                for column in selected
            ),
            return_exceptions=True,
        )

        categories = {}
        for column, rows in zip(selected, results):
            if isinstance(
                rows, Exception
            ):  # La planificación sigue con la evidencia de recuperación.
                _logger.warning(
                    "No se pudieron leer categorías de %s: %s", column, type(rows).__name__
                )
                continue

            # Una columna con un único valor no ofrece otras categorías; los textos largos no son categorías.
            values = [row["value"] for row in rows if len(row["value"]) <= _CATEGORY_LENGTH]
            if len(values) > 1:
                categories[column] = values
        return categories

    @staticmethod
    def _merged_selections(decision):
        ids = [item.dataset_id for item in decision.datasets]
        if len(ids) == len(set(ids)):
            return decision

        changed = decision.model_copy(deep=True)
        merged = {}
        for item in changed.datasets:
            previous = merged.get(item.dataset_id)
            if previous is None:
                merged[item.dataset_id] = item
                continue
            if previous.version != item.version:
                return decision  # Versiones distintas: la validación lo rechaza.
            previous.columns = list(dict.fromkeys([*previous.columns, *item.columns]))
        changed.datasets = list(merged.values())
        return changed

    @staticmethod
    def _result_references(decision):
        """Una columna citada después del read es el campo homónimo que ese read expone."""
        steps = {step.id: step for step in decision.steps}
        changed = decision.model_copy(deep=True)
        modified = False
        for step in changed.steps:
            if (
                step.operation == "read"
                or len(step.depends_on) != 1
                or step.depends_on[0] not in steps
            ):
                continue

            parent = steps[step.depends_on[0]]
            origin = parent
            while (
                origin.operation in {"sort", "latest"}
                and origin.depends_on
                and origin.depends_on[0] in steps
            ):
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
                        setattr(
                            step.visualization, name, convert(getattr(step.visualization, name))
                        )
        return changed if modified else decision

    @staticmethod
    def _unpublished_branches(decision):
        """Un paso final que continúa un resultado publicado (su gráfico, su orden) y quedó fuera de output_steps.

        Sólo se completa esa continuación: una rama auxiliar no publicada puede no tener el alcance pedido."""
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        consumed = {key for step in decision.steps for key in step.depends_on}
        published = set(decision.output_steps)
        forgotten = [
            step.id
            for step in decision.steps
            if step.id not in consumed
            and step.id not in decision.output_steps
            and PlanQueryService._ancestors(step, steps) & published
        ]
        if not forgotten:
            return decision
        return decision.model_copy(update={"output_steps": [*decision.output_steps, *forgotten]})

    @staticmethod
    def _ancestors(step, steps):
        pending, found = list(step.depends_on), set()
        while pending:
            key = pending.pop()
            if key in steps and key not in found:
                found.add(key)
                pending.extend(steps[key].depends_on)
        return found

    @staticmethod
    def _missing_groupings(decision, catalog):
        """Una categoría de texto por la que se ordena o grafica después de un aggregate sin agrupar es el group_by
        que el modelo omitió: un valor por categoría es la única lectura coherente de ese plan.

        No se repara si la columna es numérica o temporal, alimenta el cálculo o ya está agrupada con otro nombre."""
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        types = _column_types(catalog)
        modified = False
        for step in changed.steps:
            refs = [item.field for item in step.order_by]
            if step.visualization is not None:
                refs += [getattr(step.visualization, name, None) for name in ("x", "series")]

            for ref in refs:
                target = PlanQueryService._ungrouped_aggregate(ref, steps, types)
                if target is None:
                    continue
                target.group_by.append(
                    GroupingSpec(
                        field=ResultRef(
                            kind="result", step_id=target.depends_on[0], field=ref.field
                        ),
                        output_name=ref.field,
                    )
                )
                modified = True
        return changed if modified else decision

    @staticmethod
    def _ungrouped_aggregate(ref, steps, types):
        target = (
            _passthrough_origin(steps.get(ref.step_id), steps)
            if isinstance(ref, ResultRef)
            else None
        )
        if target is None or target.operation != "aggregate" or len(target.depends_on) != 1:
            return None

        read = _passthrough_origin(steps.get(target.depends_on[0]), steps)
        column = (
            next((item for item in read.columns if item.column == ref.field), None)
            if read and read.operation == "read"
            else None
        )
        if column is None or types.get((column.dataset_id, column.column)) != "TEXT":
            return None

        grouped = {
            item.field.field for item in target.group_by if isinstance(item.field, ResultRef)
        }
        inputs = {
            item.field
            for calculation in target.calculations
            for item in calculation.inputs
            if isinstance(item, ResultRef)
        }
        exposed = {item.output_name for item in [*target.group_by, *target.calculations]}
        if ref.field in exposed | grouped | inputs:
            return None
        return target

    @staticmethod
    def _extremes_as_sort(decision, catalog):
        """latest sobre una variable numérica pide su máximo: equivale a ordenar y conservar empates.

        Lo mismo un latest tras un sort por una variable numérica («el máximo y en qué fecha»): latest ignora
        el orden de su entrada, así que se pedía el extremo de ese orden y la fecha es sólo un atributo."""
        if decision.status != "plannable":
            return decision

        types = _column_types(catalog)
        steps = {step.id: step for step in decision.steps}
        changed, modified = decision.model_copy(deep=True), False
        for index, step in enumerate(changed.steps):
            if step.operation != "latest" or not step.order_by or not step.depends_on:
                continue

            parent = steps.get(step.depends_on[0])
            origin = _passthrough_origin(parent, steps)
            if origin is None or origin.operation != "read":
                continue

            ordering = PlanQueryService._extreme_ordering(step, parent, origin, types)
            if ordering is None:
                continue
            changed.steps[index] = SortPlanStep(
                id=step.id,
                tool=step.tool,
                operation="sort",
                depends_on=list(step.depends_on),
                filters=step.filters,
                order_by=[ordering],
                limit=1,
                purpose=step.purpose,
            )
            modified = True
        return changed if modified else decision

    @staticmethod
    def _extreme_ordering(step, parent, read, types):
        if PlanQueryService._is_measured(step.order_by[0].field.field, read, types):
            return OrderingSpec(field=step.order_by[0].field, direction="desc")

        ranked = (
            parent.order_by[0].field if parent.operation == "sort" and parent.order_by else None
        )
        if isinstance(ranked, ResultRef) and PlanQueryService._is_measured(
            ranked.field, read, types
        ):
            return OrderingSpec(
                field=ResultRef(kind="result", step_id=parent.id, field=ranked.field),
                direction=parent.order_by[0].direction,
            )
        return None

    @staticmethod
    def _is_measured(field, read, types):
        kinds = {
            types.get((ref.dataset_id, ref.column)) for ref in read.columns if ref.column == field
        }
        return bool(kinds) and kinds <= {"NUMERIC", "BIGINT"}

    @staticmethod
    def _result_limits(decision, operations, requested, conserved):
        """La cantidad pedida («los primeros 5») o la del resultado citado acota las filas publicadas.

        El modelo suele nombrar el paso como top5 sin fijar su limit, o pierde el límite al repetir un plan
        para graficarlo; la cantidad proviene de la consulta o del historial verificado, nunca del modelo.
        Usa los pasos validados: sus campos de salida dicen si cada orden es por un valor medido."""
        if decision.status != "plannable" or (requested is None and not conserved):
            return decision

        targets = result_limits(
            {step.id: step.model_dump(mode="json") for step in operations},
            decision.output_steps,
            requested,
            conserved,
        )
        if not targets:
            return decision

        changed = decision.model_copy(deep=True)
        for step in changed.steps:
            step.limit = targets.get(step.id, step.limit)
        return changed

    @staticmethod
    def _publication_steps(decision):
        """Pasos llamados como el campo output_steps sólo republican su dependencia.

        El modelo a veces confunde la lista de salidas con un paso; se publica directamente
        la dependencia y se elimina el paso, siempre que ningún otro paso lo use.
        """
        if decision.status != "plannable":
            return decision

        used = {key for step in decision.steps for key in step.depends_on}
        replace = {
            step.id: step.depends_on[0]
            for step in decision.steps
            if PlanQueryService._is_publication_step(step, used)
        }
        if not replace:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        changed.output_steps = list(
            dict.fromkeys(replace.get(key, key) for key in changed.output_steps)
        )
        return changed

    @staticmethod
    def _is_publication_step(step, used):
        return bool(
            re.fullmatch(r"output_steps?(?:_[a-z0-9]+)?", step.id)
            and len(step.depends_on) == 1
            and step.id not in used
            and step.operation in {"aggregate", "calculate", "sort"}
        )

    @staticmethod
    def _dependency_order(decision):
        """Ordena los pasos por dependencias sin alterarlos; los ids inexistentes se rechazan luego.

        Ids repetidos y ciclos se rechazan acá: las reparaciones siguientes recorren dependencias y no terminarían."""
        steps = {step.id: step for step in decision.steps}
        if len(steps) != len(decision.steps):
            raise PlannerValidationError("Los identificadores de pasos deben ser únicos.")
        if any(key not in steps for step in decision.steps for key in step.depends_on):
            return decision

        ordered, done = [], set()
        while len(ordered) < len(steps):
            ready = [
                step
                for step in decision.steps
                if step.id not in done and set(step.depends_on) <= done
            ]
            if not ready:
                raise PlannerValidationError(
                    f"El plan contiene un ciclo de dependencias entre los pasos {sorted(set(steps) - done)}."
                )
            ordered.extend(ready)
            done.update(step.id for step in ready)
        if [step.id for step in ordered] == [step.id for step in decision.steps]:
            return decision

        changed = decision.model_copy(deep=True)
        copies = {item.id: item for item in changed.steps}
        changed.steps = [copies[step.id] for step in ordered]
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

            ref = step.order_by[0].field
            read = PlanQueryService._read_ancestor(step, steps)
            if read is not None and any(
                (column.dataset_id, column.column) in columns and column.column == ref.field
                for column in read.columns
            ):
                step.tool = "python"
        return changed

    @staticmethod
    def _read_ancestor(step, steps):
        parent = steps.get(step.depends_on[0]) if step.depends_on else None
        while parent is not None and parent.operation != "read":
            parent = steps.get(parent.depends_on[0]) if parent.depends_on else None
        return parent

    async def _temporal_text_columns(self, dataset):
        names = [
            column.sql_name for column in dataset.columns_metadata if column.sql_type == "TEXT"
        ]
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
                found.append(
                    TemporalTextColumn(dataset_id=dataset.dataset_id, column=name, **profile)
                )
        return found
