import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import NamedTuple

from aquai.application.contracts.planning import (
    MAX_FILTER_VALUES,
    ColumnRef,
    FilterSpec,
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
from aquai.application.planning.context import (
    asked_extremes,
    build_planning_context,
    chart_requested,
    extreme_requested,
    latest_requested,
    numeric_comparison,
    period_phrase,
    requested_periods,
    requested_statistics,
    requested_thresholds,
)
from aquai.application.planning.filterValues import resolve_filter_values
from aquai.application.planning.followUp import (
    conserved_statistics,
    conserved_time_grain,
    period_fields,
    period_span_decision,
    previous_operations,
    repeated_decision,
    requested_time_grain,
    reused_decision,
    stale_values,
    with_time_grain,
    without_stale_filters,
)
from aquai.application.planning.limits import (
    conserved_limits,
    countable_mentions,
    requested_limit,
    result_limits,
)
from aquai.application.planning.temporalText import profile_temporal_text, within_period
from aquai.application.planning.validation import (
    applies_threshold,
    filters_period,
    period_columns,
    period_matches,
    requested_dimensions,
    validate_operations,
    validate_requested_scope,
    validate_selections,
    validate_temporal_scope,
    validate_text_orderings,
    validate_unrequested_latest,
)

_logger = logging.getLogger(__name__)
_USER_DECISIONS = {"ambiguous_filter_value"}
_MAX_ATTEMPTS = 2
_CATEGORY_LENGTH = 80
_EXTREME_ROWS = 5
_EXTREME_SUFFIX = "_extreme"
_CATEGORY_PREFERENCE = ({"TEXT"}, {"DATE", "TIMESTAMP"})
_MAX_PLAN_STEPS = 12
_TEXT_CALCULATIONS = {"count", "count_distinct"}
_TEXT_COMPARISONS = {"text_measurement_order", "ineligible_calculation"}
_PUBLICATION_STEP = re.compile(r"output_steps?(?:_[a-z0-9]+)?")
_MAX_STEP_ID_LENGTH = 63
_DISTINCT_TEXT_LIMIT = 500
_GRAIN_NAMES = {"day": "día", "month": "mes", "year": "año"}
_PERIOD_LITERAL = re.compile(r"(?:1[89]|2\d)\d{2}(?:-\d{1,2}(?:-\d{1,2}(?:[T ][\d:.]+)?)?)?")
_HINTS = {
    "resultado inexistente": "Cada paso expone sólo las salidas de su dependencia: para un valor por categoría, "
    "agregá esa columna como group_by del aggregate y usá su output_name después.",
    "columna inexistente": "Usá sólo sql_name que figuren en column_catalogs para ese mismo dataset; no elijas "
    "datasets que no tengan la variable pedida (revisá variable_columns).",
    "latest requiere una fecha real": "latest sólo selecciona la medición más reciente cuando la consulta la pide; "
    "nunca publica un resultado: poné en output_steps el id del paso que ya responde, sin pasos extra.",
    "dependencia directa": "Cada paso sólo ve los campos de su dependencia directa: un aggregate expone sólo sus "
    "group_by y cálculos. Si necesitás una columna del read después de agregar, agregala como group_by.",
    "dependencia futura/inexistente": "depends_on sólo puede nombrar ids de pasos definidos en steps; cada rama "
    "empieza con su propio read.",
}


class _VerifiedPlan(NamedTuple):
    decision: object
    selected: list
    operations: list
    issues: list
    resolutions: list


class _Requested(NamedTuple):
    thresholds: list
    periods: list
    statistics: list
    dimensions: dict
    chart: bool
    chart_required: bool
    values: list
    categories: bool
    several_rows: bool
    time_grain: str | None = None
    grouping: str | None = None
    stale: tuple = ()
    variables: dict | None = None
    latest_allowed: bool = True
    follow_up: bool = False
    extreme: bool = False
    extreme_orders: frozenset = frozenset()


_UNRESTRICTED = _Requested(
    thresholds=[],
    periods=[],
    statistics=[],
    dimensions={},
    chart=True,
    chart_required=False,
    values=[],
    categories=False,
    several_rows=True,
)


@dataclass
class _Proposal:
    attempts: int
    model: str | None
    tokens: list
    verified: _VerifiedPlan | None = None
    error: Exception | None = None
    rejections: list[str] = field(default_factory=list)


#
# Devuelve el paso que originó los campos, saltando los sort y latest que los conservan.
#
def _passthrough_origin(step, steps):
    while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
        step = steps.get(step.depends_on[0])
    return step


#
# Indica si un paso publica una única fila, por venir de un aggregate sin agrupar.
#
def _single_row(step, steps):
    origin = _passthrough_origin(step, steps)
    return origin is not None and origin.operation == "aggregate" and not origin.group_by


#
# Indica si todos los valores de filtro nombran períodos (años o fechas ISO).
#
def _period_values(values) -> bool:
    return bool(values) and all(_PERIOD_LITERAL.fullmatch(str(value).strip()) for value in values)


#
# Construye los filtros que acotan una columna a un período según su tipo.
#
def _period_filters(kind, sql_type, period, values) -> list[tuple[str, list]]:
    start, end = period.start, period.end
    if kind == "text":
        return [("eq" if len(values) == 1 else "in", values)]
    if kind == "year":
        if start is not None and end is not None and start.year == end.year:
            return [("eq", [start.year])]
        return [*([("ge", [start.year])] if start else []), *([("le", [end.year])] if end else [])]
    if sql_type == "TIMESTAMP":
        following = end + timedelta(days=1) if end else None
        return [
            *([("ge", [f"{start.isoformat()}T00:00:00"])] if start else []),
            *([("lt", [f"{following.isoformat()}T00:00:00"])] if following else []),
        ]
    return [*([("ge", [start.isoformat()])] if start else []), *([("le", [end.isoformat()])] if end else [])]


#
# Copia un paso con otro id, ajustando sus dependencias y las citas a resultados.
#
def _renamed_step(step, renamed):
    data = _renamed_references(step.model_dump(mode="json"), renamed)
    data["id"] = renamed[step.id]
    data["depends_on"] = [renamed.get(key, key) for key in step.depends_on]
    return type(step).model_validate_json(json.dumps(data))


#
# Reemplaza, de forma recursiva, los ids de pasos citados por los de sus copias.
#
def _renamed_references(value, renamed):
    if isinstance(value, list):
        return [_renamed_references(item, renamed) for item in value]
    if not isinstance(value, dict):
        return value

    copy = {key: _renamed_references(item, renamed) for key, item in value.items()}
    if copy.get("kind") == "result" and copy.get("step_id") in renamed:
        copy["step_id"] = renamed[copy["step_id"]]
    return copy


#
# Copia un paso citando las proyecciones numéricas en lugar de las columnas de texto que reemplazan.
#
def _projected_step(step, projections):
    data = _projected_references(step.model_dump(mode="json"), projections)
    return type(step).model_validate_json(json.dumps(data))


#
# Reemplaza, de forma recursiva, las citas a columnas y resultados por las de sus proyecciones.
#
def _projected_references(value, projections):
    if isinstance(value, list):
        return [_projected_references(item, projections) for item in value]
    if not isinstance(value, dict):
        return value

    copy = {key: _projected_references(item, projections) for key, item in value.items()}
    cited = {"column": "column", "result": "field"}.get(copy.get("kind"))
    if cited is not None and copy[cited] in projections:
        copy[cited] = projections[copy[cited]]
    return copy


#
# Devuelve el tipo SQL de cada columna del catálogo, indexado por dataset y columna.
#
def _column_types(catalog):
    return {
        (dataset.dataset_id, column.sql_name): column.sql_type
        for dataset in catalog
        for column in dataset.columns_metadata
    }


#
# Indica si el dataset registra las variables, dimensiones y valores pedidos y permite calcularlos.
#
def _registers(dataset, observed, mentions, numeric) -> bool:
    if observed is None:
        return False

    found = {
        "variable": observed["variable_columns"],
        "dimension": observed["requested_dimension_columns"],
        "value": observed["values"],
    }
    if not all(found[item["kind"]].get(item["text"]) for item in mentions if item["kind"] in found):
        return False
    if not numeric:
        return True

    computable = {
        name
        for column in dataset.columns_metadata
        if column.numeric_eligible
        for name in (column.sql_name, column.derived_from)
    }
    return all(
        computable & set(found["variable"][item["text"]])
        for item in mentions
        if item["kind"] == "variable"
    )


#
# Indica si el dataset tiene fechas para acotar los períodos pedidos o agrupar por tiempo.
#
def _dated(dataset, evidence, text_periods) -> bool:
    kinds = set(period_columns([dataset], text_periods).values())
    if evidence.get("requested_periods") and not kinds:
        return False

    return not evidence.get("requested_time_grain") or "date" in kinds


class PlanQueryService:
    #
    # Guarda el proveedor del Planner y las consultas de fuentes que usa el servicio.
    #
    def __init__(self, provider: PlannerProvider, queries: SourceQueries) -> None:
        self._provider = provider
        self._queries = queries

    #
    # Planifica la consulta: arma el contexto, propone y verifica un plan y devuelve el resultado.
    #
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
        conserved_mentions=(),
        requested_period=None,
    ) -> PlanningResult:
        question = PlanningInput(question=question).question
        own = [item for item in data_mentions if item not in conserved_mentions]
        data_mentions = self._planning_mentions(data_mentions, retrieval, conserved_mentions)
        dated_only = requested_period is not None
        if dated_only:
            data_mentions = [item for item in data_mentions if item.kind != "dimension"]
        limits = requested_limit(data_mentions), conserved_limits(data_mentions, previous_analysis)
        if question != retrieval.question.strip():
            raise PlannerValidationError(
                "La pregunta no coincide con la utilizada por el Retriever."
            )

        catalog = await self._verified_catalog(retrieval.candidates)
        asked = original_question or question
        require_latest = latest_requested(asked)
        previous = previous_operations(previous_analysis)
        temporal = (
            require_latest
            or "latest" in previous
            or any(latest_requested(item) for item in request_history)
            or dated_only
        )
        dates = requested_periods(data_mentions)
        catalog, excluded, periods = await self._temporal_scope(
            catalog, temporal or bool(dates), require_latest
        )
        evidence = await self._with_categories(
            self._catalog_evidence(retrieval.evidence, catalog), catalog
        )
        if dated_only:
            evidence = [item.model_copy(update={"dimension_matches": {}}) for item in evidence]

        thresholds, asked_statistics = requested_thresholds(data_mentions), requested_statistics(asked)
        statistics = asked_statistics or conserved_statistics(previous_analysis, asked)
        statistics = [] if dated_only else statistics
        chart_required = chart_requested(asked)
        grain = requested_time_grain(asked)
        regrouped = grain or any(item.kind in {"dimension", "date"} for item in own)
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
            thresholds=thresholds,
            periods=dates,
            statistics=statistics,
            chart=chart_required,
            time_grain=grain,
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

        values = [item.text for item in data_mentions if item.kind == "value"]
        asks_dimension = not dated_only and (
            bool(retrieval.dimension_mentions)
            or any(item.kind == "dimension" for item in data_mentions)
        )
        variables = {
            item.dataset_id: {name for names in item.variable_matches.values() for name in names}
            for item in evidence
        }
        named = [
            *(item.text for item in data_mentions if item.kind in {"value", "dimension"}),
            *self._variable_headers(catalog, variables),
        ]
        requested = _Requested(
            thresholds=thresholds,
            periods=dates,
            statistics=statistics,
            dimensions=requested_dimensions(context.evidence, catalog),
            chart=chart_required
            or "visualize" in previous
            or any(chart_requested(item) for item in request_history),
            chart_required=chart_required,
            values=values,
            categories=asks_dimension,
            several_rows=asks_dimension or limits[0] is not None or len(values) > 1,
            time_grain=grain,
            grouping=grain or (None if regrouped else conserved_time_grain(previous_analysis)),
            stale=tuple(stale_values(previous_analysis, data_mentions)),
            variables=variables,
            latest_allowed=temporal,
            follow_up=previous_analysis is not None,
            extreme=extreme_requested(asked),
            extreme_orders=frozenset() if dated_only else frozenset(asked_extremes(asked, named)),
        )
        verify = partial(
            self._verify,
            catalog=catalog,
            data_mentions=data_mentions,
            evidence=evidence,
            periods=periods,
            require_latest=require_latest,
            source_mentions=source_mentions,
            limits=limits,
            requested=requested,
        )
        if dated_only:
            proposal = await self._period_proposal(previous_analysis, catalog, periods, verify)
            if proposal is None:
                return self._without_period(common, retrieval.warnings)
            return self._planning_result(
                proposal, retrieval.unmatched_values, context, excluded, periods, catalog
            )

        reusable = chart_required and not (own or source_mentions or asked_statistics or grain)
        reused = reused_decision(previous_analysis, catalog, asked) if reusable else None
        proposal = await self._reuse(reused, verify) if reused else None
        if proposal is None:
            proposal = await self._propose(context, verify)
        if proposal.error is not None:
            return self._rejected_result(common, proposal, retrieval.warnings)

        context.evidence.pop("previous_attempt_feedback", None)
        return self._planning_result(
            proposal, retrieval.unmatched_values, context, excluded, periods, catalog
        )

    #
    # Devuelve los encabezados de las columnas que corresponden a las variables pedidas.
    #
    @staticmethod
    def _variable_headers(catalog, variables):
        return [
            column.original_name
            for dataset in catalog
            for column in dataset.columns_metadata
            if column.sql_name in variables.get(dataset.dataset_id, set())
        ]

    #
    # Combina las menciones de datos con los valores confirmados, quitando las que son columnas.
    #
    @staticmethod
    def _planning_mentions(data_mentions, retrieval, conserved=()):
        regrouped = PlanQueryService._regrouped_values(conserved, retrieval)
        mentions = [
            item
            for item in data_mentions
            if not (
                item.kind == "value"
                and (item.text in retrieval.dimension_mentions or item.text in regrouped)
            )
        ]
        mentions += [DataMention(text=value, kind="value") for value in retrieval.confirmed_values]
        return mentions

    #
    # Detecta los valores conservados que ahora corresponden a una columna por la que se agrupa.
    #
    @staticmethod
    def _regrouped_values(conserved, retrieval):
        dimensions = {
            (item.dataset_id, column)
            for item in retrieval.evidence
            for columns in item.dimension_matches.values()
            for column in columns
        }
        regrouped = set()
        for mention in conserved:
            matched = [
                (item.dataset_id, match.column)
                for item in retrieval.evidence
                for match in item.value_matches.get(mention.text, [])
            ]
            if mention.kind == "value" and matched and all(key in dimensions for key in matched):
                regrouped.add(mention.text)
        return regrouped

    #
    # Verifica los datasets candidatos contra PostgreSQL y devuelve el catálogo en el orden recuperado.
    #
    async def _verified_catalog(self, candidates):
        versions = {item.dataset_id: item.verified_version for item in candidates}
        if not versions:
            return []

        verified = await asyncio.to_thread(
            self._queries.verified_datasets, list(versions), versions=versions
        )
        by_id = {item.dataset_id: item for item in verified}
        return [by_id[item.dataset_id] for item in candidates if item.dataset_id in by_id]

    #
    # Determina el alcance temporal: períodos TEXT y candidatos sin fechas excluidos si se pide lo último.
    #
    async def _temporal_scope(self, catalog, temporal, require_latest):
        if not temporal:
            return catalog, [], []

        excluded, periods = await self._text_periods(catalog)
        if not require_latest:
            return catalog, [], periods

        dated = [item for item in catalog if str(item.dataset_id) not in excluded]
        return dated, excluded, periods

    #
    # Perfila las columnas TEXT con fechas de los datasets sin DATE/TIMESTAMP y lista los que no tienen.
    #
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

    #
    # Filtra la evidencia de recuperación a los datasets que siguen en el catálogo.
    #
    @staticmethod
    def _catalog_evidence(evidence, catalog):
        catalog_ids = {dataset.dataset_id for dataset in catalog}
        return [item for item in evidence if item.dataset_id in catalog_ids]

    #
    # Crea el resultado de cobertura insuficiente cuando ningún candidato tiene fechas interpretables.
    #
    @staticmethod
    def _without_dated_candidates(common, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="Los candidatos recuperados carecen de fechas DATE/TIMESTAMP o de períodos TEXT interpretables; no se elige un registro arbitrario.",
            missing_information=[
                "Las fuentes revisadas no tienen una columna de fecha o período interpretable para determinar "
                "la medición más reciente."
            ],
            warnings=list(retrieval_warnings),
        )

    #
    # Crea el resultado de cobertura insuficiente cuando no hay candidatos para planificar.
    #
    @staticmethod
    def _without_candidates(common, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="No hay candidatos disponibles para fundamentar una planificación.",
            missing_information=["No encontré fuentes disponibles relacionadas con esta consulta."],
            warnings=[
                *retrieval_warnings,
                "El alcance es el conjunto recuperado; no se evaluó todo el catálogo.",
            ],
        )

    #
    # Crea el resultado que pide aclaración porque el catálogo supera el presupuesto del modelo.
    #
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

    #
    # Revalida el plan del resultado citado para reutilizarlo; devuelve None si tiene defectos.
    #
    async def _reuse(self, decision, verify):
        try:
            verified = await verify(decision)
        except PlannerValidationError as error:
            _logger.warning("No se reutilizó el plan anterior: %s", error)
            return None

        if verified.decision.status != "plannable" or self._plan_defects(verified.issues):
            return None
        return _Proposal(0, None, [None, None], verified=verified)

    #
    # Resuelve la pregunta por el período repitiendo u ampliando las operaciones del resultado citado.
    #
    async def _period_proposal(self, analysis, catalog, periods, verify):
        repeated = repeated_decision(analysis, catalog)
        proposal = await self._reuse(repeated, verify) if repeated else None
        if self._dates_published(proposal, periods):
            return proposal

        spanned = period_span_decision(analysis, catalog)
        return await self._reuse(spanned, verify) if spanned else None

    #
    # Crea el resultado de cobertura insuficiente cuando el resultado citado no publica fechas.
    #
    @staticmethod
    def _without_period(common, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="El resultado citado no publica fechas y sus datasets no tienen una única columna de fecha "
            "con la que leer su período.",
            missing_information=[
                "El resultado anterior no publica fechas y sus datos no tienen una única columna de fecha con "
                "la que indicar su período."
            ],
            warnings=list(retrieval_warnings),
        )

    #
    # Indica si la propuesta reutilizada publica fechas en alguna de sus salidas.
    #
    @staticmethod
    def _dates_published(proposal, periods):
        if proposal is None:
            return False

        decision, _selected, operations, _issues, _resolutions = proposal.verified
        steps = {step.id: step for step in operations}
        return any(
            period_fields(steps[key], steps, periods)
            for key in decision.output_steps
            if key in steps
        )

    #
    # Pide planes al modelo con reintentos y retroalimentación hasta obtener uno verificado.
    #
    async def _propose(self, context, verify):
        feedback, attempt, model, tokens, rejections = [], 0, None, [None, None], []
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
                rejections.append(f"intento {attempt}: {error}")
                if attempt < _MAX_ATTEMPTS:
                    feedback = self._rejection_feedback(error)
                    continue
                return _Proposal(attempt, model, tokens, error=error, rejections=rejections)

            defects = self._plan_defects(verified.issues)
            if any(issue.code in _TEXT_COMPARISONS for issue in defects):
                context.evidence["text_comparison_rejected"] = True
            if not defects or attempt >= _MAX_ATTEMPTS:
                return _Proposal(attempt, model, tokens, verified=verified, rejections=rejections)

            _logger.warning(
                "Planner intento %s con defectos: %s",
                attempt,
                [issue.message for issue in defects],
            )
            feedback = [issue.message for issue in defects]
            rejections.append(f"intento {attempt}: {' '.join(feedback)}")

    #
    # Construye la retroalimentación para el modelo a partir del rechazo de su plan.
    #
    @staticmethod
    def _rejection_feedback(error):
        hints = [hint for symptom, hint in _HINTS.items() if symptom in str(error)]
        return [
            f"El plan anterior fue rechazado: {error} Corregilo respetando la gramática.",
            *hints,
        ]

    #
    # Devuelve los problemas de aclaración que son defectos del plan y no decisiones del usuario.
    #
    @staticmethod
    def _plan_defects(issues):
        return [
            issue
            for issue in issues
            if issue.severity == "clarification" and issue.code not in _USER_DECISIONS
        ]

    #
    # Devuelve los problemas de aclaración que requieren una decisión del usuario.
    #
    @staticmethod
    def _user_decisions(issues):
        return [
            issue
            for issue in issues
            if issue.severity == "clarification" and issue.code in _USER_DECISIONS
        ]

    #
    # Crea el resultado fallido cuando el modelo no produjo un plan verificable.
    #
    @staticmethod
    def _rejected_result(common, proposal, retrieval_warnings):
        attempt, error = proposal.attempts, proposal.error
        return PlanningResult(
            **common,
            status="failed",
            llm_status=None,
            attempts=attempt,
            model=proposal.model,
            reason=f"El modelo no produjo un plan verificable en {attempt} intentos: "
            + " | ".join(proposal.rejections or [str(error)]),
            missing_information=[],
            warnings=list(retrieval_warnings),
        )

    #
    # Construye el resultado de planificación final con estado, motivos, avisos y cobertura.
    #
    def _planning_result(self, proposal, unmatched_values, context, excluded, periods, catalog):
        decision, selected, operations, issues, resolutions = proposal.verified
        decisions = [issue.message for issue in self._user_decisions(issues)]
        defects = [issue.message for issue in self._plan_defects(issues)]
        if defects:
            _logger.warning(
                "Planner intento %s con defectos finales: %s", proposal.attempts, defects
            )
        status = self._final_status(defects, decisions, decision)

        reason = None
        absent = self._absent_values(issues, defects, unmatched_values)
        undated = self._undated_periods(issues, defects, selected, context, periods)
        if absent or undated:
            status, defects = "insufficient_coverage", []
            reason = "Los candidatos no registran los valores o las fechas que el plan necesitaba filtrar."
            filtered = any(issue.code == "filter_value_not_found" for issue in issues)
            missing = self._coverage_gaps(absent=absent if filtered else [], undated=undated)
            decision = decision.model_copy(update={"missing_information": missing})
        elif status == "insufficient_coverage":
            covered = self._covers_request(context, catalog, periods)
            reason = self._claimed_coverage_reason(decision, covered)
            status = "failed" if covered else "insufficient_coverage"
            missing = [] if covered else self._undated_request(context, catalog, periods)
            decision = decision.model_copy(update={"missing_information": missing})

        warnings = self._result_warnings(excluded, periods, operations, decision, selected, issues)
        reason = reason or self._result_reason(defects, decisions, status, selected, operations)
        plannable = status == "plannable"
        if plannable and proposal.attempts == 0:
            reason = (
                "Se reutilizaron las operaciones verificadas del resultado citado "
                "y se revalidaron contra el catálogo vigente."
            )
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

    #
    # Determina el estado final del plan según los defectos, las decisiones pendientes y el modelo.
    #
    @staticmethod
    def _final_status(defects, decisions, decision):
        if defects:
            return "failed"
        if decisions:
            return "needs_clarification"
        return decision.status

    #
    # Devuelve los valores pedidos sin registros si son lo único que impidió acotar el plan.
    #
    @staticmethod
    def _absent_values(issues, defects, unmatched_values):
        if not defects or not unmatched_values:
            return []

        blocking = [issue.code for issue in issues if issue.severity == "clarification"]
        if any(
            code not in {"filter_value_not_found", "missing_requested_filter"} for code in blocking
        ):
            return []
        return unmatched_values

    #
    # Devuelve los períodos pedidos que ningún dataset elegido puede acotar, si son el único impedimento.
    #
    @staticmethod
    def _undated_periods(issues, defects, selected, context, text_periods):
        blocking = {issue.code for issue in issues if issue.severity == "clarification"}
        if (
            not defects
            or blocking != {"missing_requested_period"}
            or period_columns(selected, text_periods)
        ):
            return []

        return list(context.evidence.get("requested_periods", []))

    #
    # Indica si algún candidato registra todo lo pedido, lo que prueba cobertura positiva.
    #
    @staticmethod
    def _covers_request(context, catalog, text_periods):
        evidence = context.evidence
        mentions = evidence.get("query_mentions", [])
        if not any(item["kind"] in {"variable", "value"} for item in mentions):
            return False

        observed = {item["dataset_id"]: item for item in evidence.get("observed_evidence", [])}
        numeric = numeric_comparison(evidence)
        return any(
            _registers(dataset, observed.get(str(dataset.dataset_id)), mentions, numeric)
            and _dated(dataset, evidence, text_periods)
            for dataset in catalog
        )

    #
    # Lista las carencias de fechas que impiden acotar períodos o agrupar por tiempo.
    #
    @classmethod
    def _undated_request(cls, context, catalog, text_periods):
        evidence = context.evidence
        kinds = set(period_columns(catalog, text_periods).values())
        undated = [] if kinds else evidence.get("requested_periods", [])
        grain = evidence.get("requested_time_grain")
        ungrouped = [_GRAIN_NAMES[grain]] if grain and "date" not in kinds else []
        return cls._coverage_gaps(undated=undated, ungrouped=ungrouped)

    #
    # Construye los enunciados de carencias: falta de registros y de columnas de fecha.
    #
    @staticmethod
    def _coverage_gaps(absent=(), undated=(), ungrouped=()):
        return [
            *(f"No hay registros de {value} en los datos consultados." for value in absent),
            *(
                f"Los datos consultados no tienen una columna de fecha con la que acotarlos {period_phrase(period)}."
                for period in undated
            ),
            *(
                f"Los datos consultados no tienen una columna de fecha con la que agruparlos por {grain}."
                for grain in ungrouped
            ),
        ]

    #
    # Describe, para trazabilidad, la cobertura insuficiente declarada por el modelo.
    #
    @staticmethod
    def _claimed_coverage_reason(decision, covered):
        claimed = (
            f"Motivo del modelo: {decision.reason.rstrip('. ')}. Faltante según el modelo: "
            + "; ".join(item.rstrip(". ") for item in decision.missing_information)
            + "."
        )
        if covered:
            return (
                "El modelo declaró cobertura insuficiente, pero un candidato revisado registra las variables, los "
                f"valores y las fechas pedidos: se trata como un plan fallido, no como falta de datos. {claimed}"
            )
        return (
            "El modelo declaró cobertura insuficiente y la evidencia observada no muestra un candidato que registre "
            f"todo lo pedido; la respuesta no repite su texto. {claimed}"
        )

    #
    # Reúne las advertencias del resultado sin duplicados.
    #
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

    #
    # Compone el motivo del resultado de planificación según su estado.
    #
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

    #
    # Devuelve la pregunta de aclaración: las decisiones del usuario, ninguna si hay defectos, o la del modelo.
    #
    @staticmethod
    def _clarification_question(decisions, defects, decision):
        if decisions:
            return " ".join(decisions)
        if defects:
            return None
        return decision.clarification_question

    #
    # Repara y verifica la decisión del modelo contra el catálogo, los valores, los períodos y lo pedido.
    #
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
        requested=_UNRESTRICTED,
    ):
        decision = self._repaired(decision, catalog, periods, requested)
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
            issues.extend(
                validate_requested_scope(
                    operations,
                    decision.output_steps,
                    thresholds=requested.thresholds,
                    statistics=requested.statistics,
                    extreme=bool(requested.extreme_orders),
                    dimensions=requested.dimensions,
                    categories=requested.categories,
                    chart=requested.chart_required,
                    time_grain=requested.time_grain,
                    periods=requested.periods,
                    dated_columns=period_columns(selected, periods),
                )
            )
            issues.extend(validate_text_orderings(operations, requested.variables or {}))
            if not requested.latest_allowed:
                issues.extend(
                    validate_unrequested_latest(
                        operations, decision.output_steps, follow_up=requested.follow_up
                    )
                )

        issues.extend(await self._catalog_changes(selected, catalog))
        return _VerifiedPlan(decision, selected, operations, issues, resolutions)

    #
    # Aplica en orden las reparaciones determinísticas a la decisión del modelo.
    #
    @staticmethod
    def _repaired(decision, catalog, periods, requested):
        decision = PlanQueryService._merged_selections(decision)
        decision = PlanQueryService._read_selections(decision, catalog)
        decision = without_stale_filters(decision, requested.stale)
        decision = PlanQueryService._publication_steps(decision)
        decision = PlanQueryService._dependency_order(decision)
        decision = PlanQueryService._unrequested_charts(decision, requested.chart)
        decision = PlanQueryService._single_row_republications(decision, {"latest"})
        decision = PlanQueryService._unpublished_branches(decision)
        decision = PlanQueryService._parallel_branches(decision, requested.variables)
        decision = PlanQueryService._unranked_extremes(decision, catalog, requested)
        decision = PlanQueryService._projected_orderings(decision, catalog)
        decision = PlanQueryService._text_measurement_branches(decision, catalog, requested.variables)
        decision = PlanQueryService._result_references(decision)
        decision = PlanQueryService._missing_groupings(decision, catalog)
        decision = PlanQueryService._omitted_categories(decision, catalog, requested)
        decision = with_time_grain(decision, requested.grouping, _column_types(catalog))
        decision = PlanQueryService._undated_groupings(decision, catalog, requested.time_grain)
        decision = PlanQueryService._omitted_thresholds(decision, catalog, requested.thresholds)
        decision = PlanQueryService._omitted_periods(decision, catalog, requested.periods, periods)
        if not requested.several_rows:
            decision = PlanQueryService._single_row_republications(decision, {"sort"})
        decision = PlanQueryService._extremes_as_sort(decision, catalog)
        if requested.extreme:
            decision = PlanQueryService._extreme_rankings(decision, catalog)
        decision = PlanQueryService._measured_latest(decision, requested.variables)
        return PlanQueryService._latest_in_python(decision, periods)

    #
    # Detecta si los datasets elegidos cambiaron en el catálogo durante la planificación.
    #
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

    #
    # Agrega a la evidencia los valores frecuentes de las columnas donde se hallaron valores pedidos.
    #
    async def _with_categories(self, evidence, catalog):
        datasets = {item.dataset_id: item for item in catalog}
        enriched = []
        for item in evidence:
            categories = await self._category_values(item, datasets)
            enriched.append(
                item.model_copy(update={"category_values": categories}) if categories else item
            )
        return enriched

    #
    # Lee los valores frecuentes de las columnas con coincidencias, ignorando los textos largos.
    #
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
            ):
                _logger.warning(
                    "No se pudieron leer categorías de %s: %s", column, type(rows).__name__
                )
                continue

            values = [row["value"] for row in rows if len(row["value"]) <= _CATEGORY_LENGTH]
            if len(values) > 1:
                categories[column] = values
        return categories

    #
    # Fusiona las selecciones repetidas de un mismo dataset unificando sus columnas.
    #
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
                return decision
            previous.columns = list(dict.fromkeys([*previous.columns, *item.columns]))
        changed.datasets = list(merged.values())
        return changed

    #
    # Completa las columnas de cada dataset elegido con las que leen realmente los pasos read.
    #
    @staticmethod
    def _read_selections(decision, catalog):
        available = {
            item.dataset_id: {column.sql_name for column in item.columns_metadata} for item in catalog
        }
        read = {}
        for step in decision.steps:
            for ref in step.columns if step.operation == "read" else ():
                read.setdefault(ref.dataset_id, []).append(ref.column)

        changed = decision.model_copy(deep=True)
        for selection in changed.datasets:
            used = read.get(selection.dataset_id, [])
            existing = available.get(selection.dataset_id, set())
            kept = [name for name in selection.columns if name in existing or name in used]
            columns = list(dict.fromkeys([*kept, *(name for name in used if name in existing)]))
            selection.columns = columns or selection.columns
        if read:
            changed.datasets = [item for item in changed.datasets if item.dataset_id in read]
        return changed if changed != decision else decision

    #
    # Convierte en referencias a resultados las columnas citadas después de un read.
    #
    @staticmethod
    def _result_references(decision):
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
                origin = steps[origin.depends_on[0]]
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

    #
    # Publica los pasos finales que continúan un resultado publicado y quedaron fuera de las salidas.
    #
    @staticmethod
    def _unpublished_branches(decision):
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

    #
    # Publica las ramas paralelas olvidadas que calculan lo mismo sobre otro dataset con la variable pedida.
    #
    @staticmethod
    def _parallel_branches(decision, variables):
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        consumed = {key for step in decision.steps for key in step.depends_on}
        chains = [
            PlanQueryService._published_chain(steps.get(key), steps) for key in decision.output_steps
        ]
        shapes = {PlanQueryService._parallel_shape(chain) for chain in chains if chain}
        read = {
            chain[-1].columns[0].dataset_id
            for chain in chains
            if chain and chain[-1].operation == "read" and chain[-1].columns
        }
        forgotten = []
        for step in decision.steps:
            if step.id in consumed or step.id in decision.output_steps:
                continue
            chain = PlanQueryService._published_chain(step, steps)
            source = chain[-1] if chain[-1].operation == "read" and chain[-1].columns else None
            dataset_id = source.columns[0].dataset_id if source else None
            measured = (variables or {}).get(dataset_id, set())
            if (
                source is None
                or dataset_id in read
                or PlanQueryService._parallel_shape(chain) not in shapes
                or not measured & {ref.column for ref in source.columns}
            ):
                continue
            forgotten.append(step.id)
            read.add(dataset_id)

        if not forgotten or len(decision.output_steps) + len(forgotten) > _MAX_PLAN_STEPS:
            return decision
        return decision.model_copy(update={"output_steps": [*decision.output_steps, *forgotten]})

    #
    # Cambia por su proyección numérica las mediciones TEXT por las que ordena una rama, con sus referencias.
    #
    @staticmethod
    def _projected_orderings(decision, catalog):
        if decision.status != "plannable":
            return decision

        projections = {
            (dataset.dataset_id, column.derived_from): column.sql_name
            for dataset in catalog
            for column in dataset.columns_metadata
            if column.derived_from and column.numeric_eligible
        }
        steps = {step.id: step for step in decision.steps}
        renamed = {}
        for key in decision.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            read = chain[-1] if chain and chain[-1].operation == "read" and chain[-1].columns else None
            found = PlanQueryService._ordered_projections(chain, read, projections) if read else {}
            if not found:
                continue

            for step in decision.steps:
                if step.id == read.id or read.id in PlanQueryService._ancestors(step, steps):
                    renamed.setdefault(step.id, {}).update(found)

        if not renamed:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [
            _projected_step(step, renamed[step.id]) if step.id in renamed else step
            for step in changed.steps
        ]
        return PlanQueryService._read_selections(changed, catalog)

    #
    # Devuelve las columnas del read por las que ordena una rama y que tienen proyección numérica.
    #
    @staticmethod
    def _ordered_projections(chain, read, projections):
        ordered = {
            item.field.field if isinstance(item.field, ResultRef) else item.field.column
            for step in chain
            for item in step.order_by
        }
        dataset_id = read.columns[0].dataset_id
        return {
            ref.column: projections[(dataset_id, ref.column)]
            for ref in read.columns
            if ref.column in ordered and (dataset_id, ref.column) in projections
        }

    #
    # Quita las ramas que calculan sobre mediciones TEXT no numéricas si quedan otras ramas.
    #
    @staticmethod
    def _text_measurement_branches(decision, catalog, variables):
        if decision.status != "plannable":
            return decision

        textual = {
            (dataset.dataset_id, column.sql_name)
            for dataset in catalog
            for column in dataset.columns_metadata
            if column.sql_type == "TEXT"
            and not column.numeric_eligible
            and column.sql_name in (variables or {}).get(dataset.dataset_id, set())
        }
        steps = {step.id: step for step in decision.steps}
        blocked = []
        for key in decision.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            read = chain[-1] if chain and chain[-1].operation == "read" and chain[-1].columns else None
            if read is None:
                continue
            dataset_id = read.columns[0].dataset_id
            if any((dataset_id, name) in textual for name in PlanQueryService._computed_fields(chain)):
                blocked.append(key)

        if not blocked or len(blocked) == len(decision.output_steps):
            return decision
        return PlanQueryService._without_outputs(decision.model_copy(deep=True), blocked)

    #
    # Devuelve los campos que una rama usa como números en cálculos y órdenes.
    #
    @staticmethod
    def _computed_fields(chain):
        refs = [
            ref
            for step in chain
            for calculation in step.calculations
            if calculation.function not in _TEXT_CALCULATIONS
            for ref in calculation.inputs
        ]
        refs += [item.field for step in chain if step.operation == "sort" for item in step.order_by]
        return {ref.field if isinstance(ref, ResultRef) else ref.column for ref in refs}

    #
    # Devuelve la forma de una rama: sus operaciones y los valores de texto que filtra.
    #
    @staticmethod
    def _parallel_shape(chain):
        values = sorted(
            {
                str(value).casefold()
                for step in chain
                for predicate in step.filters
                if predicate.operator in {"eq", "ieq", "in"}
                for value in predicate.values
                if isinstance(value, str) and not _PERIOD_LITERAL.fullmatch(value.strip())
            }
        )
        return PlanQueryService._branch_shape(chain), tuple(values)

    #
    # Devuelve las operaciones de una rama sin filtros ni columnas: cálculos y agrupaciones temporales.
    #
    @staticmethod
    def _branch_shape(chain):
        return tuple(
            (
                step.operation,
                tuple(sorted(item.function for item in step.calculations)),
                tuple(item.time_grain for item in step.group_by),
            )
            for step in chain
        )

    #
    # Devuelve los ids de todos los pasos de los que depende un paso.
    #
    @staticmethod
    def _ancestors(step, steps):
        pending, found = list(step.depends_on), set()
        while pending:
            key = pending.pop()
            if key in steps and key not in found:
                found.add(key)
                pending.extend(steps[key].depends_on)
        return found

    #
    # Agrega el group_by omitido cuando se ordena o grafica por una categoría tras un aggregate sin agrupar.
    #
    @staticmethod
    def _missing_groupings(decision, catalog):
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

    #
    # Devuelve el aggregate sin agrupar al que le falta agrupar por la categoría de texto citada.
    #
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

    #
    # Agrega la categoría omitida como group_by de un aggregate sin agrupar cuando es inequívoca.
    #
    @staticmethod
    def _omitted_categories(decision, catalog, requested):
        if decision.status != "plannable":
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        types = _column_types(catalog)
        modified = False
        for step in changed.steps:
            if step.operation != "aggregate" or step.group_by or len(step.depends_on) != 1:
                continue

            read = _passthrough_origin(steps.get(step.depends_on[0]), steps)
            if read is None or read.operation != "read":
                continue

            unused = PlanQueryService._unused_categories(step, read, types)
            preferred = [
                requested.dimensions.get(read.columns[0].dataset_id, set()),
                PlanQueryService._compared_columns(read, requested.values),
            ]
            ranked = PlanQueryService._ranks_calculation(step, changed.steps, steps)
            category = PlanQueryService._omitted_category(unused, preferred, ranked)
            if category is None:
                continue

            step.group_by.append(
                GroupingSpec(
                    field=ResultRef(kind="result", step_id=step.depends_on[0], field=category),
                    output_name=category,
                )
            )
            modified = True
        return changed if modified else decision

    #
    # Elige la categoría omitida según las preferencias, o None si no hay una única candidata.
    #
    @staticmethod
    def _omitted_category(unused, preferred, ranked):
        present = [columns for columns in preferred if unused.keys() & columns]
        chosen = [name for name in unused if name in present[0]] if present else []
        for kinds in _CATEGORY_PREFERENCE if ranked and not chosen else ():
            chosen = [name for name, kind in unused.items() if kind in kinds]
            if chosen:
                break
        return chosen[0] if len(chosen) == 1 else None

    #
    # Devuelve las columnas filtradas con dos o más de los valores pedidos, que la consulta compara.
    #
    @staticmethod
    def _compared_columns(read, values):
        compared = set()
        for item in read.filters:
            stored = {str(value).casefold() for value in item.values}
            asked = [value for value in values if value.casefold() in stored]
            if isinstance(item.field, ColumnRef) and item.operator == "in" and len(asked) > 1:
                compared.add(item.field.column)
        return compared

    #
    # Devuelve las columnas de texto o fecha del read que no se usan, con su tipo.
    #
    @staticmethod
    def _unused_categories(aggregate, read, types):
        fixed = {
            item.field.column
            for item in read.filters
            if isinstance(item.field, ColumnRef) and item.operator in {"eq", "ieq"}
        }
        calculated = {
            ref.field
            for calculation in aggregate.calculations
            for ref in calculation.inputs
            if isinstance(ref, ResultRef)
        }
        kinds = set().union(*_CATEGORY_PREFERENCE)
        return {
            ref.column: types.get((ref.dataset_id, ref.column))
            for ref in read.columns
            if ref.column not in fixed | calculated
            and types.get((ref.dataset_id, ref.column)) in kinds
        }

    #
    # Indica si algún sort ordena o filtra por un cálculo del aggregate, lo que espera varias filas.
    #
    @staticmethod
    def _ranks_calculation(aggregate, ordered_steps, steps):
        calculated = {calculation.output_name for calculation in aggregate.calculations}
        for step in ordered_steps:
            if step.operation != "sort" or not step.depends_on:
                continue
            if _passthrough_origin(steps.get(step.depends_on[0]), steps) is not aggregate:
                continue
            fields = [item.field for item in [*step.order_by, *step.filters]]
            if any(isinstance(ref, ResultRef) and ref.field in calculated for ref in fields):
                return True
        return False

    #
    # Aplica los umbrales pedidos que el modelo anunció pero no escribió en el plan.
    #
    @staticmethod
    def _omitted_thresholds(decision, catalog, thresholds):
        if not thresholds or decision.status != "plannable":
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        types = _column_types(catalog)
        modified = False
        for key in changed.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            for threshold in thresholds:
                applied = any(
                    applies_threshold(predicate, threshold)
                    for step in chain
                    for predicate in step.filters
                )
                target = PlanQueryService._threshold_target(chain, threshold, types)
                if applied or target is None:
                    continue

                step, field = target
                step.filters.append(
                    FilterSpec(field=field, operator=threshold.operator, values=[threshold.value])
                )
                modified = True
        return changed if modified else decision

    #
    # Acota cada rama al período pedido, copiándolas por período o quitando las que no tienen fechas.
    #
    @staticmethod
    def _omitted_periods(decision, catalog, requested, text_periods):
        if not requested or decision.status != "plannable":
            return decision

        dated, types = period_columns(catalog, text_periods), _column_types(catalog)
        texts = {(item.dataset_id, item.column): item for item in text_periods}
        keys = {*texts, *dated}
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        undated, kept, copied = [], [], []
        for key in changed.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            read = chain[-1] if chain and chain[-1].operation == "read" else None
            if read is None:
                continue

            dataset_id = read.columns[0].dataset_id
            targets = [
                PlanQueryService._period_target(dataset_id, dated, period, texts)
                for period in requested
            ]
            if None in targets:
                undated.append(key)
            elif len(requested) > 1:
                covered = PlanQueryService._covered_period(chain, dated, requested)
                if covered is None:
                    copied.append((key, chain, targets))
                else:
                    kept.append((chain, targets, covered))
            else:
                PlanQueryService._without_period_filters(changed, chain, keys, types)
                PlanQueryService._bounded_read(changed, read, targets[0], requested[0], types)

        if len(requested) > 1:
            changed = PlanQueryService._period_copies(changed, copied, kept, requested, keys, types)
            if changed is None:
                return decision
        if undated and len(undated) < len(changed.output_steps):
            return PlanQueryService._without_outputs(changed, undated)
        return changed if changed != decision else decision

    #
    # Devuelve el índice del único período pedido que la rama ya acota, o None.
    #
    @staticmethod
    def _covered_period(chain, dated, requested):
        applied = filters_period([predicate for step in chain for predicate in step.filters], dated)
        matched = [index for index, period in enumerate(requested) if period_matches(applied, period)]
        return matched[0] if len(matched) == 1 else None

    #
    # Copia las ramas por cada período pedido que ninguna rama equivalente responde.
    #
    @staticmethod
    def _period_copies(decision, copied, kept, periods, keys, types):
        covered, templates = {}, {}
        for _key, chain, targets in copied:
            templates.setdefault(PlanQueryService._branch_signature(chain, keys, types), (chain, targets))
        for chain, targets, index in kept:
            signature = PlanQueryService._branch_signature(chain, keys, types)
            covered.setdefault(signature, set()).add(index)
            templates.setdefault(signature, (chain, targets))

        branches = [
            (chain, targets, missing)
            for signature, (chain, targets) in templates.items()
            if (missing := [index for index in range(len(periods)) if index not in covered.get(signature, set())])
        ]
        if not branches and not copied:
            return decision

        order = {step.id: position for position, step in enumerate(decision.steps)}
        renamed, bounded = {}, set()
        for index in range(len(periods)):
            needed = [chain for chain, _, missing in branches if index in missing]
            union = sorted({step.id: step for chain in needed for step in chain}.values(), key=lambda s: order[s.id])
            renamed[index] = {step.id: f"{step.id}_p{index + 1}" for step in union}
            copies = {step.id: _renamed_step(step, renamed[index]) for step in union}
            decision.steps.extend(copies.values())
            chains = [
                ([copies[step.id] for step in chain], targets[index])
                for chain, targets, missing in branches
                if index in missing
            ]
            for chain, _target in chains:
                PlanQueryService._without_period_filters(decision, chain, keys, types)
            for chain, target in chains:
                if chain[-1].id not in bounded:
                    PlanQueryService._bounded_read(decision, chain[-1], target, periods[index], types)
                    bounded.add(chain[-1].id)

        copied_keys = {key for key, _, _ in copied}
        published = [
            renamed[index][chain[0].id] for chain, _, missing in branches for index in missing
        ]
        decision.output_steps = [
            *(key for key in decision.output_steps if key not in copied_keys),
            *published,
        ]
        changed = PlanQueryService._without_outputs(decision, [])
        too_large = (
            len(changed.steps) > _MAX_PLAN_STEPS
            or len(changed.output_steps) > _MAX_PLAN_STEPS
            or any(len(step.id) > _MAX_STEP_ID_LENGTH for step in changed.steps)
        )
        return None if too_large else changed

    #
    # Devuelve la firma de una rama sin sus filtros de período, para detectar ramas equivalentes.
    #
    @staticmethod
    def _branch_signature(chain, keys, types):
        names = PlanQueryService._period_names(chain, keys)
        scope = sorted(
            (repr(predicate.field), predicate.operator, repr(predicate.values))
            for step in chain
            for predicate in step.filters
            if not PlanQueryService._period_filter(predicate, keys, names, types)
        )
        return chain[-1].columns[0].dataset_id, PlanQueryService._branch_shape(chain), tuple(scope)

    #
    # Quita de una rama los filtros con que acota su período.
    #
    @staticmethod
    def _without_period_filters(decision, chain, periods, types):
        read = chain[-1]
        names = PlanQueryService._period_names(chain, periods)
        removed = set()
        for step in chain:
            kept = []
            for predicate in step.filters:
                if PlanQueryService._period_filter(predicate, periods, names, types):
                    if isinstance(predicate.field, ColumnRef):
                        removed.add(predicate.field.column)
                    continue
                kept.append(predicate)
            step.filters = kept

        invented = {name for name in removed if (read.columns[0].dataset_id, name) not in types}
        read.columns = [ref for ref in read.columns if ref.column not in invented] or read.columns
        for selection in decision.datasets:
            if selection.dataset_id == read.columns[0].dataset_id:
                selection.columns = [name for name in selection.columns if name not in invented]

    #
    # Devuelve los nombres con que los pasos posteriores citan una fecha, un año o una fecha agrupada.
    #
    @staticmethod
    def _period_names(chain, periods):
        return {column for _, column in periods} | {
            item.output_name for step in chain for item in step.group_by if item.time_grain != "raw"
        }

    #
    # Indica si un filtro acota el período y no una medición.
    #
    @staticmethod
    def _period_filter(predicate, periods, names, types):
        ref = predicate.field
        if not isinstance(ref, ColumnRef):
            return ref.field in names

        key = (ref.dataset_id, ref.column)
        return key in periods or (types.get(key) in {None, "TEXT"} and _period_values(predicate.values))

    #
    # Quita las ramas que agrupan por tiempo sobre datasets sin columna DATE/TIMESTAMP si quedan otras.
    #
    @staticmethod
    def _undated_groupings(decision, catalog, grain):
        if grain is None or decision.status != "plannable":
            return decision

        dated = {owner for (owner, _), kind in period_columns(catalog).items() if kind == "date"}
        steps = {step.id: step for step in decision.steps}
        undated = []
        for key in decision.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            read = chain[-1] if chain and chain[-1].operation == "read" else None
            calculates = any(step.operation == "aggregate" for step in chain)
            if read is not None and calculates and read.columns[0].dataset_id not in dated:
                undated.append(key)

        if not undated or len(undated) == len(decision.output_steps):
            return decision
        return PlanQueryService._without_outputs(decision.model_copy(deep=True), undated)

    #
    # Elige la columna y el tipo con que un dataset acota un período, con los valores TEXT incluidos.
    #
    @staticmethod
    def _period_target(dataset_id, dated, period, texts):
        columns = {}
        for (owner, column), kind in dated.items():
            if owner == dataset_id:
                columns.setdefault("text" if kind.startswith("text") else kind, []).append(column)

        whole_years = (period.start is None or (period.start.month, period.start.day) == (1, 1)) and (
            period.end is None or (period.end.month, period.end.day) == (12, 31)
        )
        if "date" in columns:
            return (columns["date"][0], "date", None) if len(columns["date"]) == 1 else None
        if len(columns.get("year", [])) == 1 and whole_years:
            return columns["year"][0], "year", None
        if len(columns.get("text", [])) != 1:
            return None

        text = texts[(dataset_id, columns["text"][0])]
        values = [
            value
            for value in text.values
            if within_period(value, text.day_first, period.start, period.end)
        ]
        return (text.column, "text", values) if 0 < len(values) <= MAX_FILTER_VALUES else None

    #
    # Acota el read al período con su columna de fecha, añadiendo la columna y los filtros necesarios.
    #
    @staticmethod
    def _bounded_read(decision, read, target, period, types):
        column, kind, values = target
        dataset_id = read.columns[0].dataset_id
        ref = ColumnRef(kind="column", dataset_id=dataset_id, column=column)
        if ref not in read.columns:
            read.columns.append(ref)
        for selection in decision.datasets:
            if selection.dataset_id == dataset_id and column not in selection.columns:
                selection.columns.append(column)

        bounds = _period_filters(kind, types.get((dataset_id, column)), period, values)
        read.filters += [
            FilterSpec(field=ref, operator=operator, values=items) for operator, items in bounds
        ]

    #
    # Quita salidas del plan junto con los pasos y datasets que sólo ellas usaban.
    #
    @staticmethod
    def _without_outputs(decision, removed):
        steps = {step.id: step for step in decision.steps}
        outputs = [key for key in decision.output_steps if key not in removed]
        used = {
            key
            for output in outputs
            for key in {output, *PlanQueryService._ancestors(steps[output], steps)}
        }
        decision.steps = [step for step in decision.steps if step.id in used]
        read = {
            ref.dataset_id
            for step in decision.steps
            if step.operation == "read"
            for ref in step.columns
        }
        decision.datasets = [item for item in decision.datasets if item.dataset_id in read]
        decision.output_steps = outputs
        return decision

    #
    # Devuelve los pasos que producen una salida, desde ella hasta su read.
    #
    @staticmethod
    def _published_chain(step, steps):
        chain = []
        while step is not None:
            chain.append(step)
            step = steps.get(step.depends_on[0]) if len(step.depends_on) == 1 else None
        return chain

    #
    # Elige el paso y el campo donde aplicar un umbral omitido, si el destino es inequívoco.
    #
    @staticmethod
    def _threshold_target(chain, threshold, types):
        if not threshold.compares_statistic:
            read = chain[-1] if chain and chain[-1].operation == "read" else None
            measured = [
                ref
                for ref in (read.columns if read else [])
                if types.get((ref.dataset_id, ref.column)) in {"NUMERIC", "BIGINT"}
            ]
            return (read, measured[0]) if len(measured) == 1 else None

        aggregate = next((step for step in chain if step.operation == "aggregate"), None)
        if aggregate is None or len(aggregate.calculations) != 1:
            return None

        sorts = chain[: chain.index(aggregate)]
        sort = next((step for step in reversed(sorts) if step.operation == "sort"), None)
        if sort is None:
            return None

        name = aggregate.calculations[0].output_name
        return sort, ResultRef(kind="result", step_id=sort.depends_on[0], field=name)

    #
    # Convierte un latest sobre una variable numérica en un sort descendente de un puesto.
    #
    @staticmethod
    def _extremes_as_sort(decision, catalog):
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

    #
    # Limita a los primeros puestos los sort de extremos y deja de publicar los read que los alimentan.
    #
    @staticmethod
    def _extreme_rankings(decision, catalog):
        if decision.status != "plannable":
            return decision

        types = _column_types(catalog)
        steps = {step.id: step for step in decision.steps}
        changed, modified = decision.model_copy(deep=True), False
        targets = {
            PlanQueryService._ranked_records(steps.get(key), steps, types)
            for key in decision.output_steps
        } - {None}
        for step in changed.steps:
            if step.id in targets:
                step.limit = _EXTREME_ROWS
                modified = True

        inputs = {
            key
            for output in decision.output_steps
            if output in steps
            for key in PlanQueryService._ancestors(steps[output], steps)
        }
        outputs = [
            key
            for key in changed.output_steps
            if not (key in inputs and key in steps and steps[key].operation == "read")
        ]
        if outputs and outputs != changed.output_steps:
            changed.output_steps, modified = outputs, True
        return changed if modified else decision

    #
    # Agrega el orden por la variable a las salidas que leen registros sin elegir el extremo pedido.
    #
    @staticmethod
    def _unranked_extremes(decision, catalog, requested):
        grouped = requested.categories or requested.statistics or requested.grouping
        unique = len(requested.extreme_orders) == 1 and len(requested.values) < 2
        if decision.status != "plannable" or grouped or not unique:
            return decision

        types = _column_types(catalog)
        changed, modified = decision.model_copy(deep=True), False
        steps = {step.id: step for step in changed.steps}
        for index, key in enumerate(changed.output_steps):
            measured = PlanQueryService._extreme_column(steps.get(key), requested.variables, types)
            if measured is None or len(changed.steps) >= _MAX_PLAN_STEPS:
                continue

            sort = PlanQueryService._extreme_sort(steps[key], measured, *requested.extreme_orders)
            changed.steps.append(sort)
            changed.output_steps[index], modified = sort.id, True
        return changed if modified else decision

    #
    # Devuelve la columna de la variable pedida en un read sin orden ni límite, si es inequívoca.
    #
    @staticmethod
    def _extreme_column(read, variables, types):
        if read is None or read.operation != "read" or read.order_by or read.limit is not None:
            return None

        asked = variables or {}
        measured = [ref for ref in read.columns if ref.column in asked.get(ref.dataset_id, set())]
        numeric = [ref for ref in measured if PlanQueryService._is_measured(ref.column, read, types)]
        found = numeric or measured
        return found[0] if len(found) == 1 else None

    #
    # Construye el sort del extremo sobre un read, que deja de traer los registros sin la variable.
    #
    @staticmethod
    def _extreme_sort(read, measured, order):
        constrained = {item.field.column for item in read.filters if isinstance(item.field, ColumnRef)}
        if measured.column not in constrained:
            read.filters.append(FilterSpec(field=measured, operator="is_not_null", values=[]))

        return SortPlanStep(
            id=f"{read.id[: _MAX_STEP_ID_LENGTH - len(_EXTREME_SUFFIX)]}{_EXTREME_SUFFIX}",
            tool="sql",
            operation="sort",
            depends_on=[read.id],
            order_by=[
                OrderingSpec(
                    field=ResultRef(kind="result", step_id=read.id, field=measured.column),
                    direction=order,
                )
            ],
            limit=_EXTREME_ROWS,
            purpose="Ordenar por la variable para mostrar el extremo pedido y sus empates.",
        )

    #
    # Devuelve el id del sort sin límite que ordena registros por una variable medida y se publica.
    #
    @staticmethod
    def _ranked_records(step, steps, types):
        if step is not None and step.operation == "visualize" and step.depends_on:
            step = steps.get(step.depends_on[0])
        if step is None or step.operation != "sort" or step.limit is not None:
            return None

        read = _passthrough_origin(steps.get(step.depends_on[0]), steps)
        field = step.order_by[0].field
        name = field.field if isinstance(field, ResultRef) else field.column
        measured = read is not None and read.operation == "read"
        return step.id if measured and PlanQueryService._is_measured(name, read, types) else None

    #
    # Construye el orden descendente por la variable medida que equivale al extremo de un latest.
    #
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

    #
    # Indica si un campo del read corresponde a una columna numérica medida.
    #
    @staticmethod
    def _is_measured(field, read, types):
        kinds = {
            types.get((ref.dataset_id, ref.column)) for ref in read.columns if ref.column == field
        }
        return bool(kinds) and kinds <= {"NUMERIC", "BIGINT"}

    #
    # Aplica la cantidad de filas pedida o conservada del resultado citado a los pasos publicados.
    #
    @staticmethod
    def _result_limits(decision, operations, requested, conserved):
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

    #
    # Reemplaza los pasos que sólo republican su dependencia por la dependencia publicada.
    #
    @staticmethod
    def _publication_steps(decision):
        if decision.status != "plannable":
            return decision

        used = {key for step in decision.steps for key in step.depends_on}
        replace = {}
        for step in decision.steps:
            source = PlanQueryService._republished_step(step, used)
            if source is not None:
                replace.setdefault(step.id, []).append(source)
        if not replace:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        steps = {step.id: step for step in changed.steps}
        feeding = {
            key
            for output in changed.output_steps
            if output in steps
            for key in PlanQueryService._ancestors(steps[output], steps)
        }
        unlisted = [
            source
            for key, sources in replace.items()
            if key not in decision.output_steps and _PUBLICATION_STEP.fullmatch(key)
            for source in sources
            if source not in feeding
        ]
        changed.output_steps = list(
            dict.fromkeys(
                [
                    *(
                        published
                        for key in changed.output_steps
                        for published in replace.get(key, [key])
                    ),
                    *unlisted,
                ]
            )
        )
        return changed

    #
    # Quita los gráficos no pedidos y publica su entrada en su lugar.
    #
    @staticmethod
    def _unrequested_charts(decision, requested):
        if requested or decision.status != "plannable":
            return decision

        charts = {
            step.id: step.depends_on[0]
            for step in decision.steps
            if step.operation == "visualize" and step.depends_on
        }
        used = {key for step in decision.steps for key in step.depends_on}
        if not charts or charts.keys() & used:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in charts]
        changed.output_steps = list(
            dict.fromkeys(charts.get(key, key) for key in changed.output_steps)
        )
        return changed

    #
    # Quita los latest o sort finales que sólo republican una fila única y publica su entrada.
    #
    @staticmethod
    def _single_row_republications(decision, operations):
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        used = {key for step in decision.steps for key in step.depends_on}
        replace = {
            step.id: step.depends_on[0]
            for step in decision.steps
            if step.operation in operations
            and step.id not in used
            and not step.filters
            and len(step.depends_on) == 1
            and _single_row(steps.get(step.depends_on[0]), steps)
        }
        if not replace:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        changed.output_steps = list(
            dict.fromkeys(replace.get(key, key) for key in changed.output_steps)
        )
        return changed

    #
    # Devuelve la dependencia que republica un paso de publicación, si corresponde.
    #
    @staticmethod
    def _republished_step(step, used):
        if step.id in used:
            return None

        republishes = (
            _PUBLICATION_STEP.fullmatch(step.id)
            and len(step.depends_on) == 1
            and step.operation in {"aggregate", "calculate", "sort"}
        )
        if republishes:
            return step.depends_on[0]

        cited = PlanQueryService._cited_steps(step) if step.operation == "read" else []
        return cited[0] if len(cited) == 1 else None

    #
    # Devuelve los ids de los pasos cuyos resultados cita un paso.
    #
    @staticmethod
    def _cited_steps(step):
        refs = [item.field for item in [*step.filters, *step.group_by, *step.order_by]]
        refs += [ref for calculation in step.calculations for ref in calculation.inputs]
        return list(dict.fromkeys(ref.step_id for ref in refs if isinstance(ref, ResultRef)))

    #
    # Ordena los pasos por dependencias y rechaza ids repetidos y ciclos.
    #
    @staticmethod
    def _dependency_order(decision):
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

    #
    # Pasa a Python los latest que ordenan por períodos guardados como texto.
    #
    @staticmethod
    def _latest_in_python(decision, periods):
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

    #
    # Excluye los nulos de la variable medida en un latest para elegir la última medición real.
    #
    @staticmethod
    def _measured_latest(decision, variables):
        if decision.status != "plannable" or not variables:
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        modified = False
        for step in changed.steps:
            read = PlanQueryService._read_ancestor(step, steps) if step.operation == "latest" else None
            if read is None:
                continue

            measured = [
                ref for ref in read.columns if ref.column in variables.get(ref.dataset_id, set())
            ]
            constrained = {
                item.field.column for item in read.filters if isinstance(item.field, ColumnRef)
            }
            if len(measured) != 1 or measured[0].column in constrained:
                continue

            read.filters.append(FilterSpec(field=measured[0], operator="is_not_null", values=[]))
            modified = True
        return changed if modified else decision

    #
    # Devuelve el read del que desciende un paso.
    #
    @staticmethod
    def _read_ancestor(step, steps):
        parent = steps.get(step.depends_on[0]) if step.depends_on else None
        while parent is not None and parent.operation != "read":
            parent = steps.get(parent.depends_on[0]) if parent.depends_on else None
        return parent

    #
    # Detecta las columnas TEXT de un dataset cuyos valores representan fechas o períodos.
    #
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
                    TemporalTextColumn(
                        dataset_id=dataset.dataset_id,
                        column=name,
                        complete=len(items) <= _DISTINCT_TEXT_LIMIT,
                        **profile,
                    )
                )
        return found
