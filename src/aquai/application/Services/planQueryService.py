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
    build_planning_context,
    chart_requested,
    extreme_requested,
    latest_requested,
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
    validate_unrequested_latest,
)

_logger = logging.getLogger(__name__)
# Sólo estas incidencias requieren una decisión del usuario; el resto son defectos del plan.
_USER_DECISIONS = {"ambiguous_filter_value"}
_MAX_ATTEMPTS = 2
_CATEGORY_LENGTH = 80
# Puestos que se publican al pedir un extremo («el valor máximo»): el extremo, sus empates y su contexto.
_EXTREME_ROWS = 5
# Una categoría omitida en group_by se busca primero entre los textos y después entre las fechas.
_CATEGORY_PREFERENCE = ({"TEXT"}, {"DATE", "TIMESTAMP"})
# Pasos y salidas que admite un plan (PlannableDecision).
_MAX_PLAN_STEPS = 12
# Cálculos que admiten texto: contar registros o valores distintos.
_TEXT_CALCULATIONS = {"count", "count_distinct"}
# Paso que el modelo llama como el campo output_steps para publicar otro.
_PUBLICATION_STEP = re.compile(r"output_steps?(?:_[a-z0-9]+)?")
# Un id copiado por período («read_1_p2») debe seguir siendo un identificador válido.
_MAX_STEP_ID_LENGTH = 63
# Valores distintos que se leen de una columna TEXT al perfilar períodos (distinct_text_values devuelve uno más).
_DISTINCT_TEXT_LIMIT = 500
_GRAIN_NAMES = {"day": "día", "month": "mes", "year": "año"}
_PERIOD_LITERAL = re.compile(r"(?:1[89]|2\d)\d{2}(?:-\d{1,2}(?:-\d{1,2}(?:[T ][\d:.]+)?)?)?")
# Errores frecuentes del modelo y cómo corregirlos; el texto de la validación sólo describe el síntoma.
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
    """Lo que la consulta pide explícitamente y todo plan debe conservar."""

    thresholds: list
    periods: list
    statistics: list
    dimensions: dict
    chart: bool
    chart_required: bool
    values: list
    categories: bool
    # Una dimensión, una cantidad o varios valores piden varias filas: una única fila ordenada delata un
    # group_by omitido y no se trata como republicación.
    several_rows: bool
    # Agrupación temporal pedida ahora («por mes»), que se exige, y la que el plan debe usar al agrupar una
    # fecha, que también puede venir del resultado citado.
    time_grain: str | None = None
    grouping: str | None = None
    # Valores que filtraban el resultado citado y la consulta vigente reemplazó o quitó.
    stale: tuple = ()
    # Columnas de las variables pedidas por dataset, según la evidencia de recuperación.
    variables: dict | None = None
    # Si la consulta, su historial o el resultado citado piden recencia; y si continúa un análisis sin ella.
    latest_allowed: bool = True
    follow_up: bool = False
    # La consulta pide el extremo de una variable («el valor máximo»), no todos sus registros ordenados.
    extreme: bool = False


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
    # Motivo de cada intento rechazado, para la respuesta guardada y no sólo el log.
    rejections: list[str] = field(default_factory=list)


def _passthrough_origin(step, steps):
    """sort y latest conservan los campos de su entrada: el paso que los produjo es el primero que no lo es."""
    while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
        step = steps.get(step.depends_on[0])
    return step


def _single_row(step, steps):
    """Un aggregate sin agrupar devuelve una única fila; sort y latest la conservan."""
    origin = _passthrough_origin(step, steps)
    return origin is not None and origin.operation == "aggregate" and not origin.group_by


def _period_values(values) -> bool:
    """Valores de filtro que nombran períodos: un año («2021», 2021) o una fecha ISO («2021-03», «2021-03-15»)."""
    return bool(values) and all(_PERIOD_LITERAL.fullmatch(str(value).strip()) for value in values)


def _period_filters(kind, sql_type, period, values) -> list[tuple[str, list]]:
    """Filtros que acotan una columna a un período; un extremo None queda sin filtro. Una fecha TEXT se acota con sus
    valores dentro del período, nunca con un rango de texto."""
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


def _renamed_step(step, renamed):
    """Copia de un paso con otro id, que depende de las copias de sus pasos previos y cita sus resultados."""
    data = _renamed_references(step.model_dump(mode="json"), renamed)
    data["id"] = renamed[step.id]
    data["depends_on"] = [renamed.get(key, key) for key in step.depends_on]
    return type(step).model_validate_json(json.dumps(data))


def _renamed_references(value, renamed):
    if isinstance(value, list):
        return [_renamed_references(item, renamed) for item in value]
    if not isinstance(value, dict):
        return value

    copy = {key: _renamed_references(item, renamed) for key, item in value.items()}
    if copy.get("kind") == "result" and copy.get("step_id") in renamed:
        copy["step_id"] = renamed[copy["step_id"]]
    return copy


def _column_types(catalog):
    return {
        (dataset.dataset_id, column.sql_name): column.sql_type
        for dataset in catalog
        for column in dataset.columns_metadata
    }


def _registers(dataset, observed, mentions, numeric) -> bool:
    """El dataset tiene columnas de cada variable y dimensión pedidas y registros de cada valor pedido; si se pide
    calcular una estadística, cada variable necesita una columna numérica o su proyección numérica."""
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


def _dated(dataset, evidence, text_periods) -> bool:
    """El dataset tiene con qué acotar los períodos pedidos y, para agrupar por día, mes o año, una fecha real."""
    kinds = set(period_columns([dataset], text_periods).values())
    if evidence.get("requested_periods") and not kinds:
        return False

    return not evidence.get("requested_time_grain") or "date" in kinds


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
        conserved_mentions=(),
        requested_period=None,
    ) -> PlanningResult:
        question = PlanningInput(question=question).question
        own = [item for item in data_mentions if item not in conserved_mentions]
        data_mentions = self._planning_mentions(data_mentions, retrieval, conserved_mentions)
        dated_only = requested_period is not None
        if dated_only:
            # Se piden las fechas del resultado citado, no otro cálculo: ni su agrupación ni su estadística se
            # exigen (las operaciones repetidas ya las conservan), tampoco al revalidar antes de ejecutar.
            data_mentions = [item for item in data_mentions if item.kind != "dimension"]
        limits = requested_limit(data_mentions), conserved_limits(data_mentions, previous_analysis)
        if question != retrieval.question.strip():
            raise PlannerValidationError(
                "La pregunta no coincide con la utilizada por el Retriever."
            )

        catalog = await self._verified_catalog(retrieval.candidates)
        asked = original_question or question
        require_latest = latest_requested(asked)
        # Un seguimiento («¿Y de qué año es?») hereda la selección temporal sin exigirla.
        previous = previous_operations(previous_analysis)
        temporal = (
            require_latest
            or "latest" in previous
            or any(latest_requested(item) for item in request_history)
            or dated_only
        )
        dates = requested_periods(data_mentions)
        # Un período pedido también se acota con fechas guardadas como texto, si sus valores las nombran.
        catalog, excluded, periods = await self._temporal_scope(
            catalog, temporal or bool(dates), require_latest
        )
        evidence = await self._with_categories(
            self._catalog_evidence(retrieval.evidence, catalog), catalog
        )
        if dated_only:
            evidence = [item.model_copy(update={"dimension_matches": {}}) for item in evidence]

        thresholds, asked_statistics = requested_thresholds(data_mentions), requested_statistics(asked)
        # Un seguimiento que no nombra otra estadística conserva la del resultado citado.
        statistics = asked_statistics or conserved_statistics(previous_analysis, asked)
        statistics = [] if dated_only else statistics
        # Sólo la pregunta actual exige un gráfico; uno pedido antes se conserva sin exigirse.
        chart_required = chart_requested(asked)
        # Lo mismo con la agrupación temporal: otra dimensión pedida ahora la reemplaza.
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
        # Un gráfico pedido antes («graficalo») se conserva al cambiar sólo el sitio o la variable.
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
            variables={
                item.dataset_id: {name for names in item.variable_matches.values() for name in names}
                for item in evidence
            },
            latest_allowed=temporal,
            follow_up=previous_analysis is not None,
            extreme=extreme_requested(asked),
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

    @staticmethod
    def _planning_mentions(data_mentions, retrieval, conserved=()):
        # Las menciones que el Retriever identificó como columnas no son filtros obligatorios, ni lo es un valor
        # conservado de una columna por la que ahora se agrupa («¿Y por sitio?» después de Punta Lara).
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
                "Las fuentes revisadas no tienen una columna de fecha o período interpretable para determinar "
                "la medición más reciente."
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
            missing_information=["No encontré fuentes disponibles relacionadas con esta consulta."],
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

    async def _reuse(self, decision, verify):
        """El plan del resultado citado se revalida como cualquier propuesta; con defectos, planifica el modelo."""
        try:
            verified = await verify(decision)
        except PlannerValidationError as error:
            _logger.warning("No se reutilizó el plan anterior: %s", error)
            return None

        if verified.decision.status != "plannable" or self._plan_defects(verified.issues):
            return None
        return _Proposal(0, None, [None, None], verified=verified)

    async def _period_proposal(self, analysis, catalog, periods, verify):
        """«¿Y de qué año es?» no se planifica de nuevo: se repiten las operaciones del resultado citado si publican
        fechas, o se leen la primera y la última fecha de sus registros. El modelo tendía a inventar otra consulta
        (la medición más reciente) cuyas fechas no son las del resultado."""
        repeated = repeated_decision(analysis, catalog)
        proposal = await self._reuse(repeated, verify) if repeated else None
        if self._dates_published(proposal, periods):
            return proposal

        spanned = period_span_decision(analysis, catalog)
        return await self._reuse(spanned, verify) if spanned else None

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
            if not defects or attempt >= _MAX_ATTEMPTS:
                return _Proposal(attempt, model, tokens, verified=verified, rejections=rejections)

            _logger.warning(
                "Planner intento %s con defectos: %s",
                attempt,
                [issue.message for issue in defects],
            )
            feedback = [issue.message for issue in defects]
            rejections.append(f"intento {attempt}: {' '.join(feedback)}")

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
            reason=f"El modelo no produjo un plan verificable en {attempt} intentos: "
            + " | ".join(proposal.rejections or [str(error)]),
            missing_information=[],
            warnings=list(retrieval_warnings),
        )

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
            # Sólo un filtro del plan sobre un valor sin registros prueba su ausencia; un valor que el plan no filtró
            # puede ser un recorte impreciso del Router («ese sitio»).
            filtered = any(issue.code == "filter_value_not_found" for issue in issues)
            missing = self._coverage_gaps(absent=absent if filtered else [], undated=undated)
            decision = decision.model_copy(update={"missing_information": missing})
        elif status == "insufficient_coverage":
            # El modelo declaró la falta de cobertura y su texto no llega al usuario. Si un candidato registra todo lo
            # pedido, su negativa es un fallo del plan; si no, sólo se agregan las fechas que faltan, que no dependen
            # de cómo el Router recortó las menciones.
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
    def _undated_periods(issues, defects, selected, context, text_periods):
        """Períodos pedidos que ningún dataset elegido puede acotar porque no tiene fechas ni años, si son lo único
        que impidió el plan: falta información, no es un defecto del modelo."""
        blocking = {issue.code for issue in issues if issue.severity == "clarification"}
        if (
            not defects
            or blocking != {"missing_requested_period"}
            or period_columns(selected, text_periods)
        ):
            return []

        return list(context.evidence.get("requested_periods", []))

    @staticmethod
    def _covers_request(context, catalog, text_periods):
        """Evidencia positiva de cobertura: un candidato revisado tiene columnas de las variables y dimensiones
        pedidas, registros de los valores pedidos y fechas para acotar los períodos o agrupar por tiempo. Una
        mención que no coincide no prueba una carencia: puede ser un recorte impreciso del Router."""
        evidence = context.evidence
        mentions = evidence.get("query_mentions", [])
        if not any(item["kind"] in {"variable", "value"} for item in mentions):
            return False

        observed = {item["dataset_id"]: item for item in evidence.get("observed_evidence", [])}
        numeric = any(
            function not in _TEXT_CALCULATIONS for function in evidence.get("requested_statistics", [])
        )
        return any(
            _registers(dataset, observed.get(str(dataset.dataset_id)), mentions, numeric)
            and _dated(dataset, evidence, text_periods)
            for dataset in catalog
        )

    @classmethod
    def _undated_request(cls, context, catalog, text_periods):
        """Fechas que ningún candidato revisado tiene para acotar los períodos pedidos o agrupar por tiempo."""
        evidence = context.evidence
        kinds = set(period_columns(catalog, text_periods).values())
        undated = [] if kinds else evidence.get("requested_periods", [])
        grain = evidence.get("requested_time_grain")
        ungrouped = [_GRAIN_NAMES[grain]] if grain and "date" not in kinds else []
        return cls._coverage_gaps(undated=undated, ungrouped=ungrouped)

    @staticmethod
    def _coverage_gaps(absent=(), undated=(), ungrouped=()):
        """Carencias verificadas como enunciados para el usuario: ausencia de registros y falta de fechas."""
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

    @staticmethod
    def _claimed_coverage_reason(decision, covered):
        """Trazabilidad de la cobertura insuficiente que declaró el modelo; su texto no llega a la respuesta."""
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
                    dimensions=requested.dimensions,
                    categories=requested.categories,
                    chart=requested.chart_required,
                    time_grain=requested.time_grain,
                    periods=requested.periods,
                    dated_columns=period_columns(selected, periods),
                )
            )
            if not requested.latest_allowed:
                issues.extend(
                    validate_unrequested_latest(
                        operations, decision.output_steps, follow_up=requested.follow_up
                    )
                )

        issues.extend(await self._catalog_changes(selected, catalog))
        return _VerifiedPlan(decision, selected, operations, issues, resolutions)

    @staticmethod
    def _repaired(decision, catalog, periods, requested):
        # Primero se quitan los pasos inventados; después se completan los omitidos.
        decision = PlanQueryService._merged_selections(decision)
        decision = PlanQueryService._read_selections(decision, catalog)
        decision = without_stale_filters(decision, requested.stale)
        # Antes de exigir ids únicos: el modelo repite pasos «output_steps», uno por rama, sólo para publicar.
        decision = PlanQueryService._publication_steps(decision)
        decision = PlanQueryService._dependency_order(decision)
        decision = PlanQueryService._unrequested_charts(decision, requested.chart)
        decision = PlanQueryService._single_row_republications(decision, {"latest"})
        decision = PlanQueryService._unpublished_branches(decision)
        decision = PlanQueryService._parallel_branches(decision, requested.variables)
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
    def _read_selections(decision, catalog):
        """datasets[].columns sólo repite lo que leen los read: se completa con las columnas reales que leen.

        Una columna inexistente que algún read usa se conserva para que la validación la rechace."""
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
        # Un dataset que ningún read lee no aporta nada al plan; la respuesta informa los que no se analizaron.
        if read:
            changed.datasets = [item for item in changed.datasets if item.dataset_id in read]
        return changed if changed != decision else decision

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
    def _parallel_branches(decision, variables):
        """Una rama que calcula lo mismo que una salida publicada, sobre otro dataset y con una variable pedida, y quedó
        fuera de output_steps es una salida olvidada («aggregate_2» junto a «aggregate_1»): se publica, y las
        reparaciones siguientes la acotan como a las demás. No se publica si sus operaciones difieren, si lee un
        dataset ya publicado o si no mide ninguna variable pedida."""
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

    @staticmethod
    def _text_measurement_branches(decision, catalog, variables):
        """Una rama que calcula u ordena una medición guardada como texto sin representación numérica no puede
        responder (un promedio de «N-NO3» con valores como «<0,001»): se quita si quedan otras ramas, y la respuesta
        explica por qué ese archivo no se incluyó."""
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

    @staticmethod
    def _computed_fields(chain):
        """Campos que la rama usa como números: entradas de cálculos que no son conteos y criterios de orden."""
        refs = [
            ref
            for step in chain
            for calculation in step.calculations
            if calculation.function not in _TEXT_CALCULATIONS
            for ref in calculation.inputs
        ]
        refs += [item.field for step in chain if step.operation == "sort" for item in step.order_by]
        return {ref.field if isinstance(ref, ResultRef) else ref.column for ref in refs}

    @staticmethod
    def _parallel_shape(chain):
        """Operaciones y valores filtrados de una rama: otra que filtre otro sitio no la duplica sobre otro dataset."""
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

    @staticmethod
    def _branch_shape(chain):
        """Operaciones de una rama sin sus filtros ni nombres de columnas: cálculos y agrupaciones temporales."""
        return tuple(
            (
                step.operation,
                tuple(sorted(item.function for item in step.calculations)),
                tuple(item.time_grain for item in step.group_by),
            )
            for step in chain
        )

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
    def _omitted_categories(decision, catalog, requested):
        """Un aggregate sin agrupar devuelve una única fila: ordenarla o comparar varios valores con ella no tiene
        sentido. La dimensión preguntada, la columna filtrada con varios valores pedidos («entre Punta Lara y
        Balandra») o, si se ordena por el cálculo, la única categoría sin uso del read es el group_by omitido.

        Con varias candidatas no se elige."""
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

    @staticmethod
    def _omitted_category(unused, preferred, ranked):
        """La primera preferencia presente entre las columnas sin uso; si no hay y se ordena por el cálculo,
        la única categoría del tipo preferido."""
        present = [columns for columns in preferred if unused.keys() & columns]
        chosen = [name for name in unused if name in present[0]] if present else []
        for kinds in _CATEGORY_PREFERENCE if ranked and not chosen else ():
            chosen = [name for name, kind in unused.items() if kind in kinds]
            if chosen:
                break
        return chosen[0] if len(chosen) == 1 else None

    @staticmethod
    def _compared_columns(read, values):
        """Columnas filtradas con dos o más de los valores pedidos: la consulta los compara entre sí."""
        compared = set()
        for item in read.filters:
            stored = {str(value).casefold() for value in item.values}
            asked = [value for value in values if value.casefold() in stored]
            if isinstance(item.field, ColumnRef) and item.operator == "in" and len(asked) > 1:
                compared.add(item.field.column)
        return compared

    @staticmethod
    def _unused_categories(aggregate, read, types):
        """Columnas de texto o fecha del read que no acotan a un único valor ni se calculan, con su tipo."""
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

    @staticmethod
    def _ranks_calculation(aggregate, ordered_steps, steps):
        """Algún sort ordena o filtra por un cálculo de ese aggregate: espera varias filas."""
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

    @staticmethod
    def _omitted_thresholds(decision, catalog, thresholds):
        """Un umbral pedido que el modelo anunció pero no escribió (un sort «filter_1» sin filtros) se aplica.

        Como la cantidad pedida, el operador y el número provienen de la consulta, nunca del modelo. Si compara
        una estadística, filtra el único cálculo del aggregate en el sort que lo publica; si no, la única
        variable numérica del read. Sin un destino inequívoco, la validación pide corregir el plan."""
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

    @staticmethod
    def _omitted_periods(decision, catalog, requested, text_periods):
        """«Durante 2021» acota la fecha. Cada rama se acota exactamente al período pedido con la única columna de
        fecha de su dataset; sin ella, con su único año numérico si el período abarca años enteros, o con su única
        fecha guardada como texto, eligiendo los valores que caen dentro del período («3/2013», nunca
        «12/2012-01/2013» para 2013). Los filtros de período del modelo se reemplazan: suele escribirlos sin una fecha
        ISO o sobre otra columna («fecha = 2021»). Un extremo abierto («desde 2020») sólo acota ese lado.

        Con varios períodos («2019 y 2021») una rama que ya acota exactamente uno se conserva; las demás se copian una
        vez por cada período que ninguna rama equivalente responde.

        Una rama cuyo dataset no tiene con qué acotar el período no puede responder por él: se quita si quedan otras
        ramas, y la respuesta informa que ese archivo no se incluyó."""
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

    @staticmethod
    def _covered_period(chain, dated, requested):
        """Índice del único período pedido que la rama ya acota exactamente; None si no acota ninguno o varios."""
        applied = filters_period([predicate for step in chain for predicate in step.filters], dated)
        matched = [index for index, period in enumerate(requested) if period_matches(applied, period)]
        return matched[0] if len(matched) == 1 else None

    @staticmethod
    def _period_copies(decision, copied, kept, periods, keys, types):
        """Una salida por período pedido. Una rama que ya acota uno se conserva; para cada período que ninguna rama
        equivalente responde, una de ellas se copia (ids «_p1», «_p2») y la copia se acota a ese período. Ramas iguales
        se copian una sola vez y los pasos que comparten (un read con dos cálculos, un gráfico y su agregado),
        también. Las ramas sin período propio se quitan. None si el plan resultante excede lo que admite un plan."""
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
            # Primero se quitan los filtros de período de todas las copias: un read compartido se acota una sola vez.
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

    @staticmethod
    def _branch_signature(chain, keys, types):
        """Una rama sin sus filtros de período: dataset, operaciones y los demás filtros."""
        names = PlanQueryService._period_names(chain, keys)
        scope = sorted(
            (repr(predicate.field), predicate.operator, repr(predicate.values))
            for step in chain
            for predicate in step.filters
            if not PlanQueryService._period_filter(predicate, keys, names, types)
        )
        return chain[-1].columns[0].dataset_id, PlanQueryService._branch_shape(chain), tuple(scope)

    @staticmethod
    def _without_period_filters(decision, chain, periods, types):
        """Quita los filtros con que la rama acota su período: los de una columna de fecha, año o período TEXT o de una
        fecha agrupada, y los que comparan otra columna TEXT o inexistente con años o fechas. Un umbral sobre una
        medición no se toca. Una columna inexistente que sólo servía a esos filtros deja de leerse."""
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

    @staticmethod
    def _period_names(chain, periods):
        """Nombres con que los pasos posteriores citan una fecha, un año o una fecha agrupada."""
        return {column for _, column in periods} | {
            item.output_name for step in chain for item in step.group_by if item.time_grain != "raw"
        }

    @staticmethod
    def _period_filter(predicate, periods, names, types):
        """El filtro acota el período: una columna de fecha, año o período TEXT, o años o fechas comparados con otra
        columna TEXT o inexistente. Un umbral sobre una medición no lo es."""
        ref = predicate.field
        if not isinstance(ref, ColumnRef):
            return ref.field in names

        key = (ref.dataset_id, ref.column)
        return key in periods or (types.get(key) in {None, "TEXT"} and _period_values(predicate.values))

    @staticmethod
    def _undated_groupings(decision, catalog, grain):
        """«Promedio por mes»: una rama que calcula sobre un dataset sin columna DATE/TIMESTAMP no puede agruparse por
        ese período. Se quita si quedan otras ramas, y la respuesta informa que ese archivo no se incluyó."""
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

    @staticmethod
    def _period_target(dataset_id, dated, period, texts):
        """Columna con que el dataset acota el período: su única fecha; si no, su único año (sólo años enteros); si no,
        su única fecha TEXT con los valores que caen dentro (al menos uno y no más de los que admite un filtro)."""
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

    @staticmethod
    def _without_outputs(decision, removed):
        """Quita salidas y los pasos y datasets que sólo ellas usaban."""
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

    @staticmethod
    def _published_chain(step, steps):
        """Pasos que producen una salida, desde ella hasta su read; un gráfico publica su entrada."""
        chain = []
        while step is not None:
            chain.append(step)
            step = steps.get(step.depends_on[0]) if len(step.depends_on) == 1 else None
        return chain

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
    def _extreme_rankings(decision, catalog):
        """«El valor máximo de Cond» se responde con el extremo y sus empates: un sort publicado de los registros por
        una variable medida, sin límite, publicaría todos (y una tabla extensa no se transfiere completa). Se
        limita a los primeros puestos conservando empates, como indica la gramática del plan; una cantidad pedida
        lo reemplaza después."""
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

        # Los registros leídos que alimentan otra salida son su entrada, no un resultado pedido.
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

    @staticmethod
    def _ranked_records(step, steps, types):
        """Id del sort sin límite que ordena registros leídos por una variable medida y publica (o grafica)."""
        if step is not None and step.operation == "visualize" and step.depends_on:
            step = steps.get(step.depends_on[0])
        if step is None or step.operation != "sort" or step.limit is not None:
            return None

        read = _passthrough_origin(steps.get(step.depends_on[0]), steps)
        field = step.order_by[0].field
        name = field.field if isinstance(field, ResultRef) else field.column
        measured = read is not None and read.operation == "read"
        return step.id if measured and PlanQueryService._is_measured(name, read, types) else None

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
        la dependencia y se elimina el paso, siempre que ningún otro paso lo use. Lo mismo un
        read que, en lugar de leer, cita los campos de un único paso previo («output_1»).
        """
        if decision.status != "plannable":
            return decision

        used = {key for step in decision.steps for key in step.depends_on}
        # Un mismo id repetido («output_steps» por cada rama) publica cada una de sus dependencias.
        replace = {}
        for step in decision.steps:
            source = PlanQueryService._republished_step(step, used)
            if source is not None:
                replace.setdefault(step.id, []).append(source)
        if not replace:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        # Un paso «output_steps» que la lista no nombra también publica su dependencia, si ninguna salida ya la usa:
        # es la salida que el modelo quiso declarar (un read que alimenta un cálculo publicado no lo es).
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

    @staticmethod
    def _unrequested_charts(decision, requested):
        """Un gráfico que la consulta no pidió es una operación inventada: se publica su entrada en su lugar."""
        if requested or decision.status != "plannable":
            return decision

        charts = {
            step.id: step.depends_on[0]
            for step in decision.steps
            if step.operation == "visualize" and step.depends_on
        }
        used = {key for step in decision.steps for key in step.depends_on}
        if not charts or charts.keys() & used:
            return decision  # Un paso que usa el gráfico como tabla lo rechaza la validación.

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in charts]
        changed.output_steps = list(
            dict.fromkeys(charts.get(key, key) for key in changed.output_steps)
        )
        return changed

    @staticmethod
    def _single_row_republications(decision, operations):
        """Un latest o un sort final sin filtros sobre una única fila (un aggregate sin agrupar) sólo la republica.

        El modelo los agrega como paso de publicación, ordenados por el propio promedio o por una fecha que ese
        resultado ya no expone; se publica su entrada. Si se pidió recencia, la validación sigue exigiendo latest."""
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

    @staticmethod
    def _cited_steps(step):
        """Pasos cuyos resultados cita el paso en filtros, orden o cálculos."""
        refs = [item.field for item in [*step.filters, *step.group_by, *step.order_by]]
        refs += [ref for calculation in step.calculations for ref in calculation.inputs]
        return list(dict.fromkeys(ref.step_id for ref in refs if isinstance(ref, ResultRef)))

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
    def _measured_latest(decision, variables):
        """«La última medición de pH» es la fecha más reciente con pH registrado: si el read de un latest lee una
        única columna de la variable pedida sin acotarla, se excluyen sus nulos antes de elegir la fecha.

        Con varias variables leídas no se elige cuál exigir."""
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
                    TemporalTextColumn(
                        dataset_id=dataset.dataset_id,
                        column=name,
                        complete=len(items) <= _DISTINCT_TEXT_LIMIT,
                        **profile,
                    )
                )
        return found
