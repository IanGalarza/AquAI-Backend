import asyncio
import json
import math
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import NamedTuple

from aquai.application.contracts.planning import (
    MAX_FILTER_VALUES,
    CalculationSpec,
    CartesianVisualizationRequest,
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
    VisualizePlanStep,
)
from aquai.application.contracts.retrieval import DatasetRetrievalResult
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import LLMProviderError, PlannerValidationError
from aquai.application.interfaces.planning import PlannerProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import (
    abbreviates,
    contains_phrase,
    normalize_name,
    normalize_text,
    parsed_number,
    singular_words,
)
from aquai.application.Rules.vocabulary import NUMBER, YEAR_COLUMN_NAMES
from aquai.application.units import counts_organisms
from aquai.application.planning.context import (
    asked_extremes,
    asks_period_span,
    build_planning_context,
    chart_requested,
    extreme_requested,
    latest_requested,
    numeric_comparison,
    period_phrase,
    requested_order,
    requested_periods,
    requested_statistics,
    requested_thresholds,
    statistic_extreme,
    unread_periods,
)
from aquai.application.planning.filterValues import resolve_filter_values
from aquai.application.planning.followUp import (
    asked_chart_type,
    conserved_chart_type,
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
    INCOMPLETE_REQUEST_CODES,
    PYTHON_ONLY_FUNCTIONS,
    UNMET_REQUEST_CODES,
    applies_threshold,
    chain_applies,
    chain_datasets,
    filters_period,
    period_columns,
    period_matches,
    requested_dimensions,
    threshold_groups,
    validate_named_sources,
    validate_operations,
    validate_requested_scope,
    validate_requested_variables,
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
_GAP_HINTS = {"missing_named_source": "Agregá un read de esa fuente y el mismo cálculo que para las demás."}
_ORDER_SUFFIX = "_ordered"
_SAMPLING_DATES = "fechas_de_muestreo_distintas"
_GROUP_COUNT = "cantidad_de_valores"
_UNIT_SCALES = (1, 1e3, 1e-3, 1e6, 1e-6)
_DATE_SPAN = (("min", "primera_fecha"), ("max", "ultima_fecha"))
_DATE_TYPES = {"DATE", "TIMESTAMP"}
_CATEGORY_PREFERENCE = ({"TEXT"}, {"DATE", "TIMESTAMP"})
_MAX_PLAN_STEPS = 12
_TEXT_CALCULATIONS = {"count", "count_distinct"}
_TEXT_COMPARISONS = {"text_measurement_order", "ineligible_calculation"}
_PUBLICATION_STEP = re.compile(r"output_steps?(?:_[a-z0-9]+)?")
_MAX_STEP_ID_LENGTH = 63
_MAX_AXIS_LABEL = 150
_MAX_CHART_TITLE = 200
_DISTINCT_TEXT_LIMIT = 500
_GRAIN_NAMES = {"day": "día", "month": "mes", "year": "año"}
_SERIES_FUNCTIONS = {
    "mean": "Promedio",
    "median": "Mediana",
    "sum": "Suma",
    "min": "Mínimo",
    "max": "Máximo",
    "count": "Cantidad de registros",
    "count_distinct": "Cantidad de valores distintos",
    "stddev": "Desvío estándar",
}
_PERIOD_LITERAL = re.compile(r"(?:1[89]|2\d)\d{2}(?:-\d{1,2}(?:-\d{1,2}(?:[T ][\d:.]+)?)?)?")
_SAMPLE_COUNT = re.compile(
    r"\b(?:cu[aá]nt[oa]s|how\s+many|cantidad\s+de|n[uú]mero\s+de|number\s+of)\s+(?:muestras?|muestreos?|samples?)\b",
    re.IGNORECASE,
)
_DATE_NAME = re.compile(r"(?:fecha|date)(?:_(?:de_)?(?:muestreo|toma|sampling))?")
_MEASURED_FUNCTIONS = {"mean", "median", "sum", "min", "max", "stddev", "quantile"}
_MEASUREMENT_COUNT = re.compile(
    r"\b(?:cu[aá]nt[oa]s|cantidad\s+de|n[uú]mero\s+de|how\s+many|number\s+of)\s+(?:mediciones|medidas|valores|datos|"
    r"lecturas|observaciones|measurements|values|readings|observations)\b",
    re.IGNORECASE,
)
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


class _ThresholdColumns(NamedTuple):
    types: dict
    asked: set
    periods: set


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
    order: str | None = None
    question: str = ""
    value_matches: dict | None = None
    statistic_extreme: bool = False
    resolved_variables: dict | None = None
    chart_form: str | None = None
    limited: bool = False
    dated_only: bool = False
    period_span: bool = False
    reference_request: str | None = None
    context_numbers: frozenset = frozenset()


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
# Devuelve los pasos cuyos resultados usa un paso en sus filtros, agrupaciones, órdenes y cálculos.
#
def _referenced_steps(step):
    refs = [
        *(item.field for item in [*step.filters, *step.group_by, *step.order_by]),
        *(ref for calculation in step.calculations for ref in calculation.inputs),
    ]

    return list(dict.fromkeys(ref.step_id for ref in refs if isinstance(ref, ResultRef)))


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
# Indica si un valor del filtro es el valor pedido, escrito igual, como palabras que lo contienen o abreviado
# («AVENIDA ALTE G BROWN N° 419…» para «almirante brown 419»).
#
def _names_asked(stored, asked) -> bool:
    return stored.casefold() == asked.casefold() or contains_phrase(stored, asked) or abbreviates(stored, asked)


#
# Devuelve la hoja de un dataset («BASE FQ» en «BaseCiano… / BASE FQ / A1:AT99»), o None si el nombre no la trae.
#
def _dataset_sheet(name) -> str | None:
    parts = [part.strip() for part in str(name).split(" / ")]

    return parts[1] if len(parts) >= 3 else None


#
# Devuelve las columnas del catálogo que lee un read, indexadas por su nombre SQL.
#
def _read_columns(read, catalog) -> dict:
    if read is None:
        return {}

    known = {(item.dataset_id, column.sql_name): column for item in catalog for column in item.columns_metadata}

    return {
        ref.column: known[(ref.dataset_id, ref.column)]
        for ref in read.columns
        if (ref.dataset_id, ref.column) in known
    }


#
# Devuelve el nombre del campo al que apunta una referencia de un paso.
#
def _ref_name(ref) -> str:
    return ref.field if ref.kind == "result" else ref.column


#
# Indica si una agrupación es por período: un grano de día, mes o año, o una columna de fecha o de año.
#
def _temporal_grouping(grouping, columns) -> bool:
    if grouping.time_grain != "raw":
        return True

    column = columns.get(_ref_name(grouping.field))
    if column is None:
        return False

    year = column.sql_type == "BIGINT" and singular_words(column.original_name) in YEAR_COLUMN_NAMES

    return column.sql_type in {"DATE", "TIMESTAMP"} or year


#
# Devuelve el rótulo del eje de una agrupación: su columna y, si la hay, el grano temporal.
#
def _grouping_label(grouping, columns) -> str:
    column = columns.get(_ref_name(grouping.field))
    name = column.original_name if column is not None else _ref_name(grouping.field)
    grain = _GRAIN_NAMES.get(grouping.time_grain)

    return f"{name} ({grain})" if grain else name


#
# Devuelve el rótulo del eje de un cálculo: la estadística y la columna que resume.
#
def _calculation_label(calculation, columns) -> str:
    statistic = _SERIES_FUNCTIONS[calculation.function]
    if not calculation.inputs:
        return statistic

    column = columns.get(_ref_name(calculation.inputs[0]))
    name = column.original_name if column is not None else _ref_name(calculation.inputs[0])

    return f"{statistic} de {name}"


#
# Devuelve una constante de filtro como número si el modelo la escribió como texto numérico.
#
def _numeric_constant(value):
    if not isinstance(value, str):
        return value

    number = parsed_number(value)

    return value if number is None else number


#
# Devuelve las variables pedidas que en todos sus datasets se miden contando organismos (cél/mL, ind/mL): «cuántas
# cianobacterias informa la base» pide ese recuento medido, no cuántos registros hay.
#
def _organism_counts(evidence, catalog):
    units = {(dataset.dataset_id, column.sql_name): column.unit for dataset in catalog for column in dataset.columns_metadata}
    counted = {}
    for item in evidence:
        for mention, names in item.variable_matches.items():
            counted.setdefault(mention, []).extend(counts_organisms(units.get((item.dataset_id, name))) for name in names)

    return [mention for mention, flags in counted.items() if flags and all(flags)]


#
# Devuelve los números de la pregunta, de los pedidos citados y de los filtros del análisis que retoma: un umbral que
# sale de ahí está justificado aunque la pregunta actual no lo repita.
#
def _context_numbers(texts, analysis) -> frozenset:
    numbers = set()
    for text in texts:
        for match in NUMBER.finditer(str(text or "")):
            value = parsed_number(match.group())
            if value is not None:
                numbers.add(float(value))
    for operation in analysis.operations if analysis else []:
        for item in operation.get("filters") or []:
            numbers |= {float(value) for value in item.get("values") or [] if isinstance(value, (int, float))}

    return frozenset(numbers)


#
# Indica si un filtro es un límite numérico sobre una variable medida que no se justifica: ni cero, ni un número
# conocido, ni una conversión de unidad de uno (por mil o por millón).
#
def _invented_bound(item, step, steps, measured, numbers) -> bool:
    if item.operator not in {"gt", "ge", "lt", "le"} or len(item.values) != 1:
        return False

    value = item.values[0]
    if not isinstance(value, (int, float)) or value == 0 or not _measured_field(item.field, step, steps, measured):
        return False

    return not any(
        math.isclose(value, number * scale, rel_tol=1e-9) for number in numbers for scale in _UNIT_SCALES
    )


#
# Indica si un campo es una columna medida del catálogo, leída directamente o republicada por un read.
#
def _measured_field(field, step, steps, measured) -> bool:
    if isinstance(field, ColumnRef):
        return (field.dataset_id, field.column) in measured

    origin = steps.get(field.step_id) if isinstance(field, ResultRef) else None
    if origin is None or origin.operation != "read":
        return False

    return any((ref.dataset_id, ref.column) in measured for ref in origin.columns if ref.column == field.field)


#
# Devuelve las columnas de las variables pedidas, con las proyecciones numéricas derivadas de ellas, indexadas por
# dataset y columna.
#
def _asked_columns(catalog, variables):
    return {
        (dataset.dataset_id, column.sql_name)
        for dataset in catalog
        for column in dataset.columns_metadata
        if {column.sql_name, column.derived_from} & set((variables or {}).get(dataset.dataset_id, ()))
    }


#
# Devuelve las columnas de fecha o de año del catálogo, indexadas por dataset y columna.
#
def _period_columns(catalog):
    return {
        (dataset.dataset_id, column.sql_name)
        for dataset in catalog
        for column in dataset.columns_metadata
        if column.sql_type in {"DATE", "TIMESTAMP"}
        or (column.sql_type == "BIGINT" and singular_words(column.original_name) in YEAR_COLUMN_NAMES)
    }


#
# Devuelve el plan reducido a esas salidas y los pasos de los que dependen, con cada dataset limitado a las columnas
# que esos pasos leen o filtran (una columna inventada que sólo usa otra rama no la invalida).
#
def _branch_decision(decision, keys):
    steps = {step.id: step for step in decision.steps}
    needed, pending = set(), list(keys)
    while pending:
        key = pending.pop()
        if key in steps and key not in needed:
            needed.add(key)
            pending.extend(steps[key].depends_on)

    used = {}
    for key in needed:
        refs = [*steps[key].columns, *(item.field for item in steps[key].filters)]
        for ref in refs:
            if isinstance(ref, ColumnRef):
                used.setdefault(ref.dataset_id, set()).add(ref.column)

    branch = decision.model_copy(deep=True)
    branch.steps = [step for step in branch.steps if step.id in needed]
    branch.output_steps = list(dict.fromkeys(key for key in keys if key in needed))
    branch.datasets = [item for item in branch.datasets if item.dataset_id in used]
    for item in branch.datasets:
        item.columns = [column for column in item.columns if column in used[item.dataset_id]]
    return branch


#
# Devuelve la referencia a un cálculo repetido apuntando al que se conserva.
#
def _aliased(ref, aliases):
    if isinstance(ref, ResultRef) and (ref.step_id, ref.field) in aliases:
        return ref.model_copy(update={"field": aliases[(ref.step_id, ref.field)]})
    return ref


#
# Indica si un paso usa un cálculo repetido fuera de otros cálculos (en filtros, agrupaciones, órdenes o gráficos).
#
def _uses_alias(step, aliases) -> bool:
    refs = [*(item.field for item in [*step.filters, *step.group_by, *step.order_by])]
    chart = getattr(step, "visualization", None)
    if chart is not None:
        refs += [ref for ref in (getattr(chart, name, None) for name in ("x", "y", "series")) if ref is not None]
    return any(isinstance(ref, ResultRef) and (ref.step_id, ref.field) in aliases for ref in refs)


#
# Devuelve el paso cuyos datos dibuja un gráfico publicado, o None si la salida no es un gráfico de un solo paso.
#
def _chart_parent(decision, key):
    step = next((item for item in decision.steps if item.id == key), None)
    if step is None or step.operation != "visualize" or len(step.depends_on) != 1:
        return None
    return step.depends_on[0]


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

    return _groups_by(evidence.get("requested_time_grain"), kinds)


#
# Indica si las columnas de período permiten agrupar por la unidad de tiempo pedida: una fecha DATE/TIMESTAMP
# agrupa por cualquiera y una columna de año, sólo por año.
#
def _groups_by(grain, kinds) -> bool:
    return not grain or "date" in kinds or (grain == "year" and "year" in kinds)


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
        reference_request=None,
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

        thresholds = requested_thresholds(data_mentions)
        asked_statistics = requested_statistics(asked, _organism_counts(evidence, catalog))
        statistics = asked_statistics or conserved_statistics(previous_analysis, asked)
        statistics = [] if dated_only else statistics
        chart_required = chart_requested(asked)
        grain = requested_time_grain(asked)
        regrouped = grain or any(item.kind in {"dimension", "date"} for item in own)
        rescoped = not (asked_statistics or grain or any(item.kind == "variable" for item in own))
        chart_form = (asked_chart_type(asked) if chart_required else None) or (
            conserved_chart_type(previous_analysis) if rescoped else None
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

        ungrouped = self._undated_request(context, catalog, periods) if grain else []
        if ungrouped:
            return self._without_time_grain(common, ungrouped, retrieval.warnings)
        unread = [] if dated_only else unread_periods(data_mentions)
        if unread:
            return self._unread_periods(common, unread)

        values = [item.text for item in data_mentions if item.kind == "value"]
        variable_texts = [item.text for item in data_mentions if item.kind == "variable"]
        asks_dimension = not dated_only and (
            bool(retrieval.dimension_mentions)
            or any(item.kind == "dimension" for item in data_mentions)
        )
        variables = {
            item.dataset_id: {name for names in item.variable_matches.values() for name in names}
            for item in evidence
        }
        asked_variables = {item.text for item in data_mentions if item.kind == "variable"}
        resolved = {
            item.dataset_id: variables[item.dataset_id]
            for item in evidence
            if asked_variables and asked_variables <= set(item.variable_matches)
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
            several_rows=asks_dimension
            or limits[0] is not None
            or len(values) > 1
            or requested_order(asked, variable_texts) is not None,
            time_grain=grain,
            grouping=grain or (None if regrouped else conserved_time_grain(previous_analysis)),
            stale=tuple(stale_values(previous_analysis, data_mentions)),
            variables=variables,
            latest_allowed=temporal,
            follow_up=previous_analysis is not None,
            extreme=extreme_requested(asked),
            statistic_extreme=not dated_only and statistic_extreme(asked),
            resolved_variables=resolved,
            chart_form=chart_form,
            extreme_orders=frozenset() if dated_only else frozenset(asked_extremes(asked, named)),
            order=None if dated_only else requested_order(asked, variable_texts),
            limited=limits[0] is not None or bool(limits[1]),
            dated_only=dated_only,
            period_span=asks_period_span(asked, data_mentions),
            reference_request=reference_request,
            context_numbers=_context_numbers([question, asked, *request_history], previous_analysis),
            question=asked,
            value_matches={
                item.dataset_id: {
                    mention: [(match.column, match.values) for match in matches]
                    for mention, matches in item.value_matches.items()
                }
                for item in evidence
            },
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
    # Crea el resultado de cobertura insuficiente cuando ningún candidato puede agrupar por el período pedido,
    # sin pedirle al modelo un plan que la validación rechazaría.
    #
    @staticmethod
    def _without_time_grain(common, missing, retrieval_warnings):
        return PlanningResult(
            **common,
            status="insufficient_coverage",
            llm_status=None,
            reason="Los candidatos no tienen una fecha DATE/TIMESTAMP (o un año, si se agrupa por año) con la que "
            "agrupar por el período pedido; las fechas TEXT no se agrupan.",
            missing_information=missing,
            warnings=list(retrieval_warnings),
        )

    #
    # Crea el resultado que pide precisar un período que nombra un año o un mes pero no se pudo interpretar:
    # calcular sin él respondería por todo el registro como si se hubiera acotado.
    #
    @staticmethod
    def _unread_periods(common, unread):
        listed = ", ".join(f"«{text}»" for text in unread)
        return PlanningResult(
            **common,
            status="needs_clarification",
            llm_status=None,
            reason=f"No se pudo convertir en fechas el período pedido: {listed}.",
            clarification_question=(
                f"No pude interpretar el período {listed} como fechas concretas. ¿Podés indicarlo con meses y "
                "años? Por ejemplo: de septiembre a noviembre de 2021."
            ),
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
    # Pide planes al modelo con reintentos y retroalimentación hasta obtener uno verificado. Un plan válido al que
    # sólo le falta una parte pedida (una fuente nombrada) se reintenta, pero se conserva: si el reintento falla,
    # se publica ése como respuesta parcial en lugar de no responder nada.
    #
    async def _propose(self, context, verify):
        feedback, attempt, model, tokens, rejections = [], 0, None, [None, None], []
        partial = fallback = None
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
                if (kept := partial or fallback) is not None:
                    return replace(kept, attempts=attempt, rejections=rejections)
                return _Proposal(attempt, model, tokens, error=error, rejections=rejections)

            defects = self._plan_defects(verified.issues)
            gaps = [
                issue
                for issue in verified.issues
                if issue.code in INCOMPLETE_REQUEST_CODES and issue.severity == "warning"
            ]
            answerable = self._answerable(verified, defects)
            if any(issue.code in _TEXT_COMPARISONS for issue in defects):
                context.evidence["text_comparison_rejected"] = True
            if defects and attempt >= _MAX_ATTEMPTS:
                current = _Proposal(attempt, model, tokens, verified=answerable) if answerable else None
                if (kept := partial or current or fallback) is not None:
                    rejections.append(f"intento {attempt}: {' '.join(issue.message for issue in defects)}")
                    return replace(kept, attempts=attempt, rejections=rejections)
            if not (defects or gaps) or attempt >= _MAX_ATTEMPTS:
                return _Proposal(attempt, model, tokens, verified=verified, rejections=rejections)
            if not defects:
                partial = _Proposal(attempt, model, tokens, verified=verified)
            elif answerable is not None:
                fallback = _Proposal(attempt, model, tokens, verified=answerable)

            _logger.warning(
                "Planner intento %s con defectos: %s",
                attempt,
                [issue.message for issue in [*defects, *gaps]],
            )
            feedback = [
                *(issue.message for issue in defects),
                *(f"{issue.message} {_GAP_HINTS.get(issue.code, '')}".strip() for issue in gaps),
            ]
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
    # Devuelve el plan publicable sin las partes pedidas que no cubrió, si sólo le faltan esas (el gráfico, una
    # estadística o un umbral): lo calculado se publica como respuesta parcial que avisa qué faltó, en vez de
    # descartarse entero. Si tiene otro defecto, devuelve None.
    #
    @staticmethod
    def _answerable(verified, defects):
        if not defects or any(issue.code not in UNMET_REQUEST_CODES for issue in defects):
            return None

        unmet = {id(issue) for issue in defects}
        issues = [
            issue.model_copy(update={"severity": "warning"}) if id(issue) in unmet else issue
            for issue in verified.issues
        ]
        return verified._replace(issues=issues)

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
        ungrouped = [] if _groups_by(grain, kinds) else [_GRAIN_NAMES[grain]]
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
        decision, unverified = self._verifiable_branches(decision, catalog, periods)
        decision, unjustified = self._unjustified_reference_thresholds(decision, catalog, requested)
        selected = validate_selections(decision, catalog)
        operations, issues = validate_operations(decision, selected, periods)
        decision, impossible = self._viable_branches(decision, issues)
        if impossible:
            selected = validate_selections(decision, catalog)
            operations, issues = validate_operations(decision, selected, periods)
        unverified = [*unverified, *unjustified, *impossible]

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
                    extreme=requested.extreme_orders,
                    dimensions=requested.dimensions,
                    categories=requested.categories,
                    chart=requested.chart_required,
                    time_grain=requested.time_grain,
                    periods=requested.periods,
                    dated_columns=period_columns(selected, periods),
                    order=requested.order,
                    variables=requested.variables,
                )
            )
            issues.extend(validate_text_orderings(operations, requested.variables or {}))
            named = self._named_sources(catalog, source_mentions, requested.question)
            issues.extend(validate_named_sources(operations, decision.output_steps, named))
            if requested.follow_up or requested.resolved_variables:
                checked = requested.variables if requested.follow_up else requested.resolved_variables
                issues.extend(
                    validate_requested_variables(operations, decision.output_steps, checked or {})
                )
            if not requested.latest_allowed:
                issues.extend(
                    validate_unrequested_latest(
                        operations, decision.output_steps, follow_up=requested.follow_up
                    )
                )

        issues.extend(unverified)
        issues.extend(await self._catalog_changes(selected, catalog))
        return _VerifiedPlan(decision, selected, operations, issues, resolutions)

    #
    # Valida el plan y, si una parte no pasa la verificación, conserva las salidas que sí la pasan, cada una con los
    # pasos de los que depende: lo omitido queda como una parte faltante, que se reintenta y, si no, deja la
    # respuesta parcial. Un gráfico que no pasa deja publicados los datos que iba a dibujar («las medias por sitio»),
    # con el aviso de que el gráfico faltó. Si ninguna salida es válida, el error sigue siendo el del plan completo.
    #
    @staticmethod
    def _verifiable_branches(decision, catalog, periods):
        if decision.status != "plannable":
            return decision, []
        try:
            validate_operations(decision, validate_selections(decision, catalog), periods)
            return decision, []
        except PlannerValidationError as error:
            failure = error

        valid, charts = [], []
        for key in decision.output_steps:
            parent = _chart_parent(decision, key)
            if PlanQueryService._verifiable(decision, [key], catalog, periods):
                valid.append(key)
            elif parent is not None and PlanQueryService._verifiable(decision, [parent], catalog, periods):
                valid.append(parent)
                charts.append(key)
        valid = list(dict.fromkeys(valid))
        if not valid or (not charts and len(valid) == len(decision.output_steps)):
            raise failure

        issues = []
        if charts:
            issues.append(
                ValidationIssue(
                    code="missing_requested_chart",
                    severity="warning",
                    message=f"El gráfico no pasó la verificación y se publican sus datos: {failure} Corregí sólo el "
                    "gráfico.",
                )
            )
        if any(key not in valid and key not in charts for key in decision.output_steps):
            issues.append(
                ValidationIssue(
                    code="unverified_branch",
                    severity="warning",
                    message=f"Una parte del plan no pasó la verificación y quedó afuera: {failure} Corregí sólo esa "
                    "parte.",
                )
            )
        return _branch_decision(decision, valid), issues

    #
    # Quita los umbrales numéricos que ocupan el lugar de una referencia documental que no se pudo verificar («cuántos
    # superan los niveles guía de la OMS» sin ningún valor verificado): un número que no surge de la pregunta, del
    # contexto ni de un umbral resuelto, ni es una conversión de unidad de alguno de ellos, sería una referencia
    # inventada. Sin esa referencia pedida no se controla nada: el modelo puede inferir umbrales («pH ácido»),
    # usar los del contexto o calcularlos. Lo quitado se avisa y la respuesta dice que la comparación no se verificó.
    #
    @staticmethod
    def _unjustified_reference_thresholds(decision, catalog, requested):
        if decision.status != "plannable" or not requested.reference_request:
            return decision, []

        numbers = {float(item.value) for item in requested.thresholds} | set(requested.context_numbers)
        measured = {
            key for key, sql_type in _column_types(catalog).items() if sql_type in {"NUMERIC", "BIGINT"}
        } - _period_columns(catalog)
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        removed = []
        for step in changed.steps:
            kept = []
            for item in step.filters:
                if not _invented_bound(item, step, steps, measured, numbers):
                    kept.append(item)
                    continue
                removed.append(f"{_ref_name(item.field)} {item.operator} {item.values[0]}")
                if not any(other.field == item.field and other.operator == "is_not_null" for other in kept):
                    kept.append(FilterSpec(field=item.field, operator="is_not_null", values=[]))
            step.filters = kept
        if not removed:
            return decision, []

        issue = ValidationIssue(
            code="unverified_reference_threshold",
            severity="warning",
            message="Se quitó el filtro "
            + ", ".join(removed)
            + ": la consulta pide un valor de referencia de los documentos que no se pudo verificar, y ese número no "
            "surge de la pregunta, del contexto ni de un umbral resuelto.",
        )
        return changed, [issue]

    #
    # Quita las salidas que exigen valores incompatibles en un mismo campo (ninguna fila puede cumplirlos) cuando otras
    # salidas del plan sí pueden responder: una rama imposible no aporta resultados y no debe bloquear las demás.
    #
    @staticmethod
    def _viable_branches(decision, issues):
        impossible = {issue.step_id for issue in issues if issue.code == "contradictory_filters" and issue.step_id}
        if not impossible or decision.status != "plannable":
            return decision, []

        viable = [
            key
            for key in decision.output_steps
            if not {step.id for step in _branch_decision(decision, [key]).steps} & impossible
        ]
        if not viable or len(viable) == len(decision.output_steps):
            return decision, []

        issue = ValidationIssue(
            code="impossible_branch",
            severity="warning",
            message="Se quitó una parte del plan que exigía valores incompatibles en el mismo campo: "
            + ", ".join(sorted(impossible))
            + ".",
        )
        return _branch_decision(decision, viable), [issue]

    #
    # Indica si las salidas, con los pasos de los que dependen, pasan la verificación.
    #
    @staticmethod
    def _verifiable(decision, keys, catalog, periods) -> bool:
        branch = _branch_decision(decision, keys)
        try:
            validate_operations(branch, validate_selections(branch, catalog), periods)
        except PlannerValidationError:
            return False
        return True

    #
    # Devuelve los datasets de cada fuente que la pregunta nombra, por su nombre o por un alias que el nombre
    # contiene («los datos de ABSA»); las fuentes heredadas de otro turno no cuentan.
    #
    @staticmethod
    def _named_sources(catalog, source_mentions, question):
        asked = [item for item in source_mentions if item and contains_phrase(question, item)]
        named = {}
        for dataset in catalog:
            provenance = dataset.provenance
            sheet = _dataset_sheet(dataset.name)
            names = {
                normalize_name(provenance.source_name),
                normalize_name(provenance.original_filename),
                *([normalize_name(sheet)] if sheet else []),
            }
            for mention in asked:
                if any(contains_phrase(name, mention) for name in names):
                    named.setdefault(mention, set()).add(dataset.dataset_id)

        return named

    #
    # Aplica en orden las reparaciones determinísticas a la decisión del modelo.
    #
    @staticmethod
    def _repaired(decision, catalog, periods, requested):
        decision = PlanQueryService._numeric_constants(decision, catalog)
        decision = PlanQueryService._merged_selections(decision)
        decision = PlanQueryService._read_selections(decision, catalog)
        decision = without_stale_filters(decision, requested.stale)
        decision = PlanQueryService._publication_steps(decision)
        decision = PlanQueryService._dependency_order(decision)
        decision = PlanQueryService._unrequested_charts(decision, requested.chart)
        decision = PlanQueryService._requested_scatter(decision, catalog, requested)
        decision = PlanQueryService._single_row_republications(decision, {"latest"})
        decision = PlanQueryService._unrequested_period_limits(decision, catalog, periods, requested)
        decision = PlanQueryService._unpublished_branches(decision)
        decision = PlanQueryService._parallel_branches(decision, requested.variables, requested.values)
        decision = PlanQueryService._misplaced_values(decision, requested)
        decision = PlanQueryService._omitted_values(decision, requested)
        decision = PlanQueryService._unranked_extremes(decision, catalog, requested)
        decision = PlanQueryService._unordered_records(decision, catalog, requested)
        decision = PlanQueryService._projected_orderings(decision, catalog)
        decision = PlanQueryService._text_measurement_branches(decision, catalog, requested.variables)
        decision = PlanQueryService._result_references(decision)
        decision = PlanQueryService._cross_step_results(decision)
        decision = PlanQueryService._missing_groupings(decision, catalog)
        decision = PlanQueryService._omitted_categories(decision, catalog, requested)
        decision = with_time_grain(decision, requested.grouping, _column_types(catalog))
        decision = PlanQueryService._omitted_dimensions(decision, requested)
        decision = PlanQueryService._grouped_series(decision)
        decision = PlanQueryService._undated_groupings(decision, catalog, requested.time_grain)
        decision = PlanQueryService._omitted_thresholds(
            decision, catalog, requested.thresholds, requested.variables
        )
        decision = PlanQueryService._omitted_periods(decision, catalog, requested.periods, periods)
        if not requested.several_rows:
            decision = PlanQueryService._single_row_republications(decision, {"sort"})
        decision = PlanQueryService._extremes_as_sort(decision, catalog)
        if requested.statistic_extreme:
            decision = PlanQueryService._statistic_extremes(decision)
        decision = PlanQueryService._misdirected_extremes(decision, requested)
        decision = PlanQueryService._sampling_events(decision, catalog, requested.question)
        decision = PlanQueryService._measured_counts(decision, requested.question)
        decision = PlanQueryService._requested_spans(decision, catalog, requested)
        if requested.extreme:
            decision = PlanQueryService._extreme_rankings(decision, catalog)
        decision = PlanQueryService._repeated_rankings(decision)
        decision = PlanQueryService._ranked_group_counts(decision)
        decision = PlanQueryService._degenerate_comparisons(decision)
        decision = PlanQueryService._series_chart(decision, catalog, requested)
        decision = PlanQueryService._measured_latest(decision, requested.variables)
        decision = PlanQueryService._latest_in_python(decision, periods)
        return PlanQueryService._python_only_steps(decision)

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
    # Quita el cálculo final que usa resultados de pasos de los que no depende («la diferencia entre el promedio de
    # una fuente y el de otra»): la gramática no lo admite y rechazarlo descartaba también los resultados que sí se
    # calcularon. Publica esos resultados, que la respuesta presenta lado a lado.
    #
    @staticmethod
    def _cross_step_results(decision):
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        used = {key for step in decision.steps for key in step.depends_on}
        dropped = {}
        for step in decision.steps:
            if step.operation not in {"aggregate", "calculate", "sort"} or step.id in used:
                continue
            foreign = [key for key in _referenced_steps(step) if key in steps and key not in step.depends_on]
            if foreign:
                dropped[step.id] = list(dict.fromkeys([*step.depends_on, *foreign]))
        if not dropped:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in dropped]
        changed.output_steps = list(
            dict.fromkeys(key for output in changed.output_steps for key in dropped.get(output, [output]))
        )
        return changed

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
    def _parallel_branches(decision, variables, mentions=()):
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        consumed = {key for step in decision.steps for key in step.depends_on}
        chains = [
            PlanQueryService._published_chain(steps.get(key), steps) for key in decision.output_steps
        ]
        shapes = {PlanQueryService._parallel_shape(chain, mentions) for chain in chains if chain}
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
                or PlanQueryService._parallel_shape(chain, mentions) not in shapes
                or not measured & {ref.column for ref in source.columns}
            ):
                continue
            forgotten.append(step.id)
            read.add(dataset_id)

        if not forgotten or len(decision.output_steps) + len(forgotten) > _MAX_PLAN_STEPS:
            return decision
        return decision.model_copy(update={"output_steps": [*decision.output_steps, *forgotten]})

    #
    # Repone en una salida el filtro de un valor pedido que el modelo omitió, si en ese dataset el valor figura en
    # una sola columna: se filtra por los valores guardados que lo escriben («PUNTA LARA» para «pta lara»).
    #
    @staticmethod
    def _omitted_values(decision, requested):
        if decision.status != "plannable" or not requested.values or not requested.value_matches:
            return decision

        changed, modified = decision.model_copy(deep=True), False
        steps = {step.id: step for step in changed.steps}
        for key in changed.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            read = chain[-1] if chain and chain[-1].operation == "read" and chain[-1].columns else None
            if read is None:
                continue

            dataset_id = read.columns[0].dataset_id
            filtered = [
                value
                for step in chain
                for predicate in step.filters
                if predicate.operator in {"eq", "ieq", "in"}
                for value in predicate.values
                if isinstance(value, str)
            ]
            for mention in requested.values:
                matches = requested.value_matches.get(dataset_id, {}).get(mention, [])
                kept = any(
                    PlanQueryService._filtered_mention(value, [mention]) == normalize_text(mention)
                    for value in filtered
                )
                if kept or len(matches) != 1 or not 0 < len(matches[0][1]) <= MAX_FILTER_VALUES:
                    continue

                column, values = matches[0]
                field = ColumnRef(kind="column", dataset_id=dataset_id, column=column)
                read.filters.append(FilterSpec(field=field, operator="in", values=list(values)))
                if column not in {ref.column for ref in read.columns}:
                    read.columns.append(field)
                for selection in changed.datasets:
                    if selection.dataset_id == dataset_id and column not in selection.columns:
                        selection.columns.append(column)
                modified = True
        return changed if modified else decision

    #
    # Mueve a la columna correcta el filtro de un valor pedido que el modelo puso en otra columna del mismo dataset
    # («Club de Pesca» en Lugar, cuando la búsqueda lo encontró sólo en Dirección): filtrado allí no encontraría
    # registros. Sólo si la búsqueda halló el valor en una única columna de ese dataset.
    #
    @staticmethod
    def _misplaced_values(decision, requested):
        if decision.status != "plannable" or not requested.values or not requested.value_matches:
            return decision

        changed, modified = decision.model_copy(deep=True), False
        for step in changed.steps:
            for predicate in step.filters if step.operation == "read" else []:
                target = PlanQueryService._value_column(predicate, requested)
                if target is None:
                    continue

                column, values = target
                predicate.field = ColumnRef(kind="column", dataset_id=predicate.field.dataset_id, column=column)
                predicate.operator, predicate.values = "in", list(values)
                if column not in {ref.column for ref in step.columns}:
                    step.columns.append(predicate.field.model_copy())
                for selection in changed.datasets:
                    if selection.dataset_id == predicate.field.dataset_id and column not in selection.columns:
                        selection.columns.append(column)
                modified = True

        return changed if modified else decision

    #
    # Devuelve la única columna donde la búsqueda encontró el valor pedido de un filtro de igualdad, con los valores
    # hallados, si el filtro está en otra columna; None si está bien ubicado o no hay una única alternativa.
    #
    @staticmethod
    def _value_column(predicate, requested):
        texts = [value for value in predicate.values if isinstance(value, str)]
        if predicate.operator not in {"eq", "ieq", "in"} or predicate.field.kind != "column" or not texts:
            return None

        matches = requested.value_matches.get(predicate.field.dataset_id, {})
        asked = [mention for mention in requested.values if any(_names_asked(text, mention) for text in texts)]
        found = [match for mention in asked for match in matches.get(mention, [])]
        columns = {column for column, _values in found}
        if len(columns) != 1 or predicate.field.column in columns:
            return None

        column = next(iter(columns))
        values = [value for match_column, found_values in found if match_column == column for value in found_values]
        if not 0 < len(values) <= MAX_FILTER_VALUES:
            return None

        return column, list(dict.fromkeys(values))

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
    def _parallel_shape(chain, mentions=()):
        values = sorted(
            {
                PlanQueryService._filtered_mention(value, mentions)
                for step in chain
                for predicate in step.filters
                if predicate.operator in {"eq", "ieq", "in"}
                for value in predicate.values
                if isinstance(value, str) and not _PERIOD_LITERAL.fullmatch(value.strip())
            }
        )
        return PlanQueryService._branch_shape(chain), tuple(values)

    #
    # Devuelve la mención pedida que representa un valor de filtro («Balneario Bagliardi» es «Bagliardi»),
    # o el valor mismo: dos ramas que filtran el mismo lugar con etiquetas distintas son paralelas.
    #
    @staticmethod
    def _filtered_mention(value, mentions):
        named = [
            mention
            for mention in mentions
            if contains_phrase(value, mention) or abbreviates(value, mention)
        ]
        return normalize_text(named[0]) if len(named) == 1 else str(value).casefold()

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
    # Agrega el group_by omitido cuando se ordena o grafica por una categoría (texto o año) tras un aggregate
    # que no la agrupa.
    #
    @staticmethod
    def _missing_groupings(decision, catalog):
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        types = _column_types(catalog)
        years = {key for key, kind in period_columns(catalog).items() if kind == "year"}
        modified = False
        for step in changed.steps:
            refs = [item.field for item in step.order_by]
            if step.visualization is not None:
                refs += [getattr(step.visualization, name, None) for name in ("x", "series")]

            for ref in refs:
                target = PlanQueryService._ungrouped_aggregate(ref, steps, types, years)
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
    # Devuelve el aggregate al que le falta agrupar por la categoría citada: una columna de texto o de año.
    #
    @staticmethod
    def _ungrouped_aggregate(ref, steps, types, years):
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
        key = (column.dataset_id, column.column) if column is not None else None
        if key is None or (types.get(key) != "TEXT" and key not in years):
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
    # Agrega a un aggregate ya agrupado (por fecha u otra columna) la dimensión pedida que omitió, cuando su
    # dataset tiene una sola columna para ella: «por año y por sitio» agrupa por las dos.
    #
    @staticmethod
    def _omitted_dimensions(decision, requested):
        if decision.status != "plannable" or not requested.dimensions:
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        modified = False
        for step in changed.steps:
            if step.operation != "aggregate" or not step.group_by or len(step.depends_on) != 1:
                continue

            read = _passthrough_origin(steps.get(step.depends_on[0]), steps)
            if read is None or read.operation != "read" or not read.columns:
                continue

            asked = requested.dimensions.get(read.columns[0].dataset_id, set())
            grouped = {item.field.field for item in step.group_by if isinstance(item.field, ResultRef)}
            if len(asked) != 1 or asked & grouped:
                continue

            column = next(iter(asked))
            PlanQueryService._selected_column(changed, read, column)
            step.group_by.append(
                GroupingSpec(
                    field=ResultRef(kind="result", step_id=step.depends_on[0], field=column),
                    output_name=column,
                )
            )
            modified = True
        return changed if modified else decision

    #
    # Agrega una columna al read y a la selección de su dataset si todavía no la tienen.
    #
    @staticmethod
    def _selected_column(decision, read, column):
        dataset_id = read.columns[0].dataset_id
        if column not in {ref.column for ref in read.columns}:
            read.columns.append(ColumnRef(kind="column", dataset_id=dataset_id, column=column))
        for selection in decision.datasets:
            if selection.dataset_id == dataset_id and column not in selection.columns:
                selection.columns.append(column)

    #
    # Usa como serie de un gráfico de líneas o barras la otra agrupación de su aggregate: con dos agrupaciones y
    # una sola en el eje, cada valor de x tiene varias filas y el gráfico no se puede dibujar.
    #
    @staticmethod
    def _grouped_series(decision):
        if decision.status != "plannable":
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        modified = False
        for step in changed.steps:
            chart = step.visualization
            if chart is None or chart.chart_type not in {"line", "bar"} or chart.series is not None:
                continue

            origin = _passthrough_origin(steps.get(step.depends_on[0]) if step.depends_on else None, steps)
            if origin is None or origin.operation != "aggregate" or not isinstance(chart.x, ResultRef):
                continue

            others = [item.output_name for item in origin.group_by if item.output_name != chart.x.field]
            if len(others) != 1:
                continue

            chart.series = ResultRef(kind="result", step_id=step.depends_on[0], field=others[0])
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
            asked = [value for value in values if any(_names_asked(str(stored), value) for stored in item.values)]
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
    # Aplica los umbrales pedidos que el modelo anunció pero no escribió en el plan, sobre la variable pedida.
    #
    @staticmethod
    def _omitted_thresholds(decision, catalog, thresholds, variables=None):
        if not thresholds or decision.status != "plannable":
            return decision

        required, alternatives = threshold_groups(thresholds)
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        columns = _ThresholdColumns(
            _column_types(catalog), _asked_columns(catalog, variables), _period_columns(catalog)
        )
        modified = False
        for key in changed.output_steps:
            chain = PlanQueryService._published_chain(steps.get(key), steps)
            for threshold in required:
                target = PlanQueryService._threshold_target(chain, threshold, columns)
                if chain_applies(chain, threshold) or target is None:
                    continue

                step, field = target
                step.filters.append(
                    FilterSpec(field=field, operator=threshold.operator, values=[threshold.value])
                )
                modified = True

        for group in alternatives:
            expanded = PlanQueryService._alternative_branches(changed, group, columns)
            if expanded is None:
                return decision
            modified |= expanded

        return changed if modified else decision

    #
    # Deja una salida por alternativa en cada dataset: aplica la que falta, separa las combinadas y copia la
    # rama para las demás. Devuelve None si las copias superan los límites del plan.
    #
    @staticmethod
    def _alternative_branches(decision, group, columns):
        steps = {step.id: step for step in decision.steps}
        chains = [(key, PlanQueryService._published_chain(steps.get(key), steps)) for key in decision.output_steps]
        by_dataset = {}
        for key, chain in chains:
            by_dataset.setdefault(chain_datasets(chain), []).append((key, chain))

        copies, modified = [], False
        for members in by_dataset.values():
            covered = set()
            for key, chain in members:
                applied = [index for index, item in enumerate(group) if chain_applies(chain, item)]
                if not applied:
                    index = next((index for index in range(len(group)) if index not in covered), 0)
                    modified |= PlanQueryService._applied_alternative(chain, None, group[index], columns)
                    applied = [index] if chain_applies(chain, group[index]) else []
                for index in applied[1:]:
                    PlanQueryService._without_threshold(chain, group[index])
                    copies.append((key, chain, group[applied[0]], group[index]))
                    modified = True
                covered.update(applied)

            template = next(((key, chain) for key, chain in members if any(chain_applies(chain, item) for item in group)), None)
            for index, item in enumerate(group):
                if index not in covered and template is not None:
                    source = next(alternative for alternative in group if chain_applies(template[1], alternative))
                    copies.append((template[0], template[1], source, item))
                    modified = True

        return None if not PlanQueryService._copied_alternatives(decision, copies, columns) else modified

    #
    # Aplica una alternativa a una rama: reemplaza la que ya filtra o agrega el filtro donde corresponde.
    #
    @staticmethod
    def _applied_alternative(chain, source, threshold, columns) -> bool:
        if source is not None:
            for step in chain:
                for position, predicate in enumerate(step.filters):
                    if applies_threshold(predicate, source):
                        step.filters[position] = predicate.model_copy(update={"values": [threshold.value]})
                        return True
            return False

        target = PlanQueryService._threshold_target(chain, threshold, columns)
        if target is None:
            return False

        step, field = target
        step.filters.append(FilterSpec(field=field, operator=threshold.operator, values=[threshold.value]))

        return True

    #
    # Quita de una rama el filtro de una alternativa.
    #
    @staticmethod
    def _without_threshold(chain, threshold):
        for step in chain:
            step.filters = [item for item in step.filters if not applies_threshold(item, threshold)]

    #
    # Agrega al plan una copia de rama por alternativa, publicada junto a su original; False si no caben.
    #
    @staticmethod
    def _copied_alternatives(decision, copies, columns) -> bool:
        outputs = list(decision.output_steps)
        for number, (key, chain, source, threshold) in enumerate(copies, 1):
            renamed = {step.id: f"{step.id}_t{number}" for step in chain}
            copied = [_renamed_step(step, renamed) for step in reversed(chain)]
            if not PlanQueryService._applied_alternative(copied, source, threshold, columns):
                return False
            decision.steps.extend(copied)
            outputs.insert(outputs.index(key) + 1 if key in outputs else len(outputs), renamed[key])

        decision.output_steps = outputs
        too_large = (
            len(decision.steps) > _MAX_PLAN_STEPS
            or len(decision.output_steps) > _MAX_PLAN_STEPS
            or any(len(step.id) > _MAX_STEP_ID_LENGTH for step in decision.steps)
        )

        return not too_large

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
    def _threshold_target(chain, threshold, columns):
        if not threshold.compares_statistic:
            read = chain[-1] if chain and chain[-1].operation == "read" else None
            measured = [
                ref
                for ref in (read.columns if read else [])
                if columns.types.get((ref.dataset_id, ref.column)) in {"NUMERIC", "BIGINT"}
            ]
            if len(measured) > 1:
                measured = [
                    ref for ref in measured if (ref.dataset_id, ref.column) in columns.asked
                ] or [ref for ref in measured if (ref.dataset_id, ref.column) not in columns.periods]
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
    # Quita las salidas que repiten un ranking publicado: un sort sin filtros de los mismos registros, por el mismo
    # campo y en el mismo sentido, con un límite que no supera al de otro, sólo muestra filas que ya están en él.
    #
    @staticmethod
    def _repeated_rankings(decision):
        if decision.status != "plannable":
            return decision

        steps = {step.id: step for step in decision.steps}
        rankings = {key: PlanQueryService._ranking(steps.get(key), steps) for key in decision.output_steps}
        kept, repeated = [], set()
        for key in sorted(rankings, key=lambda item: -(rankings[item] or (None, 0))[1]):
            ranking = rankings[key]
            if ranking is not None and any(rankings[other][0] == ranking[0] for other in kept):
                repeated.add(key)
            elif ranking is not None:
                kept.append(key)
        used = {item for step in decision.steps if step.id not in repeated for item in step.depends_on}
        removed = repeated - used
        if not removed:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in removed]
        changed.output_steps = [key for key in changed.output_steps if key not in removed]
        return changed

    #
    # Devuelve qué ordena un sort publicado y con qué límite: sus registros de origen, el campo y el sentido; None
    # si filtra, ordena por más de un campo o no limita sus filas.
    #
    @staticmethod
    def _ranking(step, steps):
        if step is None or step.operation != "sort" or step.limit is None or len(step.order_by) != 1:
            return None

        current = step
        while current is not None and current.operation in {"sort", "latest"}:
            if current.filters or not current.depends_on:
                return None
            current = steps.get(current.depends_on[0])
        ordering = step.order_by[0]
        field = ordering.field.field if isinstance(ordering.field, ResultRef) else ordering.field.column
        return (current.id if current is not None else None, field, ordering.direction), step.limit

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
    # Ordena en el sentido pedido («de mayor a menor») las salidas que publican la variable sin ordenarla:
    # un read de registros recibe el orden; un aggregate agrupado, un sort por su único cálculo.
    #
    @staticmethod
    def _unordered_records(decision, catalog, requested):
        if decision.status != "plannable" or requested.order is None or requested.extreme_orders:
            return decision

        types = _column_types(catalog)
        changed, modified = decision.model_copy(deep=True), False
        steps = {step.id: step for step in changed.steps}
        for index, key in enumerate(changed.output_steps):
            step = steps.get(key)
            if PlanQueryService._reversed_order(step, steps, requested.variables, requested.order):
                modified = True
                continue

            measured = PlanQueryService._extreme_column(step, requested.variables, types)
            if measured is not None:
                step.order_by = [OrderingSpec(field=measured, direction=requested.order)]
                modified = True
                continue

            sort = PlanQueryService._ordered_aggregate(step, steps, requested)
            if sort is not None and len(changed.steps) < _MAX_PLAN_STEPS:
                changed.steps.append(sort)
                changed.output_steps[index], modified = sort.id, True
        return changed if modified else decision

    #
    # Invierte el orden de la salida que ordena por la variable pedida, sin límite, en el sentido contrario.
    #
    @staticmethod
    def _reversed_order(step, steps, variables, order, *, limited=False):
        if step is not None and step.operation == "visualize" and step.depends_on:
            step = steps.get(step.depends_on[0])
        if step is None or not step.order_by or step.operation == "latest":
            return False
        if step.limit is not None and not limited:
            return False

        first = step.order_by[0]
        name = first.field.field if isinstance(first.field, ResultRef) else first.field.column
        asked = {column for columns in (variables or {}).values() for column in columns}
        if name not in asked or first.direction == order:
            return False

        first.direction = order
        return True

    #
    # Cuando se preguntan muestras, agrega a cada conteo de filas las fechas de muestreo distintas: en una tabla
    # con una fila por taxón o parámetro, las filas no son muestras y la respuesta debe poder distinguirlos.
    #
    @staticmethod
    def _sampling_events(decision, catalog, question):
        if decision.status != "plannable" or _SAMPLE_COUNT.search(question) is None:
            return decision

        types = _column_types(catalog)
        changed, modified = decision.model_copy(deep=True), False
        steps = {step.id: step for step in changed.steps}
        for key in changed.output_steps:
            count = _passthrough_origin(steps.get(key), steps)
            read = steps.get(count.depends_on[0]) if count is not None and count.depends_on else None
            date = PlanQueryService._sampling_date(count, read, types)
            if date is None:
                continue

            if date not in {ref.column for ref in read.columns}:
                read.columns.append(ColumnRef(kind="column", dataset_id=read.columns[0].dataset_id, column=date))
            for selection in changed.datasets:
                if selection.dataset_id == read.columns[0].dataset_id and date not in selection.columns:
                    selection.columns.append(date)
            count.calculations.append(
                CalculationSpec(
                    function="count_distinct",
                    inputs=[ResultRef(kind="result", step_id=read.id, field=date)],
                    output_name=_SAMPLING_DATES,
                )
            )
            modified = True
        return changed if modified else decision

    #
    # Quita los cálculos repetidos de un mismo paso (dos promedios de la misma columna con nombres distintos) y la
    # diferencia o el cociente entre ellos, que siempre darían 0 o 1: se publica el paso que ya calcula el valor, no
    # una comparación sin sentido. Si otra parte del plan usa el cálculo repetido de otra forma, no se toca.
    #
    @staticmethod
    def _degenerate_comparisons(decision):
        if decision.status != "plannable":
            return decision

        changed = decision.model_copy(deep=True)
        aliases = {}
        for step in changed.steps:
            if step.operation != "aggregate":
                continue
            seen, kept = {}, []
            for item in step.calculations:
                key = (
                    item.function,
                    json.dumps([ref.model_dump(mode="json") for ref in item.inputs], sort_keys=True),
                    item.quantile,
                    item.ddof,
                    item.null_policy,
                )
                if key in seen:
                    aliases[(step.id, item.output_name)] = seen[key]
                else:
                    seen[key] = item.output_name
                    kept.append(item)
            step.calculations = kept
        if not aliases or any(_uses_alias(step, aliases) for step in changed.steps):
            return decision

        replaced = {}
        for step in changed.steps:
            for item in step.calculations:
                item.inputs = [_aliased(ref, aliases) for ref in item.inputs]
            step.calculations = [
                item
                for item in step.calculations
                if not (item.function in {"difference", "ratio"} and len(item.inputs) == 2 and item.inputs[0] == item.inputs[1])
            ]
            if step.operation == "calculate" and not step.calculations:
                if len(step.depends_on) != 1:
                    return decision
                replaced[step.id] = step.depends_on[0]
        if any(key in replaced for step in changed.steps for key in step.depends_on):
            return decision

        changed.steps = [step for step in changed.steps if step.id not in replaced]
        changed.output_steps = list(dict.fromkeys(replaced.get(key, key) for key in changed.output_steps))
        return changed

    #
    # Agrega la cantidad de valores a un cálculo por grupo que se ordena para elegir los primeros («qué sitio tiene la
    # mayor clorofila promedio»): un promedio de un solo valor no es comparable con el de un grupo con muchas
    # mediciones, y sin esa cantidad la respuesta no puede advertirlo.
    #
    @staticmethod
    def _ranked_group_counts(decision):
        if decision.status != "plannable":
            return decision

        ranked = {
            step.depends_on[0] for step in decision.steps if step.operation == "sort" and step.limit and step.depends_on
        }
        changed, modified = decision.model_copy(deep=True), False
        for step in changed.steps:
            if step.id not in ranked or step.operation != "aggregate" or not step.group_by:
                continue
            if any(item.function in {"count", "count_distinct"} for item in step.calculations):
                continue

            sources = [ref for item in step.calculations if item.function in _MEASURED_FUNCTIONS for ref in item.inputs]
            if len({ref.model_dump_json() for ref in sources}) != 1:
                continue

            names = {item.output_name for item in step.calculations}
            name = next(
                candidate
                for candidate in (_GROUP_COUNT, *(f"{_GROUP_COUNT}_{number}" for number in range(2, 10)))
                if candidate not in names
            )
            step.calculations.append(
                CalculationSpec(function="count", inputs=[sources[0].model_copy()], output_name=name)
            )
            modified = True
        return changed if modified else decision

    #
    # Cuenta los valores de la variable, no las filas, cuando la consulta pide cuántas mediciones o valores hay y el
    # mismo cálculo resume una sola variable («la turbiedad media… con cuántas mediciones»): el conteo de filas
    # incluye registros sin esa medición, que el promedio dejó afuera, y no sería su n.
    #
    @staticmethod
    def _measured_counts(decision, question):
        if decision.status != "plannable" or _MEASUREMENT_COUNT.search(question) is None:
            return decision

        changed, modified = decision.model_copy(deep=True), False
        for step in changed.steps:
            if step.operation != "aggregate":
                continue

            sources = [ref for item in step.calculations if item.function in _MEASURED_FUNCTIONS for ref in item.inputs]
            if len({ref.model_dump_json() for ref in sources}) != 1:
                continue

            for item in step.calculations:
                if item.function == "count" and not item.inputs:
                    item.inputs = [sources[0].model_copy()]
                    modified = True

        return changed if modified else decision

    #
    # Agrega la primera y la última fecha a un cálculo sin agrupar cuando la pregunta pide el período que cubren los
    # datos («indicá promedios, cantidad de mediciones y períodos»): el mínimo y el máximo de la única columna
    # DATE/TIMESTAMP del dataset que lee. Un dataset sin esa columna queda sin período y la respuesta lo avisa.
    #
    @staticmethod
    def _requested_spans(decision, catalog, requested):
        if decision.status != "plannable" or not requested.period_span or requested.periods or requested.time_grain:
            return decision

        types = _column_types(catalog)
        changed, modified = decision.model_copy(deep=True), False
        steps = {step.id: step for step in changed.steps}
        for step in changed.steps:
            read = steps.get(step.depends_on[0]) if step.depends_on else None
            ungrouped = step.operation == "aggregate" and not step.group_by
            if not ungrouped or read is None or read.operation != "read" or not read.columns:
                continue

            dataset_id = read.columns[0].dataset_id
            dated = [
                name for (owner, name), sql_type in types.items() if owner == dataset_id and sql_type in _DATE_TYPES
            ]
            names = {item.output_name for item in step.calculations}
            spanned = {ref.field for item in step.calculations if item.function in {"min", "max"} for ref in item.inputs}
            if len(dated) != 1 or dated[0] in spanned or names & {name for _, name in _DATE_SPAN}:
                continue

            PlanQueryService._with_read_column(changed, read, dated[0])
            step.calculations += [
                CalculationSpec(
                    function=function,
                    inputs=[ResultRef(kind="result", step_id=read.id, field=dated[0])],
                    output_name=name,
                )
                for function, name in _DATE_SPAN
            ]
            modified = True
        return changed if modified else decision

    #
    # Agrega una columna al read y a la selección de su dataset, si todavía no la leían.
    #
    @staticmethod
    def _with_read_column(decision, read, column):
        dataset_id = read.columns[0].dataset_id
        if column not in {ref.column for ref in read.columns}:
            read.columns.append(ColumnRef(kind="column", dataset_id=dataset_id, column=column))
        for selection in decision.datasets:
            if selection.dataset_id == dataset_id and column not in selection.columns:
                selection.columns.append(column)

    #
    # Devuelve la columna de fecha con que contar muestreos bajo un conteo de filas sin agrupar por fecha.
    #
    @staticmethod
    def _sampling_date(count, read, types):
        counts_rows = (
            count is not None
            and count.operation == "aggregate"
            and [item.function for item in count.calculations] == ["count"]
            and not count.calculations[0].inputs
        )
        if not counts_rows or read is None or read.operation != "read" or not read.columns:
            return None

        dataset_id = read.columns[0].dataset_id
        dated = [
            name
            for (owner, name), sql_type in types.items()
            if owner == dataset_id and (sql_type in {"DATE", "TIMESTAMP"} or _DATE_NAME.fullmatch(name))
        ]
        grouped = {getattr(item.field, "field", None) for item in count.group_by}
        if len(dated) != 1 or dated[0] in grouped:
            return None

        return dated[0]

    #
    # Invierte el sort de un extremo que ordena la variable al revés de lo pedido («la turbiedad mínima»).
    #
    @staticmethod
    def _misdirected_extremes(decision, requested):
        if decision.status != "plannable" or len(requested.extreme_orders) != 1:
            return decision

        order = next(iter(requested.extreme_orders))
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        reversed_steps = [
            PlanQueryService._reversed_order(steps.get(key), steps, requested.variables, order, limited=True)
            for key in changed.output_steps
        ]
        return changed if any(reversed_steps) else decision

    #
    # Cuando se pide el extremo de una estadística («qué sitio tuvo el pH promedio más alto»), ordena por el cálculo
    # del aggregate agrupado los sort que ordenaban los registros leídos de esa misma variable: el registro más
    # alto no es el grupo de promedio más alto.
    #
    @staticmethod
    def _statistic_extremes(decision):
        if decision.status != "plannable":
            return decision

        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        modified = False
        for step in changed.steps:
            if step.operation != "sort" or step.filters or len(step.order_by) != 1:
                continue

            ordering = step.order_by[0]
            target = PlanQueryService._calculated_ordering(ordering.field, steps)
            if target is None:
                continue

            aggregate, calculation = target
            step.depends_on = [aggregate.id]
            step.order_by = [
                OrderingSpec(
                    field=ResultRef(kind="result", step_id=aggregate.id, field=calculation.output_name),
                    direction=ordering.direction,
                )
            ]
            modified = True
        return changed if modified else decision

    #
    # Devuelve el único aggregate agrupado, y su cálculo, que resume la variable leída por la que ordena un sort.
    #
    @staticmethod
    def _calculated_ordering(field, steps):
        origin = _passthrough_origin(steps.get(field.step_id), steps) if isinstance(field, ResultRef) else None
        if origin is None or origin.operation != "read":
            return None

        found = [
            (step, calculation)
            for step in steps.values()
            if step.operation == "aggregate"
            and step.group_by
            and len(step.depends_on) == 1
            and _passthrough_origin(steps.get(step.depends_on[0]), steps) is origin
            for calculation in step.calculations
            if calculation.function not in _TEXT_CALCULATIONS
            and any(isinstance(item, ResultRef) and item.field == field.field for item in calculation.inputs)
        ]
        return found[0] if len(found) == 1 else None

    #
    # Devuelve el sort por el único cálculo de un aggregate agrupado que mide la variable pedida.
    #
    @staticmethod
    def _ordered_aggregate(step, steps, requested):
        unordered = step is not None and step.operation == "aggregate" and step.group_by and not step.order_by
        if not unordered or len(step.calculations) != 1 or step.id in {
            key for other in steps.values() for key in other.depends_on
        }:
            return None

        calculation = step.calculations[0]
        read = _passthrough_origin(steps.get(step.depends_on[0]), steps)
        asked = (requested.variables or {}).get(read.columns[0].dataset_id, set()) if read and read.columns else set()
        operands = {item.field if isinstance(item, ResultRef) else item.column for item in calculation.inputs}
        if calculation.function in _TEXT_CALCULATIONS or not operands & asked:
            return None

        return SortPlanStep(
            id=f"{step.id[: _MAX_STEP_ID_LENGTH - len(_ORDER_SUFFIX)]}{_ORDER_SUFFIX}",
            tool="sql",
            operation="sort",
            depends_on=[step.id],
            order_by=[
                OrderingSpec(
                    field=ResultRef(kind="result", step_id=step.id, field=calculation.output_name),
                    direction=requested.order,
                )
            ],
            purpose="Ordenar el resultado en el sentido que pide la consulta.",
        )

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
    # Quita los gráficos no pedidos y publica su entrada en su lugar, salvo que esa entrada ya alimente otro
    # resultado publicado: los registros leídos para un promedio no se publican por un gráfico que nadie pidió.
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

        published = set(decision.output_steps) - charts.keys()
        feeding = {key for step in decision.steps if step.id in published for key in step.depends_on}
        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in charts]
        changed.output_steps = list(
            dict.fromkeys(
                charts.get(key, key)
                for key in changed.output_steps
                if key not in charts or charts[key] not in feeding
            )
        )
        return changed

    #
    # Agrega la dispersión pedida, o conservada por un seguimiento que sólo cambia el alcance, cuando el plan lee
    # las dos variables pedidas pero no publicó ningún gráfico: cada registro es un punto con una variable en cada
    # eje, sin agregar nada.
    #
    @staticmethod
    def _requested_scatter(decision, catalog, requested):
        charted = any(step.operation == "visualize" for step in decision.steps)
        if decision.status != "plannable" or requested.chart_form != "scatter" or charted:
            return decision

        columns = {(item.dataset_id, column.sql_name): column for item in catalog for column in item.columns_metadata}
        for read in (step for step in decision.steps if step.operation == "read" and step.columns):
            variables = (requested.variables or {}).get(read.columns[0].dataset_id, set())
            axes = [
                columns[(ref.dataset_id, ref.column)]
                for ref in read.columns
                if ref.column in variables and columns.get((ref.dataset_id, ref.column)) is not None
                and columns[(ref.dataset_id, ref.column)].numeric_eligible
            ]
            if len(axes) == 2:
                return PlanQueryService._with_scatter(decision, read, *axes)

        return decision

    #
    # Devuelve el plan con un gráfico de dispersión de dos columnas de un read, publicado como salida.
    #
    @staticmethod
    def _with_scatter(decision, read, x, y):
        changed = decision.model_copy(deep=True)
        steps = {step.id: step for step in changed.steps}
        key = next(name for index in range(len(steps) + 2) if (name := f"visualize_scatter_{index}") not in steps)
        x_label, y_label = (column.original_name[:_MAX_AXIS_LABEL] for column in (x, y))
        chart = VisualizePlanStep(
            id=key,
            tool="python",
            operation="visualize",
            depends_on=[read.id],
            visualization=CartesianVisualizationRequest(
                chart_type="scatter",
                title=f"{y_label} según {x_label}"[:_MAX_CHART_TITLE],
                x=ResultRef(kind="result", step_id=read.id, field=x.sql_name),
                y=ResultRef(kind="result", step_id=read.id, field=y.sql_name),
                x_label=x_label,
                y_label=y_label,
            ),
            purpose="Graficar la dispersión pedida entre las dos variables leídas.",
        )
        changed.steps.append(chart)
        changed.output_steps = [*changed.output_steps, key]
        return changed

    #
    # Agrega un gráfico de líneas a una serie temporal que la consulta no pidió graficar: un resultado agrupado por
    # día, mes, año o fecha se lee mejor como evolución que como tabla. Sólo si es la única salida, con un único
    # cálculo y sin otro gráfico; una segunda agrupación queda como serie.
    #
    @staticmethod
    def _series_chart(decision, catalog, requested):
        charted = any(step.operation == "visualize" for step in decision.steps)
        asked = requested.chart or requested.dated_only
        if decision.status != "plannable" or asked or charted or len(decision.output_steps) != 1:
            return decision

        steps = {step.id: step for step in decision.steps}
        output = steps.get(decision.output_steps[0])
        if output is None or output.operation != "aggregate" or len(output.calculations) != 1:
            return decision
        if output.calculations[0].function not in _SERIES_FUNCTIONS:
            return decision

        columns = _read_columns(PlanQueryService._read_ancestor(output, steps), catalog)
        temporal = [item for item in output.group_by if _temporal_grouping(item, columns)]
        others = [item for item in output.group_by if item not in temporal]
        if len(temporal) != 1 or len(others) > 1:
            return decision

        return PlanQueryService._with_series_chart(decision, output, temporal[0], others, columns)

    #
    # Devuelve el plan con un gráfico de líneas del cálculo agrupado por período, publicado como salida.
    #
    @staticmethod
    def _with_series_chart(decision, output, temporal, others, columns):
        changed = decision.model_copy(deep=True)
        used = {step.id for step in changed.steps}
        key = next(name for index in range(len(used) + 2) if (name := f"visualize_series_{index}") not in used)
        calculation = output.calculations[0]
        x_label = _grouping_label(temporal, columns)[:_MAX_AXIS_LABEL]
        y_label = _calculation_label(calculation, columns)[:_MAX_AXIS_LABEL]
        series = ResultRef(kind="result", step_id=output.id, field=others[0].output_name) if others else None
        chart = VisualizePlanStep(
            id=key,
            tool="python",
            operation="visualize",
            depends_on=[output.id],
            visualization=CartesianVisualizationRequest(
                chart_type="line",
                title=f"{y_label} por {x_label}"[:_MAX_CHART_TITLE],
                x=ResultRef(kind="result", step_id=output.id, field=temporal.output_name),
                y=ResultRef(kind="result", step_id=output.id, field=calculation.output_name),
                series=series,
                x_label=x_label,
                y_label=y_label,
            ),
            purpose="Mostrar la evolución del resultado agrupado por período.",
        )
        changed.steps.append(chart)
        changed.output_steps = [*changed.output_steps, key]

        return changed

    #
    # Convierte en número las constantes que el modelo escribió como texto en filtros sobre columnas numéricas
    # («20.000», «25 °C»): la validación rechazaría el plan por la forma del número y no por lo que calcula. Un
    # texto que no empieza con un número queda como está y la validación lo sigue rechazando.
    #
    @staticmethod
    def _numeric_constants(decision, catalog):
        if decision.status != "plannable":
            return decision

        types = _column_types(catalog)
        changed = decision.model_copy(deep=True)
        for step in changed.steps:
            for item in getattr(step, "filters", None) or []:
                numeric = item.field.kind == "column" and types.get((item.field.dataset_id, item.field.column)) in {
                    "NUMERIC",
                    "BIGINT",
                }
                if numeric:
                    item.values = [_numeric_constant(value) for value in item.values]

        return changed

    #
    # Quita el sort final que ordena por un año o una fecha con un tope que la consulta no pidió: el modelo agrega
    # «el más reciente» a «durante 2005» y, si el período abarca varios años, el tope dejaría sólo el último sin
    # decirlo. No se toca si la consulta pide lo más reciente, un extremo, un orden o una cantidad de registros.
    #
    @staticmethod
    def _unrequested_period_limits(decision, catalog, periods, requested):
        asked = requested.latest_allowed or requested.extreme or requested.order or requested.limited
        if decision.status != "plannable" or asked:
            return decision

        dated = _period_columns(catalog) | {(item.dataset_id, item.column) for item in periods}
        steps = {step.id: step for step in decision.steps}
        used = {key for step in decision.steps for key in step.depends_on}
        replace = {
            step.id: step.depends_on[0]
            for step in decision.steps
            if step.operation == "sort"
            and step.limit
            and step.id not in used
            and len(step.depends_on) == 1
            and not step.filters
            and PlanQueryService._orders_by_period(step, steps, dated)
        }
        if not replace:
            return decision

        changed = decision.model_copy(deep=True)
        changed.steps = [step for step in changed.steps if step.id not in replace]
        changed.output_steps = list(dict.fromkeys(replace.get(key, key) for key in changed.output_steps))

        return changed

    #
    # Indica si un paso ordena primero por una columna de año o de fecha de su lectura.
    #
    @staticmethod
    def _orders_by_period(step, steps, dated) -> bool:
        if not step.order_by:
            return False

        ref = step.order_by[0].field
        name = ref.field if ref.kind == "result" else ref.column
        read = PlanQueryService._read_ancestor(step, steps)

        return read is not None and any(
            column.column == name and (column.dataset_id, column.column) in dated for column in read.columns
        )

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
    # Pasa a Python los pasos que el modelo dejó en SQL pero sólo Python calcula (mediana, desvío, cuantiles,
    # gráficos) y los que dependen de ellos, porque un paso SQL no lee resultados de Python: el error se corrige
    # sin volver a planificar.
    #
    @staticmethod
    def _python_only_steps(decision):
        if decision.status != "plannable":
            return decision

        python = {
            step.id
            for step in decision.steps
            if step.tool == "python"
            or step.operation == "visualize"
            or any(item.function in PYTHON_ONLY_FUNCTIONS for item in step.calculations)
        }
        for step in decision.steps:
            if step.operation != "read" and set(step.depends_on) & python:
                python.add(step.id)
        if all(step.tool == "python" for step in decision.steps if step.id in python):
            return decision

        changed = decision.model_copy(deep=True)
        for step in changed.steps:
            if step.id in python:
                step.tool = "python"
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
