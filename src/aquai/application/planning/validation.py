from datetime import date, datetime, timedelta
from difflib import get_close_matches
from typing import NamedTuple
from uuid import UUID

from aquai.application.contracts.planning import (
    ColumnRef,
    DatasetSelection,
    OutputField,
    PlannableDecision,
    PlannerDecision,
    PlanningResult,
    RequestedPeriod,
    RequestedThreshold,
    SelectedDataset,
    ValidatedStep,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.errors import PlannerValidationError
from aquai.application.matching import contains_phrase, singular_words
from aquai.application.planning.temporalText import period_bounds
from aquai.application.Rules.vocabulary import YEAR_COLUMN_NAMES
from aquai.application.units import comparable_unit, presentation_unit_status

_EQUALITY_OPERATORS = {"eq", "ieq", "in"}
_RANGE_OPERATORS = {"lt", "le", "gt", "ge"}
_NUMERIC_TYPES = {"BIGINT", "NUMERIC"}
_TEMPORAL_TYPES = {"DATE", "TIMESTAMP"}
# Orden de día y mes demostrado por una columna TEXT de períodos (None: no hay fechas D/M/AAAA ambiguas).
_TEXT_PERIOD_KINDS = {None: "text", True: "text-dmy", False: "text-mdy"}
# Palabras de una columna de fecha inventada («fecha», «anio», «date»), en singular.
_DATE_WORDS = {"fecha", "date", "ano", "anio", "year", "mes", "month", "dia", "day"}
_ISO_PARSERS = {"DATE": date.fromisoformat, "TIMESTAMP": datetime.fromisoformat}
_COMBINE_OPERATIONS = {"join", "append"}
_PASSTHROUGH_OPERATIONS = {"sort", "latest"}
_GRAIN_NAMES = {"day": "día", "month": "mes", "year": "año"}
_AGGREGATE_FUNCTIONS = {
    "count",
    "count_distinct",
    "sum",
    "mean",
    "min",
    "max",
    "median",
    "stddev",
    "quantile",
}
_PYTHON_ONLY_FUNCTIONS = {"median", "stddev", "quantile"}
_COUNT_FUNCTIONS = {"count", "count_distinct"}
# Una estadística pedida por su nombre puede calcularse con cualquiera de estas funciones.
_STATISTIC_FUNCTIONS = {"count": _COUNT_FUNCTIONS}
_EXTREME_FUNCTIONS = {"min", "max"}
_AUTHORIZED_METADATA = (
    "source_id",
    "semantic_point_id",
    "physical_table_name",
    "row_count",
    "columns_metadata",
    "provenance",
    "warnings",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlannerValidationError(message)


def _is_numeric(field: OutputField) -> bool:
    return field.numeric_eligible and field.sql_type in _NUMERIC_TYPES


def _equality_text_values(step) -> set[str]:
    values = set()

    for predicate in step.filters:
        if predicate.operator in _EQUALITY_OPERATORS:
            values.update(value for value in predicate.values if isinstance(value, str))

    return values


def _keeps_value(scope, value) -> bool:
    # Una resolución a valores almacenados («5Punta piedras») conserva la mención pedida.
    return any(
        item.casefold() == value.casefold() or contains_phrase(item, value) for item in scope
    )


def validate_temporal_scope(operations, output_steps, *, require_latest=False, requested_values=()):
    """El alcance temporal se revalida también antes de ejecutar y publicar."""
    steps = {step.id: step for step in operations}
    relevant = _reachable_step_ids(steps, output_steps)
    targets = [step for step in operations if step.id in relevant and step.operation == "latest"]
    issues = []

    if require_latest and not targets:
        issues.append(
            ValidationIssue(
                code="missing_latest_selection",
                severity="clarification",
                message="La consulta pide recencia, pero el plan no selecciona la fecha máxima conservando sus mediciones empatadas.",
            )
        )

    scopes = [_latest_scope(target, steps) for target in targets]
    issues.extend(_unscoped_latest_issues(targets, scopes, requested_values))

    if not requested_values or targets:
        return issues

    # Sin recencia: cada salida publicada debe acotarse a alguno de los valores pedidos y
    # cada valor pedido debe aparecer en alguna salida (p. ej., un sitio nuevo tras una aclaración).
    outputs = [(key, _output_scope(steps[key], steps)) for key in output_steps if key in steps]
    issues.extend(_unscoped_output_issues(outputs, requested_values))

    return issues


def validate_unrequested_latest(operations, output_steps, *, follow_up=False) -> list[ValidationIssue]:
    """Un latest que la consulta no pidió restringe el resultado a la última fecha sin que la respuesta lo diga:
    un promedio pasa a ser el de un solo día. Publicado directamente se lee como «la medición más reciente», salvo
    en un seguimiento de un resultado que no lo usaba («¿Y el pH?» después de un máximo), que cambia lo pedido."""
    steps = {step.id: step for step in operations}
    relevant = _reachable_step_ids(steps, output_steps)
    calculated = {
        key
        for step in operations
        if step.id in relevant and step.operation in {"aggregate", "calculate"}
        for key in _reachable_step_ids(steps, step.depends_on)
    }

    return [
        ValidationIssue(
            code="unrequested_latest",
            severity="clarification",
            step_id=step.id,
            message=f"La consulta no pide la medición más reciente, pero el paso {step.id} (latest) restringe el "
            "resultado a la última fecha: quitalo y calculá sobre todas las fechas del alcance pedido.",
        )
        for step in operations
        if step.operation == "latest"
        and step.id in relevant
        and (follow_up or step.id in calculated)
    ]


def _reachable_step_ids(steps, output_steps) -> set[str]:
    relevant, visit = set(), list(output_steps)

    while visit:
        key = visit.pop()
        if key in relevant or key not in steps:
            continue
        relevant.add(key)
        visit.extend(steps[key].depends_on)

    return relevant


def _latest_scope(target, steps) -> set[str]:
    values, visit = set(), [target]

    while visit:
        step = visit.pop()
        values.update(_equality_text_values(step))
        visit.extend(steps[key] for key in step.depends_on)

    return values


def _output_scope(output, steps) -> set[str]:
    values, visit, seen = set(), [output], set()

    while visit:
        step = visit.pop()
        if step.id in seen:
            continue
        seen.add(step.id)
        values.update(_equality_text_values(step))
        visit.extend(steps[parent] for parent in step.depends_on if parent in steps)

    return values


def _unscoped_latest_issues(targets, scopes, requested_values) -> list[ValidationIssue]:
    issues = []

    for target, scope in zip(targets, scopes):
        if requested_values and not any(_keeps_value(scope, value) for value in requested_values):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    step_id=target.id,
                    message="La selección temporal omite los valores de filtro solicitados por el usuario.",
                )
            )

    for value in requested_values:
        if targets and not any(_keeps_value(scope, value) for scope in scopes):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    message=f"La selección temporal no conserva el filtro solicitado para «{value}».",
                )
            )

    return issues


def _unscoped_output_issues(outputs, requested_values) -> list[ValidationIssue]:
    issues = []

    for key, scope in outputs:
        if not any(_keeps_value(scope, value) for value in requested_values):
            listed = ", ".join(f"«{value}»" for value in requested_values)
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    step_id=key,
                    message=f"La salida {key} no se acota a ninguno de los valores pedidos: {listed}.",
                )
            )

    for value in requested_values:
        if outputs and not any(_keeps_value(scope, value) for _, scope in outputs):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    message=f"Ninguna salida filtra por «{value}», que la consulta pide.",
                )
            )

    return issues


def validate_requested_scope(
    operations,
    output_steps,
    *,
    thresholds=(),
    statistics=(),
    dimensions=None,
    categories=False,
    chart=False,
    time_grain=None,
    periods=(),
    dated_columns=None,
) -> list[ValidationIssue]:
    """Cada umbral, período, estadística, dimensión, agrupación temporal y gráfico pedidos deben llegar a una salida
    publicada; dated_columns son las columnas de fecha o año de los datasets (period_columns).

    También se revalida antes de ejecutar: un plan que pierde un filtro pedido no responde la consulta."""
    steps = {step.id: step for step in operations}
    relevant = _reachable_step_ids(steps, output_steps)
    published = [step for step in operations if step.id in relevant]
    issues = [
        *_threshold_issues(steps, output_steps, thresholds),
        *_period_issues(steps, output_steps, periods, dated_columns or {}),
        *_statistic_issues(published, statistics),
    ]

    if chart and not any(step.operation == "visualize" for step in published):
        issues.append(
            ValidationIssue(
                code="missing_requested_chart",
                severity="clarification",
                message="La consulta pide un gráfico, pero el plan no publica ningún visualize: agregá uno "
                "que dependa del paso que ya responde.",
            )
        )

    if time_grain:
        issues.extend(_time_grain_issues(steps, output_steps, time_grain))

    # Contar una categoría («cuántos sitios») no la enumera.
    if categories and "count" not in statistics:
        issues.extend(_dimension_issues(steps, output_steps, dimensions or {}))

    return issues


def period_columns(datasets, text_periods=()) -> dict[tuple[UUID, str], str]:
    """Columnas que ubican cada registro en el tiempo: fechas DATE/TIMESTAMP («date»), años numéricos («year») y
    columnas TEXT cuyos valores nombran períodos verificados («text», «text-dmy» o «text-mdy» según el orden de día y
    mes que demuestra la columna). Una columna TEXT sólo cuenta si se leyeron todos sus valores."""
    columns = {
        (dataset.dataset_id, column.sql_name): (
            "date" if column.sql_type in _TEMPORAL_TYPES else "year"
        )
        for dataset in datasets
        for column in dataset.columns_metadata
        if column.sql_type in _TEMPORAL_TYPES
        or (
            column.sql_type == "BIGINT"
            and singular_words(column.original_name) in YEAR_COLUMN_NAMES
        )
    }
    ids = {dataset.dataset_id for dataset in datasets}
    columns |= {
        (item.dataset_id, item.column): _TEXT_PERIOD_KINDS[item.day_first]
        for item in text_periods
        if item.dataset_id in ids and item.values and item.complete
    }
    return columns


def text_period_day_first(kind) -> bool | None:
    return {value: key for key, value in _TEXT_PERIOD_KINDS.items()}[kind]


class _Applied(NamedTuple):
    """Cómo acota su fecha una rama: extremos (None si queda abierto) y los períodos que nombran sus valores TEXT."""

    start: date | None
    end: date | None
    named: tuple = ()


def _applied_period(chain, steps, columns) -> _Applied | None:
    """Intervalo al que los filtros de una rama acotan su fecha o su año; None si no la acotan."""
    return _period_of(
        (predicate, _period_kind(_filtered_field(step, predicate, steps), columns))
        for step in chain
        for predicate in step.filters
    )


def filters_period(filters, columns) -> _Applied | None:
    """Período al que acotan filtros sobre columnas, antes de validar los pasos: sirve para reparar un plan."""
    return _period_of(
        (predicate, columns.get((predicate.field.dataset_id, predicate.field.column)))
        for predicate in filters
        if isinstance(predicate.field, ColumnRef)
    )


def period_matches(applied, period) -> bool:
    return _bounds_period(applied, period)


def _period_of(pairs) -> _Applied | None:
    lower, upper, named = [], [], []
    for predicate, kind in pairs:
        if kind is None:
            continue
        if kind.startswith("text"):
            if predicate.operator in {"eq", "in"}:
                named += [
                    bounds for value in predicate.values if (bounds := _period_value(value, kind))
                ]
            continue

        low, high = _predicate_bounds(predicate, kind)
        lower += low
        upper += high

    if not (lower or upper or named):
        return None
    return _Applied(max(lower) if lower else None, min(upper) if upper else None, tuple(named))


def _filtered_field(step, predicate, steps) -> OutputField | None:
    ref = predicate.field
    if isinstance(ref, ColumnRef):
        return next((item for item in step.output_fields if item.name == ref.column), None)

    source = steps.get(ref.step_id)
    return next((item for item in source.output_fields if item.name == ref.field), None)


def _period_kind(field, columns) -> str | None:
    if field is None:
        return None
    if field.sql_type in _TEMPORAL_TYPES:
        return "date"
    if len(field.sources) != 1:
        return None

    source = field.sources[0]
    kind = columns.get((source.dataset_id, source.column))
    # Un entero que proviene de una fecha es su año agrupado (time_grain=year).
    if field.sql_type == "BIGINT":
        return "year" if kind else None
    return kind if field.sql_type == "TEXT" and kind and kind.startswith("text") else None


def _predicate_bounds(predicate, kind) -> tuple[list[date], list[date]]:
    """Días extremos que admite un filtro: «< 2022-01-01» termina el 31/12/2021 y «> 2020» empieza en 2021."""
    values = [
        parsed for value in predicate.values if (parsed := _period_value(value, kind)) is not None
    ]
    if not values:
        return [], []

    first = min(start for start, _ in values)
    last = max(end for _, end in values)
    day = timedelta(days=1)
    bounds = {
        "eq": ([first], [last]),
        "in": ([first], [last]),
        "ge": ([first], []),
        "gt": ([last + day], []),
        "le": ([], [last]),
        "lt": ([], [first - day]),
    }
    return bounds.get(predicate.operator, ([], []))


def _period_value(value, kind) -> tuple[date, date] | None:
    """Primer y último día que nombra un valor de filtro: un año entero, el día de una fecha o el período que nombra
    un texto («3/2013», «10-11/2007»)."""
    if kind == "year":
        plausible = type(value) is int and 1000 <= value <= 9999
        return (date(value, 1, 1), date(value, 12, 31)) if plausible else None
    if not isinstance(value, str):
        return None
    if kind.startswith("text"):
        bounds = period_bounds(value, text_period_day_first(kind))
        return bounds[:2] if bounds else None

    try:
        day = datetime.fromisoformat(value).date()
    except ValueError:
        return None

    return day, day


def _period_issues(steps, output_steps, periods, columns) -> list[ValidationIssue]:
    """Como un umbral, cada salida debe acotar su fecha exactamente al período pedido; un extremo abierto («desde
    2020») no admite otra cota. Con varios períodos cada salida acota uno de ellos y ninguno queda sin salida; si son
    contiguos («2019 y 2020»), también vale una salida que los abarque juntos."""
    if not periods:
        return []

    applied = {
        key: _applied_period(
            [steps[item] for item in _reachable_step_ids(steps, [key])], steps, columns
        )
        for key in output_steps
    }
    accepted = [*periods, *_contiguous_span(periods)]
    covered, issues = set(), []
    for key, interval in applied.items():
        matched = {index for index, period in enumerate(accepted) if _bounds_period(interval, period)}
        if not matched:
            issues.append(_missing_period(key, periods))
        covered |= matched

    if len(periods) > 1 and len(periods) not in covered:
        issues += [
            _missing_period(None, [period])
            for index, period in enumerate(periods)
            if index not in covered
        ]
    return issues


def _bounds_period(applied, period) -> bool:
    """Valores TEXT acotan el período si todos los períodos que nombran caen dentro de él (el Planner los elige
    así); una fecha o un año, si sus cotas son exactamente las pedidas."""
    if applied is None:
        return False
    if applied.named:
        return (
            applied.start is None
            and applied.end is None
            and all(
                (period.start is None or start >= period.start)
                and (period.end is None or end <= period.end)
                for start, end in applied.named
            )
        )
    return (applied.start, applied.end) == (period.start, period.end)


def _contiguous_span(periods) -> list[RequestedPeriod]:
    """Períodos cerrados y seguidos («2019 y 2020») también se responden juntos; separados («2019 y 2021»), no."""
    if len(periods) < 2 or any(period.start is None or period.end is None for period in periods):
        return []

    ordered = sorted(periods, key=lambda period: period.start)
    if any(after.start != before.end + timedelta(days=1) for before, after in zip(ordered, ordered[1:])):
        return []

    text = " y ".join(period.text for period in ordered)
    return [RequestedPeriod(text=text, start=ordered[0].start, end=ordered[-1].end)]


def _missing_period(key, periods) -> ValidationIssue:
    requested = " o ".join(f"{period.text} ({_interval_text(period)})" for period in periods)
    target = f"la salida {key} no" if key is not None else "ninguna salida"
    return ValidationIssue(
        code="missing_requested_period",
        severity="clarification",
        step_id=key,
        message=f"La consulta pide {requested}, pero {target} acota la fecha a ese intervalo: filtrá la columna de "
        "fecha del read con ge y le esos días, sólo con la cota que existe si el período es abierto (o el año).",
    )


def _interval_text(period) -> str:
    if period.start is None:
        return f"hasta el {period.end.isoformat()}"
    if period.end is None:
        return f"desde el {period.start.isoformat()}"
    return f"del {period.start.isoformat()} al {period.end.isoformat()}"


def _time_grain_issues(steps, output_steps, time_grain) -> list[ValidationIssue]:
    """«Promedio por mes»: cada salida que calcula debe agrupar una fecha por ese período; que otra rama lo haga no
    alcanza. Sin cálculos (registros leídos), la agrupación no aplica a esa salida."""
    issues = []
    for key in output_steps:
        chain = [steps[item] for item in _reachable_step_ids(steps, [key])]
        calculates = any(step.operation == "aggregate" for step in chain)
        grouped = {grouping.time_grain for step in chain for grouping in step.group_by}
        if calculates and time_grain not in grouped:
            issues.append(
                ValidationIssue(
                    code="missing_requested_time_grain",
                    severity="clarification",
                    step_id=key,
                    message=f"La consulta pide agrupar por {_GRAIN_NAMES[time_grain]}: el aggregate de la "
                    f"salida {key} debe agrupar la fecha con time_grain={time_grain}, no por fecha exacta.",
                )
            )

    published = [steps[key] for key in _reachable_step_ids(steps, output_steps)]
    calculated = any(step.operation == "aggregate" for step in published)
    grouped = {grouping.time_grain for step in published for grouping in step.group_by}
    if not calculated and time_grain not in grouped:
        issues.append(
            ValidationIssue(
                code="missing_requested_time_grain",
                severity="clarification",
                message=f"La consulta pide agrupar por {_GRAIN_NAMES[time_grain]}: el aggregate debe agrupar "
                f"la fecha con time_grain={time_grain}, no por fecha exacta.",
            )
        )

    return issues


def requested_dimensions(evidence, datasets) -> dict[UUID, set[str]]:
    """Columnas no numéricas de lo que se pregunta («qué sitios»), por dataset."""
    types = {
        (dataset.dataset_id, column.sql_name): column.sql_type
        for dataset in datasets
        for column in dataset.columns_metadata
    }
    dimensions = {}

    for item in evidence.get("observed_evidence", []):
        dataset_id = UUID(item["dataset_id"])
        columns = {
            column
            for names in item.get("requested_dimension_columns", {}).values()
            for column in names
            if types.get((dataset_id, column)) not in {*_NUMERIC_TYPES, None}
        }
        if columns:
            dimensions[dataset_id] = columns

    return dimensions


def _threshold_issues(steps, output_steps, thresholds) -> list[ValidationIssue]:
    """Como los valores pedidos, cada salida publicada debe conservar cada umbral."""
    issues = []

    for key in output_steps:
        chain = [steps[item] for item in _reachable_step_ids(steps, [key])]
        for threshold in thresholds:
            issues.extend(_output_threshold_issues(key, chain, steps, threshold))

    return issues


def _output_threshold_issues(key, chain, steps, threshold) -> list[ValidationIssue]:
    applied = [
        predicate
        for step in chain
        for predicate in step.filters
        if applies_threshold(predicate, threshold)
    ]
    if not applied:
        return [
            ValidationIssue(
                code="missing_requested_threshold",
                severity="clarification",
                step_id=key,
                message=f"La consulta pide {threshold.text}, pero la salida {key} no filtra con "
                f"{threshold.operator} {threshold.value}.",
            )
        ]

    on_calculation = any(_filters_calculation(predicate, steps) for predicate in applied)
    if threshold.compares_statistic and not on_calculation:
        return [
            ValidationIssue(
                code="threshold_before_calculation",
                severity="clarification",
                step_id=key,
                message=f"{threshold.text} compara el resultado del cálculo: filtrá su alias en un sort "
                "que dependa del aggregate, no los registros antes de calcular.",
            )
        ]

    return []


def applies_threshold(predicate, threshold: RequestedThreshold) -> bool:
    """El filtro aplica exactamente la comparación pedida: mismo operador y mismo número."""
    return predicate.operator == threshold.operator and any(
        type(value) in {int, float} and value == threshold.value for value in predicate.values
    )


def _filters_calculation(predicate, steps) -> bool:
    ref = predicate.field
    if ref.kind != "result":
        return False

    origin = steps.get(ref.step_id)
    while origin is not None and origin.operation in _PASSTHROUGH_OPERATIONS and origin.depends_on:
        origin = steps.get(origin.depends_on[0])

    return origin is not None and any(
        calculation.output_name == ref.field for calculation in origin.calculations
    )


def _statistic_issues(published, statistics) -> list[ValidationIssue]:
    computed = {calculation.function for step in published for calculation in step.calculations}

    return [
        ValidationIssue(
            code="missing_requested_statistic",
            severity="clarification",
            message=f"La consulta pide la estadística {function}, pero ninguna salida la calcula: "
            "agregá un aggregate con ese cálculo; un read sin aggregate no la responde.",
        )
        for function in statistics
        if not _STATISTIC_FUNCTIONS.get(function, {function}) & computed
    ]


def _dimension_issues(steps, output_steps, dimensions) -> list[ValidationIssue]:
    """La categoría preguntada debe verse en cada salida: su columna, si se identificó, o al menos varias filas."""
    issues = []

    for key in output_steps:
        rows = _published_rows(steps.get(key), steps)
        if rows is None:
            continue

        requested = {
            (dataset_id, column)
            for dataset_id in _read_datasets(rows, steps)
            for column in dimensions.get(dataset_id, ())
        }
        if requested and not _exposes(rows, requested):
            names = ", ".join(sorted({column for _, column in requested}))
            issues.append(
                ValidationIssue(
                    code="missing_requested_dimension",
                    severity="clarification",
                    step_id=key,
                    message=f"La consulta pregunta por {names}, pero la salida {key} no la incluye: agrupá "
                    "por esa columna en el aggregate o publicá registros que la contengan (un sort del read).",
                )
            )
        elif not requested and _is_ungrouped(rows, steps):
            issues.append(
                ValidationIssue(
                    code="missing_requested_grouping",
                    severity="clarification",
                    step_id=key,
                    message=f"La consulta pide un resultado por categoría, pero la salida {key} es un único "
                    "valor sin agrupar: agregá al aggregate un group_by con la columna de esa categoría.",
                )
            )

    return issues


def _is_ungrouped(step, steps) -> bool:
    while step.operation in _PASSTHROUGH_OPERATIONS and step.depends_on and step.depends_on[0] in steps:
        step = steps[step.depends_on[0]]

    return step.operation == "aggregate" and not step.group_by


def _published_rows(step, steps):
    """Un gráfico publica las filas de su entrada."""
    if step is not None and step.operation == "visualize" and step.depends_on:
        return steps.get(step.depends_on[0])

    return step


def _read_datasets(step, steps) -> set[UUID]:
    datasets, visit = set(), [step] if step is not None else []

    while visit:
        current = visit.pop()
        if current.operation == "read":
            datasets.update(ref.dataset_id for ref in current.columns)
        visit.extend(steps[key] for key in current.depends_on if key in steps)

    return datasets


def _exposes(step, requested) -> bool:
    return any(
        not field.numeric_eligible
        and {(ref.dataset_id, ref.column) for ref in field.sources} & requested
        for field in step.output_fields
    )


def validate_selections(
    decision: PlannerDecision, catalog: list[VerifiedDataset]
) -> list[SelectedDataset]:
    available = {dataset.dataset_id: dataset for dataset in catalog}
    result = []
    seen = set()

    for selection in decision.datasets:
        _require(selection.dataset_id not in seen, "El modelo repitió un dataset.")
        seen.add(selection.dataset_id)

        dataset = available.get(selection.dataset_id)
        _require(
            dataset is not None, "El modelo seleccionó un dataset fuera del contexto verificado."
        )

        result.append(_selected_dataset(selection, dataset))

    return result


def _invented_columns(dataset: VerifiedDataset, invented: list[str]) -> str:
    """Rechazo que nombra las columnas inventadas y las reales más parecidas, para el reintento y la trazabilidad."""
    names = [column.sql_name for column in dataset.columns_metadata]
    similar = list(
        dict.fromkeys(match for name in invented for match in get_close_matches(name, names, n=3, cutoff=0.6))
    )
    message = (
        f"El modelo seleccionó una columna inexistente en {dataset.name}: {', '.join(invented)}."
        + (f" Columnas parecidas que sí existen: {', '.join(similar)}." if similar else "")
    )
    dated = any(_names_date(column.original_name) for column in dataset.columns_metadata)
    if not dated and not period_columns([dataset]) and any(_names_date(name) for name in invented):
        message += " Ese dataset no tiene columna de fecha ni año: no se puede acotar por período."
    return message


def _names_date(name: str) -> bool:
    return bool(set(singular_words(name)) & _DATE_WORDS)


def _selected_dataset(selection: DatasetSelection, dataset: VerifiedDataset) -> SelectedDataset:
    _require(
        selection.version == dataset.verified_version,
        "La versión propuesta no coincide con PostgreSQL.",
    )
    columns = {column.sql_name: column for column in dataset.columns_metadata}
    _require(len(set(selection.columns)) == len(selection.columns), "La selección repite columnas.")
    invented = [name for name in selection.columns if name not in columns]
    _require(not invented, _invented_columns(dataset, invented))

    names = _with_projection_origins(selection.columns, columns)

    return SelectedDataset(
        **dataset.model_dump(exclude={"columns_metadata"}),
        columns_metadata=[columns[name].model_copy(deep=True) for name in names],
        selection_reason=selection.reason,
    )


def _with_projection_origins(selected: list[str], columns: dict) -> list[str]:
    names = list(selected)

    for name in selected:
        origin = columns[name].derived_from
        if not origin:
            continue
        _require(
            origin in columns, "La proyección numérica no tiene columna de origen verificable."
        )
        if origin not in names:
            names.append(origin)

    return names


def validate_operations(
    decision: PlannerDecision, selected: list[SelectedDataset], temporal_text=()
) -> tuple[list[ValidatedStep], list[ValidationIssue]]:
    return _OperationValidator(decision, selected, temporal_text).validate()


class _OperationValidator:
    def __init__(self, decision: PlannerDecision, selected: list[SelectedDataset], temporal_text):
        self._decision = decision
        self._datasets = {dataset.dataset_id: dataset for dataset in selected}
        self._periods = {(item.dataset_id, item.column) for item in temporal_text}
        self._requested = {item.dataset_id: set(item.columns) for item in decision.datasets}
        self._steps: dict[str, ValidatedStep] = {}
        self._outputs: dict[str, dict[str, OutputField]] = {}
        self._used_datasets: set[UUID] = set()
        self._limited_steps: set[str] = set()
        self._issues: list[ValidationIssue] = []

    def validate(self) -> tuple[list[ValidatedStep], list[ValidationIssue]]:
        for step in self._decision.steps:
            self._check_dependencies(step)

            if step.operation in _COMBINE_OPERATIONS:
                self._check_combination(step)
                return [], self._issues

            fields = self._step_fields(step)
            self._check_filters(step)
            self._check_ordering(step)
            self._check_limit(step)
            self._register(step, fields)

        if self._decision.status == "plannable":
            self._check_plan_outputs()

        return list(self._steps.values()), self._issues

    def _issue(self, code, message, step_id, severity="clarification") -> None:
        self._issues.append(
            ValidationIssue(code=code, message=message, step_id=step_id, severity=severity)
        )

    def _check_dependencies(self, step) -> None:
        _require(step.id not in self._steps, "Los identificadores de pasos deben ser únicos.")
        _require(len(set(step.depends_on)) == len(step.depends_on), "Un paso repite dependencias.")
        _require(
            all(dependency in self._steps for dependency in step.depends_on),
            f"El plan contiene un ciclo o dependencia futura/inexistente: el paso {step.id} depende de {step.depends_on}.",
        )

    def _check_combination(self, step) -> None:
        _require(len(step.depends_on) >= 2, "Combinar requiere al menos dos resultados previos.")

        for key in step.join_on:
            self._resolve(step, key.left)
            self._resolve(step, key.right)

        self._issue(
            "unverified_observation_compatibility",
            "La combinación requiere confirmar claves, granularidad, cardinalidad y solapamientos; el catálogo actual no los garantiza.",
            step.id,
        )

    def _raw_field(self, ref: ColumnRef) -> OutputField:
        _require(
            ref.dataset_id in self._requested and ref.column in self._requested[ref.dataset_id],
            "Una operación usa una columna no seleccionada.",
        )
        column = next(
            column
            for column in self._datasets[ref.dataset_id].columns_metadata
            if column.sql_name == ref.column
        )

        return OutputField(
            name=column.sql_name,
            sql_type=column.sql_type,
            numeric_eligible=column.numeric_eligible is True,
            unit=column.unit,
            unit_status="documented" if column.unit is not None else "unknown",
            sources=[ref],
        )

    def _resolve(self, step, ref) -> OutputField:
        if isinstance(ref, ColumnRef):
            _require(
                step.operation == "read" and ref in step.columns,
                f"Sólo read puede usar referencias directas, declaradas en columns: el paso {step.id} usa la columna {ref.column}.",
            )
            return self._raw_field(ref)

        _require(
            ref.step_id in step.depends_on,
            f"La referencia debe provenir de una dependencia directa: el paso {step.id} usa "
            f"{ref.step_id}.{ref.field}, pero depende de {step.depends_on}.",
        )
        _require(
            ref.field in self._outputs[ref.step_id],
            f"La operación refiere a un resultado inexistente: {ref.step_id} no expone «{ref.field}»; "
            f"expone {sorted(self._outputs[ref.step_id])}.",
        )

        return self._outputs[ref.step_id][ref.field]

    def _step_fields(self, step) -> list[OutputField]:
        _require(not step.join_on, "Las claves de unión sólo corresponden a join.")
        _require(
            (step.operation == "visualize") == (step.visualization is not None),
            "visualize requiere su especificación exclusiva de gráfico.",
        )

        if step.operation == "read":
            return self._read_fields(step)

        return self._dependent_fields(step)

    def _read_fields(self, step) -> list[OutputField]:
        _require(
            step.tool == "sql" and not step.depends_on and bool(step.columns),
            "read requiere SQL, columnas y ninguna dependencia.",
        )
        _require(
            not step.group_by and not step.calculations,
            "read no puede ocultar agrupaciones ni cálculos.",
        )

        ids = {ref.dataset_id for ref in step.columns}
        _require(
            len(ids) == 1,
            f"Cada read debe leer un único dataset: el paso {step.id} mezcla columnas de {len(ids)} datasets. "
            "Usá un read por dataset y una rama independiente para cada uno (read -> sort, aggregate o latest "
            "propios) y poné el último paso de cada rama en output_steps.",
        )
        self._used_datasets.update(ids)

        return [self._raw_field(ref) for ref in step.columns]

    def _dependent_fields(self, step) -> list[OutputField]:
        source = self._single_source(step)
        inherited = list(self._outputs[source].values())

        if step.operation not in _PASSTHROUGH_OPERATIONS:
            _require(
                not step.order_by and step.limit is None,
                "Orden y límite requieren un paso sort explícito después del cálculo.",
            )

        if step.operation == "visualize":
            self._check_visualization(step, source)
            return []

        if step.operation == "sort":
            self._check_sort(step, source)
            return inherited

        if step.operation == "latest":
            self._check_latest(step, source)
            return inherited

        return self._calculated_fields(step, source)

    def _single_source(self, step) -> str:
        _require(
            len(step.depends_on) == 1 and not step.columns,
            "La operación requiere un resultado previo; no columnas directas.",
        )
        source = step.depends_on[0]
        _require(
            self._steps[source].operation != "visualize",
            "Un gráfico no es una tabla de entrada para cálculos.",
        )

        if step.tool == "sql" and self._steps[source].tool == "python":
            self._issue(
                "materialization_required",
                "El paso SQL queda pendiente: consumir resultados Python requiere una estrategia de materialización todavía no implementada.",
                step.id,
                "warning",
            )

        return source

    def _check_visualization(self, step, source: str) -> None:
        _require(
            step.tool == "python"
            and not step.group_by
            and not step.calculations
            and not step.filters,
            "visualize requiere Python y una entrada explícita sin cálculos ni filtros ocultos.",
        )

        chart = step.visualization
        x = self._resolve(step, chart.x)
        y = self._resolve(step, chart.y) if chart.y is not None else None

        if chart.series is not None:
            self._resolve(step, chart.series)
            _require(
                chart.series != chart.x,
                "La serie del gráfico debe ser un campo distinto del eje x.",
            )

        if chart.chart_type in {"scatter", "histogram"}:
            _require(
                _is_numeric(x), "El eje x del gráfico requiere una variable numérica elegible."
            )

        if chart.chart_type == "line":
            _require(
                _is_numeric(x) or x.sql_type in _TEMPORAL_TYPES,
                "Una línea requiere un eje numérico o temporal verificado; no se interpretan fechas TEXT.",
            )

        if y is not None:
            _require(
                _is_numeric(y), "El eje y del gráfico requiere una variable numérica elegible."
            )
            _require(
                chart.x != chart.y,
                "Los ejes x e y del gráfico deben ser campos distintos: x es la categoría "
                "(por ejemplo, el sitio agrupado) e y el valor calculado.",
            )

        if source in self._limited_steps:
            self._issue(
                "explicitly_limited_chart",
                "El gráfico representa el límite explícito del plan, no todo el dataset.",
                step.id,
                "warning",
            )

    def _check_sort(self, step, source: str) -> None:
        _require(
            not step.group_by and not step.calculations and bool(step.order_by),
            "sort sólo debe ordenar/filtrar/limitar un resultado previo.",
        )

        # Filtrar una única fila (¿supera el umbral?) tiene sentido; ordenarla, no.
        if not step.filters and self._is_single_row(source):
            self._issue(
                "degenerate_sort",
                f"El paso {step.id} ordena el resultado de {source}, que es una única fila sin agrupación: "
                "si la pregunta pide un valor por categoría, agregá esa columna como group_by del aggregate; "
                f"si no, publicá {source} sin ordenarlo.",
                step.id,
            )

    def _is_single_row(self, key: str) -> bool:
        step = self._steps[key]
        while step.operation in _PASSTHROUGH_OPERATIONS and step.depends_on:
            step = self._steps[step.depends_on[0]]

        return step.operation == "aggregate" and not step.group_by

    def _check_latest(self, step, source: str) -> None:
        _require(
            not step.group_by
            and not step.calculations
            and step.limit is None
            and len(step.order_by) == 1
            and step.order_by[0].direction == "desc",
            "latest requiere una fecha descendente, sin agregaciones ni límites que corten empates.",
        )

        temporal = self._resolve(step, step.order_by[0].field)
        _require(
            temporal.sql_type in _TEMPORAL_TYPES
            or (self._is_verified_period(temporal) and step.tool == "python"),
            "latest requiere una fecha real DATE/TIMESTAMP o períodos TEXT verificados; no ordena textos libres ni números de fila.",
        )

        if source in self._limited_steps:
            self._issue(
                "limited_latest_input",
                "La fecha más reciente requiere la entrada completa; no se calcula sobre una lectura limitada.",
                step.id,
            )

    def _is_verified_period(self, field: OutputField) -> bool:
        # Un TEXT sólo ordena si su columna fue perfilada como períodos explícitos;
        # Python interpreta cada valor con las mismas reglas y descarta los ambiguos.
        return (
            field.sql_type == "TEXT"
            and len(field.sources) == 1
            and (field.sources[0].dataset_id, field.sources[0].column) in self._periods
        )

    def _calculated_fields(self, step, source: str) -> list[OutputField]:
        _require(
            bool(step.calculations),
            f"El paso {step.id} ({step.operation}) debe declarar cálculos.",
        )
        self._check_calculation_input(step, source)

        if step.operation == "calculate":
            self._check_scalar_calculation(step)

        fields = [self._grouping_field(step, grouping) for grouping in step.group_by]
        for calculation in step.calculations:
            fields.append(self._calculation_field(step, calculation))

        return fields

    def _check_calculation_input(self, step, source: str) -> None:
        reaggregates_single_row = (
            step.operation == "aggregate"
            and self._steps[source].operation == "aggregate"
            and not self._steps[source].group_by
        )
        if reaggregates_single_row:
            # Agregar una única fila ya agregada (contarla, promediarla) no responde nada.
            self._issue(
                "degenerate_aggregate",
                f"El paso {step.id} vuelve a agregar el resultado de {source}, que ya es "
                f"una única fila sin agrupación; publicá {source} en output_steps en lugar de agregarlo otra vez.",
                step.id,
            )

        if source in self._limited_steps:
            self._issue(
                "limited_calculation_input",
                "El cálculo depende de una lectura limitada; hay que precisar si se pretende calcular sobre una muestra.",
                step.id,
            )

    def _check_scalar_calculation(self, step) -> None:
        _require(not step.group_by, "calculate no admite agrupaciones implícitas.")

        functions = {calculation.function for calculation in step.calculations}
        _require(
            not ("correlation" in functions and len(functions) > 1),
            "No mezclar una correlación escalar con cálculos por registro.",
        )

    def _grouping_field(self, step, grouping) -> OutputField:
        _require(step.operation == "aggregate", "Las agrupaciones pertenecen a aggregate.")
        field = self._resolve(step, grouping.field).model_copy(deep=True)

        if grouping.time_grain != "raw":
            if field.sql_type not in _TEMPORAL_TYPES:
                self._issue(
                    "ambiguous_date",
                    "No se puede agrupar por calendario una columna TEXT o de fecha parcial sin una interpretación confirmada.",
                    step.id,
                )
            field.sql_type = "BIGINT" if grouping.time_grain == "year" else "DATE"
            field.numeric_eligible = False
            field.unit = None
            field.unit_status = "dimensionless"

        field.name = grouping.output_name

        return field

    def _calculation_field(self, step, calculation) -> OutputField:
        function = calculation.function
        _require(
            (function in _AGGREGATE_FUNCTIONS) == (step.operation == "aggregate"),
            "La función no corresponde al tipo de operación.",
        )
        if function in _PYTHON_ONLY_FUNCTIONS:
            _require(
                step.tool == "python",
                "La nueva función estadística está habilitada exclusivamente en Python.",
            )

        operands = [self._resolve(step, ref) for ref in calculation.inputs]
        refs = [ref for operand in operands for ref in operand.sources]
        sql_type, unit, unit_status = self._calculation_result(step, function, operands)

        return OutputField(
            name=calculation.output_name,
            sql_type=sql_type,
            numeric_eligible=sql_type in _NUMERIC_TYPES,
            unit=unit,
            unit_status=unit_status,
            sources=list({(ref.dataset_id, ref.column): ref for ref in refs}.values()),
        )

    def _calculation_result(self, step, function: str, operands: list[OutputField]):
        if function in _COUNT_FUNCTIONS:
            return "BIGINT", None, "dimensionless"

        self._check_operands_eligible(step, function, operands)
        sql_type = operands[0].sql_type if function in _EXTREME_FUNCTIONS else "NUMERIC"
        unit, unit_status = operands[0].unit, operands[0].unit_status

        if function in {"difference", "ratio"}:
            self._check_comparable_units(step, operands)

        if function in {"ratio", "correlation"}:
            return sql_type, None, "dimensionless"

        if presentation_unit_status(operands[0]) == "unknown" and sql_type in _NUMERIC_TYPES:
            self._issue(
                "undocumented_unit",
                "El resultado conserva una unidad desconocida; no se infiere a partir del nombre.",
                step.id,
                "warning",
            )

        return sql_type, unit, unit_status

    def _check_operands_eligible(self, step, function: str, operands: list[OutputField]) -> None:
        for operand in operands:
            temporal_extreme = (
                function in _EXTREME_FUNCTIONS and operand.sql_type in _TEMPORAL_TYPES
            )
            if temporal_extreme or _is_numeric(operand):
                continue
            self._issue(
                "ineligible_calculation",
                "El cálculo requiere una columna numérica elegible o una proyección numérica documentada; no se convierte TEXT automáticamente.",
                step.id,
            )

    def _comparable(self, field: OutputField) -> tuple[str, str | None]:
        source = field.sources[0] if len(field.sources) == 1 else None
        dataset = self._datasets.get(source.dataset_id) if source else None
        column = next(
            (item for item in dataset.columns_metadata if item.sql_name == source.column), None
        ) if dataset else None
        return comparable_unit(column.original_name if column else field.name, field.unit)

    def _check_comparable_units(self, step, operands: list[OutputField]) -> None:
        # La misma unidad con otra grafía («mg/L» y «mg L-1») es comparable si también lo es su base química (mg N/L
        # y mg/L sin base no lo son); no se convierte entre unidades.
        compatible = (
            operands[0].unit is not None
            and operands[1].unit is not None
            and self._comparable(operands[0]) == self._comparable(operands[1])
        )
        same_variable = operands[0].sources == operands[1].sources

        if not compatible and not same_variable:
            self._issue(
                "unverified_units",
                "La operación necesita unidades comparables documentadas; no se infieren equivalencias ni conversiones.",
                step.id,
            )

    def _check_filters(self, step) -> None:
        self._check_contradictory_filters(step)

        for predicate in step.filters:
            field = self._resolve(step, predicate.field)
            _require(
                predicate.operator != "ieq" or field.sql_type == "TEXT",
                "ieq sólo admite igualdad literal de columnas TEXT sin distinguir mayúsculas.",
            )

            if predicate.operator in _RANGE_OPERATORS and field.sql_type == "TEXT":
                self._issue(
                    "text_range",
                    "Un rango sobre TEXT necesita confirmar si el orden es textual, numérico o temporal.",
                    step.id,
                )

            for value in predicate.values:
                _check_filter_value(field, value)

    def _check_contradictory_filters(self, step) -> None:
        equalities = {}

        for predicate in step.filters:
            if predicate.operator not in _EQUALITY_OPERATORS:
                continue

            key = predicate.field.model_dump_json()
            values = {str(value).casefold() for value in predicate.values}
            if key in equalities and not equalities[key] & values:
                self._issue(
                    "contradictory_filters",
                    f"El paso {step.id} exige valores incompatibles en el mismo campo "
                    f"({sorted(equalities[key])} y {sorted(values)}); ninguna fila puede cumplir ambos. "
                    "Para varios valores usá in o ramas separadas.",
                    step.id,
                )

            equalities[key] = equalities.get(key, values) & values or values

    def _check_ordering(self, step) -> None:
        for ordering in step.order_by:
            field = self._resolve(step, ordering.field)
            _require(
                not (step.limit == 1 and field.sql_type in _TEMPORAL_TYPES),
                "Un límite de una fila sobre fechas puede elegir arbitrariamente un empate; usá latest para conservar todas las mediciones.",
            )

    def _check_limit(self, step) -> None:
        if step.limit is None:
            return

        ancestors = list(step.depends_on)
        while ancestors:
            ancestor = self._steps[ancestors.pop()]
            _require(
                ancestor.operation != "latest",
                "No se permite recortar con un límite las mediciones empatadas de latest.",
            )
            ancestors.extend(ancestor.depends_on)

    def _register(self, step, fields: list[OutputField]) -> None:
        _require(
            len({field.name for field in fields}) == len(fields),
            "Las columnas de salida de un paso deben tener nombres únicos.",
        )

        self._steps[step.id] = ValidatedStep(**step.model_dump(), output_fields=fields)
        self._outputs[step.id] = {field.name: field for field in fields}

        inherits_limit = any(dependency in self._limited_steps for dependency in step.depends_on)
        if step.limit is not None or inherits_limit:
            self._limited_steps.add(step.id)

    def _check_plan_outputs(self) -> None:
        output_steps = self._decision.output_steps
        _require(
            self._used_datasets == set(self._datasets),
            "Todo dataset seleccionado debe participar en el plan.",
        )
        _require(
            len(set(output_steps)) == len(output_steps),
            "Las salidas finales no deben repetirse.",
        )
        _require(
            all(step_id in self._steps for step_id in output_steps),
            "Una salida final refiere a un paso inexistente.",
        )


def _check_filter_value(field: OutputField, value) -> None:
    if field.sql_type in _NUMERIC_TYPES:
        _require(type(value) in {int, float}, "Un filtro numérico requiere constantes numéricas.")
        return

    if field.sql_type == "BOOLEAN":
        _require(type(value) is bool, "Un filtro booleano requiere valores booleanos.")
        return

    _require(isinstance(value, str), "Un filtro de texto/fecha requiere constantes textuales.")

    parse = _ISO_PARSERS.get(field.sql_type)
    if parse is None:
        return

    try:
        parse(value)
    except ValueError as error:
        raise PlannerValidationError("El filtro temporal no usa una fecha ISO completa.") from error


def validate_execution_plan(plan: PlanningResult, catalog: list[VerifiedDataset]):
    """Reutiliza las verificaciones del Planner con un catálogo fresco y confiable."""
    _require_execution_authorized(plan)

    decision = _revalidation_decision(plan)
    snapshots = [VerifiedDataset.model_validate(item.model_dump()) for item in catalog]
    selected = validate_selections(decision, snapshots)
    _require_unchanged_metadata(plan.selected_datasets, selected)

    operations, issues = validate_operations(decision, selected, plan.temporal_text_columns)
    evidence = plan.context.evidence
    issues.extend(
        validate_temporal_scope(
            operations,
            plan.output_steps,
            require_latest=_latest_required(evidence),
            requested_values=_requested_values(evidence),
        )
    )
    issues.extend(
        validate_requested_scope(
            operations,
            plan.output_steps,
            thresholds=_requested_thresholds(evidence),
            statistics=evidence.get("requested_statistics", []),
            dimensions=requested_dimensions(evidence, selected),
            categories=_asks_categories(evidence),
            chart=evidence.get("requested_chart", False),
            time_grain=evidence.get("requested_time_grain"),
            periods=_requested_periods(evidence),
            dated_columns=period_columns(selected, plan.temporal_text_columns),
        )
    )
    _require(
        not any(issue.severity == "clarification" for issue in issues),
        "La operación requiere resolver una aclaración antes de ejecutarse.",
    )

    schema_changed = any(
        old.output_fields != new.output_fields
        for old, new in zip(plan.operations, operations, strict=True)
    )
    _require(not schema_changed, "El esquema del plan no coincide con el derivado del catálogo.")

    return selected, operations


def _require_execution_authorized(plan: PlanningResult) -> None:
    authorized = plan.status == "plannable" and not any(
        issue.severity == "clarification" for issue in plan.validation_issues
    )
    _require(authorized, "El plan no está autorizado para ejecución.")


def _revalidation_decision(plan: PlanningResult) -> PlannableDecision:
    return PlannableDecision(
        status="plannable",
        reason="Revalidación previa a ejecución.",
        datasets=[_dataset_selection(item) for item in plan.selected_datasets],
        steps=[step.model_dump(exclude={"output_fields"}) for step in plan.operations],
        output_steps=plan.output_steps,
    )


def _dataset_selection(dataset: SelectedDataset) -> DatasetSelection:
    return DatasetSelection(
        dataset_id=dataset.dataset_id,
        version=dataset.verified_version,
        columns=[column.sql_name for column in dataset.columns_metadata],
        reason=dataset.selection_reason,
    )


def _require_unchanged_metadata(proposed_datasets, actual_datasets) -> None:
    for proposed, actual in zip(proposed_datasets, actual_datasets, strict=True):
        changed = any(
            getattr(proposed, field) != getattr(actual, field) for field in _AUTHORIZED_METADATA
        )
        _require(not changed, "Los metadatos autorizados cambiaron; hay que volver a planificar.")


def _latest_required(evidence) -> bool:
    return evidence.get("temporal_requirements", {}).get("operation") == "latest"


def _requested_values(evidence) -> list[str]:
    return [item["text"] for item in evidence.get("query_mentions", []) if item["kind"] == "value"]


def _asks_categories(evidence) -> bool:
    observed = evidence.get("observed_evidence", [])
    return any(item["kind"] == "dimension" for item in evidence.get("query_mentions", [])) or any(
        item.get("requested_dimension_columns") for item in observed
    )


def _requested_periods(evidence) -> list[RequestedPeriod]:
    return [
        RequestedPeriod.model_validate(item, strict=False)
        for item in evidence.get("requested_periods", [])
    ]


def _requested_thresholds(evidence) -> list[RequestedThreshold]:
    return [
        RequestedThreshold.model_validate(item) for item in evidence.get("requested_thresholds", [])
    ]
