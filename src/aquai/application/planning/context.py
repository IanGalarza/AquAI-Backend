import json
import re
from hashlib import sha256

from aquai.application.contracts.planning import (
    PlanningContextRecord,
    RequestedPeriod,
    RequestedThreshold,
)
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.matching import (
    aligned_span,
    contains_any_phrase,
    contains_phrase,
    names_period,
    normalize_text,
    numeric_comparisons,
    period_ranges,
    period_text,
)
from aquai.application.Rules.vocabulary import (
    ASCENDING_ORDER_CUES,
    CHART_WORDS,
    COUNT_BOUNDS,
    COUNT_CUE,
    DESCENDING_ORDER_CUES,
    FULL_ORDER_CUES,
    HIGHEST_CUES,
    LOWEST_CUES,
    NUMBER_WORDS,
    OBSERVATION_NOUNS_EN,
    OBSERVATION_NOUNS_ES,
    ORDER_KEY,
    PERIOD_SPAN_CUES,
    RATE_COUNT,
    RECENT_INTERVAL_NOUNS,
    RECENT_WORDS,
    REQUESTED_STATISTICS,
    SUPERLATIVE_WORDS,
)
from aquai.application.Rules.language import INTERROGATIVE_WORDS
from aquai.application.units import unitless_name

_COLUMN_FIELDS = (
    "sql_name",
    "original_name",
    "sql_type",
    "unit",
    "numeric_eligible",
    "annotations",
)
_ANNOTATION_FIELDS = (
    "derived_from",
    "representation",
    "excluded_count",
    "header_missing",
    "meaning",
)
_CATALOG_SCOPE = "Sólo candidatos recuperados y revalidados en PostgreSQL; no es el catálogo completo. Sin registros ni evidencia de claves únicas o cardinalidad de joins."
_MAX_STORED_VALUES = 30
_MAX_DESCRIPTION_LENGTH = 160
_MAX_MEANING_LENGTH = 100
_COUNT_STATISTICS = {"count", "count_distinct"}


_NUMBER_WORDS = "|".join(word for words in NUMBER_WORDS.values() for word in words)
_LATEST = re.compile(
    rf"\b(?:(?:[uú]ltim[oa]s?|m[aá]s\s+recientes?)\s+{OBSERVATION_NOUNS_ES}|"
    rf"(?:latest|most\s+recent|last)\s+{OBSERVATION_NOUNS_EN})\b|"
    rf"\b(?:m[aá]s\s+recientes?|most\s+recent|actualmente)\b|"
    r"(?<=\w\s)actual(?:es)?\b(?!\s+[^\W\d_])|(?<=\w\s)actual(?:es)?(?=\s+(?:en|de|del|para|seg[uú]n)\b)|"
    rf"(?<!\bpor\s)\b{RECENT_WORDS}\s+(?:(?:de|del|of|the)\s+)?"
    rf"(?!(?:{RECENT_INTERVAL_NOUNS}|{_NUMBER_WORDS})\b)(?!\d)\w",
    re.IGNORECASE,
)
_SUPERLATIVE = re.compile(SUPERLATIVE_WORDS)
_ORDER_KEY = re.compile(ORDER_KEY, re.IGNORECASE)
_VALUE_WORDS = {"valor", "valores", "value", "values", "magnitud", "concentracion", "nivel", "niveles"}
_COUNT_BOUND = re.compile(COUNT_BOUNDS)
_STATISTIC_WORDS = {
    word for function, words in REQUESTED_STATISTICS.items() if function != "count" for word in words if " " not in word
}
_EXTREME_WORDS = {
    *(word for cue in (*HIGHEST_CUES, *LOWEST_CUES) for word in cue.split() if word not in {"mas", "valor"}),
    "mayor",
    "menor",
}
_CLAUSE_BREAKS = ("y", "e", "o", "u", "and", "or", "pero", "but")
_STATISTIC_EXTREME_WINDOW = 4


#
# Indica si la pregunta pide la medición más reciente.
#
def latest_requested(question: str) -> bool:
    return _LATEST.search(question) is not None


#
# Indica si la pregunta pide el valor extremo de una variable y no todos sus registros ordenados.
#
def extreme_requested(question: str) -> bool:
    return bool(extreme_orders(question))


#
# Devuelve los órdenes que pide un extremo: desc para el mayor y asc para el menor.
#
def extreme_orders(question: str) -> set[str]:
    if contains_any_phrase(question, FULL_ORDER_CUES):
        return set()

    text = _COUNT_BOUND.sub(" ", normalize_text(question))
    superlatives = {match.group(1) for match in _SUPERLATIVE.finditer(text)}
    asked = {
        "desc": "mayor" in superlatives or contains_any_phrase(text, HIGHEST_CUES),
        "asc": "menor" in superlatives or contains_any_phrase(text, LOWEST_CUES),
    }

    return {order for order, found in asked.items() if found}


#
# Devuelve el sentido del orden completo que pide la pregunta por la variable («de mayor a menor»), o None si
# no lo dice o si ordena por otra cosa («ordenada por fecha de menor a mayor»).
#
def requested_order(question: str, variables=()) -> str | None:
    key = _ORDER_KEY.search(question)
    if key is not None and not _names_value(key.group("key"), variables):
        return None

    found = {
        "desc": contains_any_phrase(question, DESCENDING_ORDER_CUES),
        "asc": contains_any_phrase(question, ASCENDING_ORDER_CUES),
    }
    directions = [direction for direction, asked in found.items() if asked]

    return directions[0] if len(directions) == 1 else None


#
# Indica si la clave de un orden nombra el valor medido o una de las variables pedidas.
#
def _names_value(key, variables) -> bool:
    word = normalize_text(key)
    if word in _VALUE_WORDS:
        return True

    return any(aligned_span(key, variable) for variable in variables)


#
# Indica si el extremo pedido es el de una estadística y no el de los registros: la estadística y el extremo
# están en la misma cláusula y a pocas palabras («el pH promedio más alto», «la mayor mediana»), y no en
# cláusulas distintas («el pH máximo y el promedio por sitio»).
#
def statistic_extreme(question: str) -> bool:
    clause = []
    for word in [*normalize_text(question).split(), _CLAUSE_BREAKS[0]]:
        if word not in _CLAUSE_BREAKS:
            clause.append(word)
            continue
        statistics = [index for index, item in enumerate(clause) if item in _STATISTIC_WORDS]
        extremes = _extreme_positions(clause)
        if any(abs(left - right) <= _STATISTIC_EXTREME_WINDOW for left in statistics for right in extremes):
            return True
        clause = []

    return False


#
# Devuelve las posiciones de las palabras de extremo de una cláusula, sin las comparaciones («mayor que»).
#
def _extreme_positions(clause) -> list[int]:
    return [
        index
        for index, word in enumerate(clause)
        if word in _EXTREME_WORDS
        and not (word in {"mayor", "menor"} and clause[index + 1 : index + 2] in (["que"], ["a"], ["al"], ["de"], ["o"]))
    ]


#
# Devuelve los órdenes del extremo pedido, sin contar las palabras que integran un nombre: Merismopedia minima.
#
def asked_extremes(question: str, names) -> set[str]:
    text = normalize_text(question)
    for name in names:
        if extreme_requested(name):
            text = text.replace(normalize_text(unitless_name(name)), " ")

    return extreme_orders(text)


#
# Indica si lo pedido exige comparar la variable como número: una estadística, un extremo o un orden rechazado.
#
def numeric_comparison(evidence) -> bool:
    statistics = set(evidence.get("requested_statistics", [])) - _COUNT_STATISTICS

    return (
        bool(statistics)
        or evidence.get("text_comparison_rejected", False)
        or extreme_requested(evidence.get("question", ""))
    )


#
# Indica si la pregunta pide un gráfico.
#
def chart_requested(question: str) -> bool:
    return bool(re.search(rf"\b{CHART_WORDS}\b", question, re.IGNORECASE))


#
# Indica si la pregunta pide las fechas que cubren los datos: por cómo lo dice (PERIOD_SPAN_CUES) o porque el
# intérprete marcó como fecha una pregunta que no nombra ninguna («qué fechas abarcan»): no acota, pregunta.
#
def asks_period_span(question: str, mentions) -> bool:
    if PERIOD_SPAN_CUES.search(question):
        return True

    return any(
        item.kind == "date" and not names_period(item.text) and not period_ranges(item.text)
        and bool(set(normalize_text(item.text).split()) & INTERROGATIVE_WORDS)
        for item in mentions
        if not isinstance(item, str)
    )


#
# Devuelve las estadísticas que nombra la pregunta. Un «cuántos» que sólo pide concentraciones («¿cuántas
# cianobacterias por mililitro?») no pide un conteo.
#
def requested_statistics(question: str, counted=()) -> list[str]:
    rates = {match.start() for match in RATE_COUNT.finditer(question)}
    measured = all(
        cue.start() in rates or _introduces(question[cue.end() :], counted) for cue in COUNT_CUE.finditer(question)
    )
    return [
        function
        for function, words in REQUESTED_STATISTICS.items()
        if any(contains_phrase(question, word) for word in words) and not (function == "count" and measured)
    ]


#
# Indica si el texto empieza por una de las variables: «cuántas cianobacterias informa la base» pregunta por la
# variable que se mide contando organismos, no por cuántos registros hay.
#
def _introduces(text: str, variables) -> bool:
    words = normalize_text(text).split()

    return any((target := normalize_text(item).split()) and words[: len(target)] == target for item in variables)


#
# Devuelve los umbrales numéricos pedidos en las menciones de tipo umbral.
#
def requested_thresholds(mentions) -> list[RequestedThreshold]:
    thresholds = {}
    for item in mentions:
        if item.kind != "threshold":
            continue
        for comparison in numeric_comparisons(item.text)[:1]:
            thresholds[comparison.text.casefold()] = RequestedThreshold(**comparison._asdict())

    return list(thresholds.values())


#
# Devuelve los períodos pedidos en las menciones de tipo fecha.
#
def requested_periods(mentions) -> list[RequestedPeriod]:
    periods = {}
    for item in mentions:
        ranges = period_ranges(item.text) if item.kind == "date" else []
        for start, end in ranges:
            text = item.text if len(ranges) == 1 else period_text(start, end)
            periods[(start, end)] = RequestedPeriod(text=text, start=start, end=end)

    return list(periods.values())


#
# Devuelve las fechas pedidas que nombran un año o un mes pero no se pueden convertir en un período: calcular
# sin ellas respondería por todo el registro como si se hubiera acotado.
#
def unread_periods(mentions) -> list[str]:
    return [
        item.text
        for item in mentions
        if item.kind == "date" and names_period(item.text) and not period_ranges(item.text)
    ]


#
# Devuelve el texto que describe el período pedido, cerrado o abierto, en el idioma indicado.
#
def period_phrase(period: dict, language: str = "es") -> str:
    closed = period.get("start") is not None and period.get("end") is not None
    if language == "es":
        return ("a " if closed else "al período ") + period["text"]
    return ("to " if closed else "to the period ") + period["text"]


#
# Construye el contexto de planificación con el catálogo de columnas y los requisitos de la pregunta.
#
def build_planning_context(
    question: str,
    datasets: list[VerifiedDataset],
    budget_bytes: int,
    *,
    temporal_candidates_excluded=(),
    require_latest=False,
    query_mentions=(),
    temporal_text_columns=(),
    dataset_evidence=(),
    unmatched_values=(),
    source_mentions=(),
    thresholds=(),
    periods=(),
    statistics=(),
    chart=False,
    time_grain=None,
) -> PlanningContextRecord:
    observed = [_observed_dataset(item) for item in dataset_evidence]
    warnings = list(dict.fromkeys(warning for dataset in datasets for warning in dataset.warnings))
    column_catalogs, projected = _project_datasets(datasets, warnings)

    evidence = {
        "question": question,
        "column_fields": list(_COLUMN_FIELDS),
        "column_catalogs": column_catalogs,
        "warning_catalog": warnings,
        "datasets": projected,
        "catalog_scope": _CATALOG_SCOPE,
    }
    if require_latest:
        evidence["temporal_requirements"] = _temporal_requirements(temporal_candidates_excluded)
    if temporal_text_columns:
        evidence["text_period_columns"] = [
            _text_period_column(item) for item in temporal_text_columns
        ]
    if query_mentions:
        evidence["query_mentions"] = [item.model_dump(mode="json") for item in query_mentions]
    if thresholds:
        evidence["requested_thresholds"] = [item.model_dump(mode="json") for item in thresholds]
    if periods:
        evidence["requested_periods"] = [item.model_dump(mode="json") for item in periods]
    if statistics:
        evidence["requested_statistics"] = list(statistics)
    if chart:
        evidence["requested_chart"] = True
    if time_grain:
        evidence["requested_time_grain"] = time_grain
    if observed:
        evidence["observed_evidence"] = observed
    if unmatched_values:
        evidence["values_not_found_in_candidates"] = list(unmatched_values)
    if source_mentions:
        evidence["user_named_sources"] = list(source_mentions)

    compacted = []
    content = _serialized(evidence)
    if len(content) > budget_bytes:
        compacted.extend(_truncate_descriptions(projected))
        compacted.extend(_truncate_meanings(column_catalogs))
        content = _serialized(evidence)

    return PlanningContextRecord(
        evidence=evidence,
        sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
        budget_bytes=budget_bytes,
        compacted_fields=compacted,
    )


#
# Resume los valores y columnas observados en un dataset para la planificación.
#
def _observed_dataset(item) -> dict:
    observed = {
        "dataset_id": str(item.dataset_id),
        "values": {
            mention: [_stored_match(match) for match in matches]
            for mention, matches in item.value_matches.items()
        },
        "variable_columns": item.variable_matches,
        "requested_dimension_columns": item.dimension_matches,
    }
    if item.category_values:
        observed["category_values"] = item.category_values

    return observed


#
# Resume una coincidencia de valores almacenados, limitando los valores mostrados.
#
def _stored_match(match) -> dict:
    return {
        "column": match.column,
        "stored_values": match.values[:_MAX_STORED_VALUES],
        "rows": match.rows,
    }


#
# Proyecta los datasets del contexto y agrupa los catálogos de columnas repetidos.
#
def _project_datasets(datasets, warnings) -> tuple[dict[str, list], list[dict]]:
    column_catalogs = {}
    projected = []

    for dataset in datasets:
        rows = [_column_row(column) for column in dataset.columns_metadata]
        catalog_id = _register_catalog(rows, column_catalogs)
        projected.append(_dataset_projection(dataset, warnings, catalog_id))

    return column_catalogs, projected


#
# Convierte una columna en una fila compacta con sus campos y anotaciones.
#
def _column_row(column) -> list:
    row = [getattr(column, field) for field in _COLUMN_FIELDS[:-1]]
    annotations = {}

    for field in _ANNOTATION_FIELDS:
        value = getattr(column, field)
        if value not in (None, False, ""):
            annotations[field] = value

    if annotations:
        row.append(annotations)

    return row


#
# Registra un catálogo de columnas y devuelve el identificador de uno idéntico si ya existe.
#
def _register_catalog(rows, column_catalogs) -> str:
    for catalog_id, catalog in column_catalogs.items():
        if catalog == rows:
            return catalog_id

    catalog_id = f"columns_{len(column_catalogs) + 1}"
    column_catalogs[catalog_id] = rows

    return catalog_id


#
# Proyecta un dataset con sus metadatos y el catálogo de columnas que usa.
#
def _dataset_projection(dataset, warnings, catalog_id) -> dict:
    return {
        "dataset_id": str(dataset.dataset_id),
        "version": dataset.verified_version,
        "source_id": str(dataset.source_id),
        "name": dataset.name,
        "description": dataset.description,
        "row_count": dataset.row_count,
        "source_locator": dataset.provenance.source_locator,
        "warning_ids": [warnings.index(warning) for warning in dataset.warnings],
        "column_catalog_id": catalog_id,
    }


#
# Devuelve los requisitos temporales para seleccionar la medición más reciente.
#
def _temporal_requirements(temporal_candidates_excluded) -> dict:
    return {
        "operation": "latest",
        "date_types": ["DATE", "TIMESTAMP"],
        "keep_all_ties": True,
        "exclude_null_measurements_before_latest": True,
        "candidates_without_verified_dates": list(temporal_candidates_excluded),
    }


#
# Resume una columna TEXT cuyos valores nombran períodos.
#
def _text_period_column(item) -> dict:
    return {
        "dataset_id": str(item.dataset_id),
        "column": item.column,
        "precisions": item.precisions,
    }


#
# Serializa la evidencia como JSON compacto en bytes.
#
def _serialized(evidence) -> bytes:
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


#
# Recorta las descripciones largas de los datasets y devuelve los campos compactados.
#
def _truncate_descriptions(projected) -> list[str]:
    compacted = []

    for dataset in projected:
        description = dataset["description"] or ""
        if len(description) > _MAX_DESCRIPTION_LENGTH:
            dataset["description"] = description[:_MAX_DESCRIPTION_LENGTH] + " [truncado]"
            compacted.append(f"{dataset['dataset_id']}.description")

    return compacted


#
# Recorta los significados largos de las columnas y devuelve los campos compactados.
#
def _truncate_meanings(column_catalogs) -> list[str]:
    compacted = []

    for catalog_id, columns in column_catalogs.items():
        for column in columns:
            annotations = column[-1] if isinstance(column[-1], dict) else {}
            meaning = annotations.get("meaning") or ""
            if len(meaning) > _MAX_MEANING_LENGTH:
                annotations["meaning"] = meaning[:_MAX_MEANING_LENGTH] + " [truncado]"
                compacted.append(f"{catalog_id}.{column[0]}.meaning")

    return compacted
