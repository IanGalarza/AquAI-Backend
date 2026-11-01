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
    contains_any_phrase,
    contains_phrase,
    numeric_comparisons,
    period_ranges,
    period_text,
)
from aquai.application.Rules.vocabulary import (
    CHART_WORDS,
    EXTREME_CUES,
    FULL_ORDER_CUES,
    NUMBER_WORDS,
    OBSERVATION_NOUNS_EN,
    OBSERVATION_NOUNS_ES,
    RECENT_INTERVAL_NOUNS,
    RECENT_WORDS,
    REQUESTED_STATISTICS,
)

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


_NUMBER_WORDS = "|".join(word for words in NUMBER_WORDS.values() for word in words)
_LATEST = re.compile(
    rf"\b(?:(?:[uú]ltim[oa]s?|m[aá]s\s+recientes?)\s+{OBSERVATION_NOUNS_ES}|"
    rf"(?:latest|most\s+recent|last)\s+{OBSERVATION_NOUNS_EN})\b|"
    # «la turbiedad más reciente», «el pH actual», «el último pH», «the latest pH». «Actual» sólo en castellano,
    # tras lo que califica: en inglés («the actual values») significa «real».
    rf"\b(?:m[aá]s\s+recientes?|most\s+recent|actualmente)\b|"
    r"(?<=\w\s)actual(?:es)?\b(?!\s+[^\W\d_])|(?<=\w\s)actual(?:es)?(?=\s+(?:en|de|del|para|seg[uú]n)\b)|"
    # «Por último, …» ordena la pregunta; no pide la última medición.
    rf"(?<!\bpor\s)\b{RECENT_WORDS}\s+(?:(?:de|del|of|the)\s+)?"
    rf"(?!(?:{RECENT_INTERVAL_NOUNS}|{_NUMBER_WORDS})\b)(?!\d)\w",
    re.IGNORECASE,
)


def latest_requested(question: str) -> bool:
    return _LATEST.search(question) is not None


def extreme_requested(question: str) -> bool:
    """La pregunta pide el extremo de una variable («el valor máximo»), no todos sus registros ordenados."""
    return contains_any_phrase(question, EXTREME_CUES) and not contains_any_phrase(
        question, FULL_ORDER_CUES
    )


def chart_requested(question: str) -> bool:
    return bool(re.search(rf"\b{CHART_WORDS}\b", question, re.IGNORECASE))


def requested_statistics(question: str) -> list[str]:
    """Estadísticas que la pregunta nombra («pH promedio» → mean); un máximo puede ser un orden y no se exige."""
    return [
        function
        for function, words in REQUESTED_STATISTICS.items()
        if any(contains_phrase(question, word) for word in words)
    ]


def requested_thresholds(mentions) -> list[RequestedThreshold]:
    """Umbrales de las menciones threshold; una comparación sin número legible no fija ninguno."""
    thresholds = {}
    for item in mentions:
        if item.kind != "threshold":
            continue
        for comparison in numeric_comparisons(item.text)[:1]:
            thresholds[comparison.text.casefold()] = RequestedThreshold(**comparison._asdict())

    return list(thresholds.values())


def requested_periods(mentions) -> list[RequestedPeriod]:
    """Períodos de las menciones date con una fecha legible; «la última fecha» no fija ninguno. Una mención con fechas
    sueltas («2019 y 2021») pide cada período por separado."""
    periods = {}
    for item in mentions:
        ranges = period_ranges(item.text) if item.kind == "date" else []
        for start, end in ranges:
            text = item.text if len(ranges) == 1 else period_text(start, end)
            periods[(start, end)] = RequestedPeriod(text=text, start=start, end=end)

    return list(periods.values())


def period_phrase(period: dict, language: str = "es") -> str:
    """Período pedido tras «acotarlos»: «a 2021» o, si es abierto, «al período desde 2020»."""
    closed = period.get("start") is not None and period.get("end") is not None
    if language == "es":
        return ("a " if closed else "al período ") + period["text"]
    return ("to " if closed else "to the period ") + period["text"]


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
        # Columnas TEXT cuyos valores distintos nombran períodos explícitos (12/2008, 2013-02-08).
        evidence["text_period_columns"] = [
            _text_period_column(item) for item in temporal_text_columns
        ]
    if query_mentions:
        evidence["query_mentions"] = [item.model_dump(mode="json") for item in query_mentions]
    if thresholds:
        # Operador y valor exactos de cada comparación pedida; se revalidan antes de ejecutar.
        evidence["requested_thresholds"] = [item.model_dump(mode="json") for item in thresholds]
    if periods:
        # Intervalo exacto de cada período pedido; toda salida debe acotar su fecha a él.
        evidence["requested_periods"] = [item.model_dump(mode="json") for item in periods]
    if statistics:
        evidence["requested_statistics"] = list(statistics)
    if chart:
        evidence["requested_chart"] = True
    if time_grain:
        # «Promedio por mes» agrupa la fecha por mes; un group_by de la fecha exacta no lo responde.
        evidence["requested_time_grain"] = time_grain
    if observed:
        # Hechos observados en PostgreSQL: valores TEXT almacenados que contienen cada mención.
        evidence["observed_evidence"] = observed
    if unmatched_values:
        evidence["values_not_found_in_candidates"] = list(unmatched_values)
    if source_mentions:
        # Nombres con los que el usuario eligió la fuente: ya definen los candidatos.
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


def _stored_match(match) -> dict:
    return {
        "column": match.column,
        "stored_values": match.values[:_MAX_STORED_VALUES],
        "rows": match.rows,
    }


def _project_datasets(datasets, warnings) -> tuple[dict[str, list], list[dict]]:
    column_catalogs = {}
    projected = []

    for dataset in datasets:
        rows = [_column_row(column) for column in dataset.columns_metadata]
        catalog_id = _register_catalog(rows, column_catalogs)
        projected.append(_dataset_projection(dataset, warnings, catalog_id))

    return column_catalogs, projected


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


def _register_catalog(rows, column_catalogs) -> str:
    for catalog_id, catalog in column_catalogs.items():
        if catalog == rows:
            return catalog_id

    catalog_id = f"columns_{len(column_catalogs) + 1}"
    column_catalogs[catalog_id] = rows

    return catalog_id


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


def _temporal_requirements(temporal_candidates_excluded) -> dict:
    return {
        "operation": "latest",
        "date_types": ["DATE", "TIMESTAMP"],
        "keep_all_ties": True,
        "exclude_null_measurements_before_latest": True,
        "candidates_without_verified_dates": list(temporal_candidates_excluded),
    }


def _text_period_column(item) -> dict:
    return {
        "dataset_id": str(item.dataset_id),
        "column": item.column,
        "precisions": item.precisions,
    }


def _serialized(evidence) -> bytes:
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _truncate_descriptions(projected) -> list[str]:
    compacted = []

    for dataset in projected:
        description = dataset["description"] or ""
        if len(description) > _MAX_DESCRIPTION_LENGTH:
            dataset["description"] = description[:_MAX_DESCRIPTION_LENGTH] + " [truncado]"
            compacted.append(f"{dataset['dataset_id']}.description")

    return compacted


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
