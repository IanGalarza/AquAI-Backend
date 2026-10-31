import json
import re
from hashlib import sha256

from aquai.application.contracts.planning import PlanningContextRecord, RequestedThreshold
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.matching import contains_phrase, numeric_comparisons
from aquai.application.Rules.vocabulary import (
    CHART_WORDS,
    OBSERVATION_NOUNS_EN,
    OBSERVATION_NOUNS_ES,
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


def latest_requested(question: str) -> bool:
    observation = OBSERVATION_NOUNS_ES
    english = OBSERVATION_NOUNS_EN

    return bool(
        re.search(
            rf"\b(?:(?:[uú]ltim[oa]s?|m[aá]s\s+recientes?)\s+{observation}|"
            rf"{observation}\s+m[aá]s\s+recientes?|(?:latest|most\s+recent|last)\s+{english})\b",
            question,
            re.IGNORECASE,
        )
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
    statistics=(),
    chart=False,
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
    if statistics:
        evidence["requested_statistics"] = list(statistics)
    if chart:
        evidence["requested_chart"] = True
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
