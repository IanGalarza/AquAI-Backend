"""Seguimientos: qué conserva, reemplaza o reutiliza un plan del análisis verificado que cita."""

from uuid import UUID

from pydantic import ValidationError

from aquai.application.contracts.planning import (
    ColumnRef,
    GroupingSpec,
    PlannableDecision,
    ResultRef,
)
from aquai.application.matching import contains_any_phrase, contains_phrase
from aquai.application.Rules.vocabulary import (
    CHART_TYPE_WORDS,
    MEASURED_STATISTICS,
    TIME_GRAIN_CUES,
)

_VALUE_OPERATORS = {"eq", "ieq", "in"}
_TEMPORAL_TYPES = {"DATE", "TIMESTAMP"}
_NUMERIC_TYPES = {"NUMERIC", "BIGINT"}
# Una fecha agrupada se rotula por su período, no como un día concreto.
_GRAIN_LABELS = {"day": "Día", "month": "Mes", "year": "Año"}
# Cálculos que un seguimiento sin otra estadística conserva y la validación exige.
_CONSERVED_STATISTICS = {"mean", "median", "sum", "stddev", "count"}
_STEP_FIELDS = (
    "id",
    "tool",
    "operation",
    "depends_on",
    "columns",
    "filters",
    "group_by",
    "calculations",
    "order_by",
    "limit",
)


def previous_operations(analysis) -> set[str]:
    """Operaciones del resultado citado: un latest o un gráfico se conservan aunque su pedido ya no esté en el texto."""
    return {str(item.get("operation")) for item in analysis.operations} if analysis else set()


def requested_time_grain(question: str) -> str | None:
    return next(
        (grain for grain, cues in TIME_GRAIN_CUES.items() if contains_any_phrase(question, cues)),
        None,
    )


def conserved_time_grain(analysis) -> str | None:
    """Agrupación temporal del resultado citado («por mes»), si fue una sola."""
    if analysis is None:
        return None

    grains = {
        grouping.get("time_grain")
        for item in analysis.operations
        for grouping in item.get("group_by") or []
    } - {None, "raw"}
    return grains.pop() if len(grains) == 1 else None


def conserved_statistics(analysis, question) -> list[str]:
    """«¿Y la turbiedad?» después de un promedio pide el mismo cálculo; nombrar otra estadística lo reemplaza."""
    if analysis is None or contains_any_phrase(question, MEASURED_STATISTICS):
        return []

    functions = [
        calculation.get("function")
        for step in analysis.operations
        for calculation in step.get("calculations") or []
    ]
    return [item for item in dict.fromkeys(functions) if item in _CONSERVED_STATISTICS]


def stale_values(analysis, mentions) -> list[str]:
    """Valores que filtraban el resultado citado y la consulta vigente ya no nombra: se reemplazaron o se quitaron.

    «¿Y en Punta Piedras?» después de Punta Lara: PUNTA LARA deja de ser un filtro aunque el modelo lo copie."""
    if analysis is None:
        return []

    named = [item.text for item in mentions if item.kind == "value"]
    values = [
        value
        for step in analysis.operations
        if step.get("operation") == "read"
        for predicate in step.get("filters") or []
        if predicate.get("operator") in _VALUE_OPERATORS
        for value in predicate.get("values") or []
        if isinstance(value, str)
    ]
    return [value for value in dict.fromkeys(values) if not _names_value(value, named)]


def _names_value(value, mentions) -> bool:
    return any(contains_phrase(value, text) or contains_phrase(text, value) for text in mentions)


def without_stale_filters(decision, stale):
    """Quita de los read los filtros de igualdad cuyos valores son todos del alcance reemplazado."""
    if not stale or decision.status != "plannable":
        return decision

    known = {value.casefold() for value in stale}
    changed = decision.model_copy(deep=True)
    modified = False
    for step in changed.steps:
        if step.operation != "read":
            continue
        kept = [item for item in step.filters if not _stale(item, known)]
        modified = modified or len(kept) != len(step.filters)
        step.filters = kept
    return changed if modified else decision


def _stale(predicate, known) -> bool:
    texts = [value for value in predicate.values if isinstance(value, str)]
    return (
        predicate.operator in _VALUE_OPERATORS
        and bool(texts)
        and len(texts) == len(predicate.values)
        and all(value.casefold() in known for value in texts)
    )


def with_time_grain(decision, grain, column_types):
    """«Promedio por mes»: el group_by de una fecha exacta pasa a agruparla por mes.

    Sólo si ningún group_by usa ya esa agrupación; column_types son los tipos SQL por (dataset_id, sql_name)."""
    if grain is None or decision.status != "plannable":
        return decision

    steps = {step.id: step for step in decision.steps}
    groupings = [grouping for step in decision.steps for grouping in step.group_by]
    if any(grouping.time_grain == grain for grouping in groupings):
        return decision

    changed = decision.model_copy(deep=True)
    modified = False
    for step in changed.steps:
        for grouping in step.group_by:
            temporal = _field_type(grouping.field, steps, column_types) in _TEMPORAL_TYPES
            if grouping.time_grain == "raw" and temporal:
                grouping.time_grain = grain
                modified = True
    if not modified:
        modified = _grouped_periods(changed, grain, column_types)
    return changed if modified else decision


def _grouped_periods(decision, grain, column_types) -> bool:
    """Un paso que usa un campo que su aggregate no expone («mes») delata el group_by temporal omitido.

    Con la agrupación pedida o conservada, ese campo es la única fecha del dataset agrupada por ese período."""
    steps = {step.id: step for step in decision.steps}
    modified = False
    for ref in (ref for step in decision.steps for ref in _used_fields(step)):
        target = steps.get(ref.step_id)
        if target is None or target.operation != "aggregate" or len(target.depends_on) != 1:
            continue
        exposed = {item.output_name for item in [*target.group_by, *target.calculations]}
        read = steps.get(target.depends_on[0])
        column = _only_date(read, decision, column_types) if ref.field not in exposed else None
        if column is None:
            continue

        target.group_by.append(
            GroupingSpec(
                field=ResultRef(kind="result", step_id=read.id, field=column.column),
                output_name=ref.field,
                time_grain=grain,
            )
        )
        modified = True
    return modified


def _used_fields(step):
    refs = [item.field for item in [*step.order_by, *step.filters]]
    if step.visualization is not None:
        refs += [getattr(step.visualization, name, None) for name in ("x", "series")]
    return [ref for ref in refs if isinstance(ref, ResultRef)]


def _only_date(read, decision, column_types):
    """La única columna DATE/TIMESTAMP del dataset del read; se agrega a la lectura si no la incluía."""
    if read is None or read.operation != "read" or not read.columns:
        return None

    dataset_id = read.columns[0].dataset_id
    dates = [
        column
        for (dataset, column), sql_type in column_types.items()
        if dataset == dataset_id and sql_type in _TEMPORAL_TYPES
    ]
    if len(dates) != 1:
        return None

    column = ColumnRef(kind="column", dataset_id=dataset_id, column=dates[0])
    if column not in read.columns:
        read.columns.append(column)
        for selection in decision.datasets:
            if selection.dataset_id == dataset_id and column.column not in selection.columns:
                selection.columns.append(column.column)
    return column


def _field_type(ref, steps, column_types):
    """Tipo de la columna leída que llega a un campo a través de sort y latest."""
    if isinstance(ref, ColumnRef):
        return column_types.get((ref.dataset_id, ref.column))

    step = steps.get(ref.step_id)
    while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
        step = steps.get(step.depends_on[0])
    if step is None or step.operation != "read":
        return None

    column = next((item for item in step.columns if item.column == ref.field), None)
    return column_types.get((column.dataset_id, column.column)) if column else None


def reused_decision(analysis, catalog, question):
    """«Ahora mostralo en un gráfico»: las mismas operaciones verificadas y un visualize sobre su resultado.

    Sólo si el resultado citado es una única tabla de varias filas con una categoría o fecha y un valor numérico
    y sus datasets conservan la versión verificada; si no, el Planner vuelve a planificar."""
    if analysis is None or not analysis.complete or analysis.status != "validated":
        return None
    if not _same_versions(analysis.sources, catalog):
        return None

    steps = {item["id"]: item for item in analysis.operations if item["operation"] != "visualize"}
    outputs = list(
        dict.fromkeys(
            item.get("operation_id")
            for item in analysis.operation_references
            if item.get("operation_id") in steps
        )
    )
    if len(outputs) != 1:
        return None

    target = steps[outputs[0]]
    chart = _chart(target, steps, question, catalog)
    if chart is None:
        return None

    chain = [steps[key] for key in steps if key in _ancestry(target["id"], steps)]
    if any(item.get("limit") == 1 for item in chain):
        return None  # Una única fila no se grafica.

    data = {
        "status": "plannable",
        "reason": "Se grafica el resultado verificado anterior con sus mismas operaciones.",
        "datasets": _selections(chain, analysis.sources),
        "steps": [*(_proposed(item) for item in chain), chart],
        "output_steps": [target["id"], chart["id"]],
        "warnings": [
            "Se reutilizaron las operaciones verificadas del resultado anterior sin planificar de nuevo; "
            "los datos se consultaron otra vez sobre la versión vigente."
        ],
    }
    try:
        return PlannableDecision.model_validate(data, strict=False)
    except ValidationError:
        return None


def _same_versions(sources, catalog) -> bool:
    versions = {item.dataset_id: item.verified_version for item in catalog}
    return bool(sources) and all(
        versions.get(UUID(str(item.get("dataset_id")))) == item.get("version") for item in sources
    )


def _ancestry(key, steps) -> set[str]:
    found, pending = set(), [key]
    while pending:
        item = pending.pop()
        if item in steps and item not in found:
            found.add(item)
            pending.extend(steps[item].get("depends_on") or [])
    return found


def _proposed(step) -> dict:
    proposed = {key: step[key] for key in _STEP_FIELDS if key in step}
    proposed["purpose"] = "Repite el paso verificado del resultado anterior."
    return proposed


def _selections(chain, sources) -> list[dict]:
    columns = {}
    reads = [step for step in chain if step["operation"] == "read"]
    for ref in (ref for step in reads for ref in step.get("columns") or []):
        columns.setdefault(str(ref["dataset_id"]), []).append(ref["column"])

    return [
        {
            "dataset_id": str(source["dataset_id"]),
            "version": source["version"],
            "columns": list(dict.fromkeys(columns[str(source["dataset_id"])])),
            "reason": "Mismo dataset del resultado verificado que se grafica.",
        }
        for source in sources
        if str(source["dataset_id"]) in columns
    ]


def _chart(target, steps, question, catalog):
    fields = target.get("output_fields") or []
    origin = _origin(target, steps)
    if origin is None or (origin["operation"] == "aggregate" and not origin.get("group_by")):
        return None  # Un único valor no se grafica.

    grouped = [item.get("output_name") for item in origin.get("group_by") or []]
    x = _category(fields, grouped)
    measures = [
        item
        for item in fields
        if item["numeric_eligible"] and item is not x and item["name"] not in grouped
    ]
    y = measures[0] if measures else None
    kind = _chart_type(question, x)
    if x is None or y is None or kind is None:
        return None

    labels = _labels(catalog)
    grains = {item.get("output_name"): item.get("time_grain") for item in origin.get("group_by") or []}
    x_label = _GRAIN_LABELS.get(grains.get(x["name"])) or labels.get(_source(x), x["name"])
    y_label = labels.get(_source(y), y["name"])
    return {
        "id": _unused_id("visualize_result", steps),
        "tool": "python",
        "operation": "visualize",
        "depends_on": [target["id"]],
        "visualization": {
            "chart_type": kind,
            "title": f"{y_label} por {x_label}",
            "x": {"kind": "result", "step_id": target["id"], "field": x["name"]},
            "y": {"kind": "result", "step_id": target["id"], "field": y["name"]},
            "x_label": x_label,
            "y_label": y_label,
        },
        "purpose": "Graficar el resultado verificado anterior sin recalcularlo.",
    }


def _origin(step, steps):
    while step is not None and step["operation"] in {"sort", "latest"} and step.get("depends_on"):
        step = steps.get(step["depends_on"][0])
    return step


def _category(fields, grouped):
    """El eje x es la agrupación del resultado; sin agrupar, su fecha o su primera categoría de texto."""
    by_name = {item["name"]: item for item in fields}
    candidates = [
        *(by_name[name] for name in grouped if name in by_name),
        *(item for item in fields if item["sql_type"] in _TEMPORAL_TYPES),
        *(item for item in fields if item["sql_type"] == "TEXT"),
    ]
    return candidates[0] if candidates else None


def _chart_type(question, x):
    """El tipo pedido, si los ejes lo admiten; sin pedido, línea para fechas y barras para categorías."""
    temporal = x is not None and x["sql_type"] in _TEMPORAL_TYPES
    numeric = x is not None and x["sql_type"] in _NUMERIC_TYPES
    asked = next(
        (kind for kind, words in CHART_TYPE_WORDS.items() if contains_any_phrase(question, words)),
        None,
    )
    if asked is None:
        return "line" if temporal else "bar"
    if asked == "line" and not (temporal or numeric):
        return None
    if asked in {"scatter", "histogram"}:
        return None  # Requieren otros ejes: los decide el Planner.
    return asked


def _labels(catalog) -> dict[tuple, str]:
    return {
        (str(dataset.dataset_id), column.sql_name): column.original_name
        for dataset in catalog
        for column in dataset.columns_metadata
    }


def _source(field) -> tuple | None:
    sources = field.get("sources") or []
    if len(sources) != 1:
        return None
    return str(sources[0].get("dataset_id")), sources[0].get("column")


def _unused_id(base, steps) -> str:
    key, index = base, 1
    while key in steps:
        index += 1
        key = f"{base}_{index}"
    return key
