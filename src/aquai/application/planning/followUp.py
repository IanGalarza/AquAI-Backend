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
    PERIOD_QUESTION_CUES,
    TIME_GRAIN_CUES,
)

_VALUE_OPERATORS = {"eq", "ieq", "in"}
_TEMPORAL_TYPES = {"DATE", "TIMESTAMP"}
_NUMERIC_TYPES = {"NUMERIC", "BIGINT"}
_GRAIN_LABELS = {"day": "Día", "month": "Mes", "year": "Año"}
_GRAIN_FIELDS = {"day": "dia", "month": "mes", "year": "anio"}
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


#
# Devuelve las operaciones del análisis citado.
#
def previous_operations(analysis) -> set[str]:
    return {str(item.get("operation")) for item in analysis.operations} if analysis else set()


#
# Devuelve la agrupación temporal que pide la pregunta, si la hay.
#
def requested_time_grain(question: str) -> str | None:
    return next(
        (grain for grain, cues in TIME_GRAIN_CUES.items() if contains_any_phrase(question, cues)),
        None,
    )


#
# Devuelve la precisión del período que la pregunta pide conocer, si la hay.
#
def requested_period(question: str) -> str | None:
    return next(
        (
            precision
            for precision, cues in PERIOD_QUESTION_CUES.items()
            if contains_any_phrase(question, cues)
        ),
        None,
    )


#
# Devuelve la agrupación temporal del análisis citado si fue una sola.
#
def conserved_time_grain(analysis) -> str | None:
    if analysis is None:
        return None

    grains = {
        grouping.get("time_grain")
        for item in analysis.operations
        for grouping in item.get("group_by") or []
    } - {None, "raw"}
    return grains.pop() if len(grains) == 1 else None


#
# Devuelve las estadísticas del análisis citado que un seguimiento conserva.
#
def conserved_statistics(analysis, question) -> list[str]:
    if analysis is None or contains_any_phrase(question, MEASURED_STATISTICS):
        return []

    functions = [
        calculation.get("function")
        for step in analysis.operations
        for calculation in step.get("calculations") or []
    ]
    return [item for item in dict.fromkeys(functions) if item in _CONSERVED_STATISTICS]


#
# Devuelve los valores de filtro del análisis citado que la consulta vigente ya no nombra.
#
def stale_values(analysis, mentions) -> list[str]:
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


#
# Indica si algún texto mencionado nombra el valor.
#
def _names_value(value, mentions) -> bool:
    return any(contains_phrase(value, text) or contains_phrase(text, value) for text in mentions)


#
# Quita de las lecturas los filtros de igualdad cuyos valores pertenecen al alcance reemplazado.
#
def without_stale_filters(decision, stale):
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


#
# Indica si un filtro de igualdad solo contiene valores obsoletos.
#
def _stale(predicate, known) -> bool:
    texts = [value for value in predicate.values if isinstance(value, str)]
    return (
        predicate.operator in _VALUE_OPERATORS
        and bool(texts)
        and len(texts) == len(predicate.values)
        and all(value.casefold() in known for value in texts)
    )


#
# Cambia la agrupación por fecha exacta a la agrupación temporal pedida.
#
def with_time_grain(decision, grain, column_types):
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
    if not modified:
        modified = _undated_aggregates(changed, grain, column_types)
    return changed if modified else decision


#
# Agrega la fecha agrupada por período a los aggregate que no agrupan ninguna fecha.
#
def _undated_aggregates(decision, grain, column_types) -> bool:
    steps = {step.id: step for step in decision.steps}
    modified = False
    for step in decision.steps:
        if step.operation != "aggregate" or len(step.depends_on) != 1:
            continue
        if any(
            _field_type(item.field, steps, column_types) in _TEMPORAL_TYPES for item in step.group_by
        ):
            continue

        read = steps.get(step.depends_on[0])
        column = _only_date(read, decision, column_types)
        if column is None:
            continue

        exposed = {item.output_name for item in [*step.group_by, *step.calculations]}
        step.group_by.append(
            GroupingSpec(
                field=ResultRef(kind="result", step_id=read.id, field=column.column),
                output_name=_unused_id(_GRAIN_FIELDS[grain], exposed),
                time_grain=grain,
            )
        )
        modified = True
    return modified


#
# Agrega la agrupación temporal que otros pasos usan pero su aggregate no expone.
#
def _grouped_periods(decision, grain, column_types) -> bool:
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


#
# Devuelve las referencias a resultados que usan el orden, los filtros y el gráfico de un paso.
#
def _used_fields(step):
    refs = [item.field for item in [*step.order_by, *step.filters]]
    if step.visualization is not None:
        refs += [getattr(step.visualization, name, None) for name in ("x", "series")]
    return [ref for ref in refs if isinstance(ref, ResultRef)]


#
# Devuelve la única columna de fecha del dataset de la lectura, agregándola si falta.
#
def _only_date(read, decision, column_types):
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


#
# Devuelve el tipo SQL de la columna leída que llega a un campo.
#
def _field_type(ref, steps, column_types):
    if isinstance(ref, ColumnRef):
        return column_types.get((ref.dataset_id, ref.column))

    step = steps.get(ref.step_id)
    while step is not None and step.operation in {"sort", "latest"} and step.depends_on:
        step = steps.get(step.depends_on[0])
    if step is None or step.operation != "read":
        return None

    column = next((item for item in step.columns if item.column == ref.field), None)
    return column_types.get((column.dataset_id, column.column)) if column else None


#
# Construye un plan que grafica el resultado verificado anterior reutilizando sus operaciones.
#
def reused_decision(analysis, catalog, question):
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
        return None

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


#
# Construye un plan que repite las operaciones verificadas del resultado citado para leer sus fechas.
#
def repeated_decision(analysis, catalog):
    if analysis is None or not analysis.complete or analysis.status != "validated":
        return None
    if not _same_versions(analysis.sources, catalog):
        return None

    steps = {item["id"]: item for item in analysis.operations if item["operation"] != "visualize"}
    referenced = list(
        dict.fromkeys(
            item.get("operation_id")
            for item in analysis.operation_references
            if item.get("operation_id") in steps
        )
    )
    inputs = {key for output in referenced for key in _ancestry(output, steps) - {output}}
    outputs = [key for key in referenced if key not in inputs]
    if not outputs:
        return None

    used = {key for output in outputs for key in _ancestry(output, steps)}
    chain = [step for key, step in steps.items() if key in used]
    data = {
        "status": "plannable",
        "reason": "Se repiten las operaciones verificadas del resultado citado para leer su período.",
        "datasets": _selections(chain, analysis.sources),
        "steps": [_proposed(item) for item in chain],
        "output_steps": outputs,
        "warnings": [
            "Se reutilizaron las operaciones verificadas del resultado anterior sin planificar de nuevo; "
            "los datos se consultaron otra vez sobre la versión vigente."
        ],
    }
    try:
        return PlannableDecision.model_validate(data, strict=False)
    except ValidationError:
        return None


#
# Construye un plan que calcula la primera y última fecha de los registros del resultado citado.
#
def period_span_decision(analysis, catalog):
    if analysis is None or not analysis.complete or analysis.status != "validated":
        return None
    if not _same_versions(analysis.sources, catalog):
        return None

    dates = {}
    for dataset in catalog:
        columns = [
            item.sql_name for item in dataset.columns_metadata if item.sql_type in _TEMPORAL_TYPES
        ]
        if len(columns) == 1:
            dates[str(dataset.dataset_id)] = columns[0]

    reads = [item for item in analysis.operations if item["operation"] == "read"]
    used = {item["id"] for item in analysis.operations}
    steps, outputs = [], []
    for read in reads:
        dataset_id = str(read["columns"][0]["dataset_id"])
        if dataset_id not in dates:
            continue

        date = dates[dataset_id]
        known = {(str(item["dataset_id"]), item["column"]) for item in read["columns"]}
        added = [] if (dataset_id, date) in known else [
            {"kind": "column", "dataset_id": dataset_id, "column": date}
        ]
        span = _unused_id(f"periodo_{read['id']}", used)
        used.add(span)
        steps += [
            {**_proposed(read), "columns": [*read["columns"], *added]},
            _date_span(span, read["id"], date),
        ]
        outputs.append(span)
    if not outputs:
        return None

    data = {
        "status": "plannable",
        "reason": "Primera y última fecha de los registros que leyó el resultado citado.",
        "datasets": _selections(steps, analysis.sources),
        "steps": steps,
        "output_steps": outputs,
        "warnings": [
            "El período se calculó sobre las mismas lecturas verificadas del resultado anterior, "
            "consultadas otra vez sobre la versión vigente."
        ],
    }
    try:
        return PlannableDecision.model_validate(data, strict=False)
    except ValidationError:
        return None


#
# Construye el paso que calcula la primera y última fecha de una lectura.
#
def _date_span(step_id, read_id, column) -> dict:
    field = {"kind": "result", "step_id": read_id, "field": column}
    return {
        "id": step_id,
        "tool": "sql",
        "operation": "aggregate",
        "depends_on": [read_id],
        "calculations": [
            {"function": "min", "inputs": [field], "output_name": "primera_fecha"},
            {"function": "max", "inputs": [field], "output_name": "ultima_fecha"},
        ],
        "purpose": "Primera y última fecha de los registros del resultado citado.",
    }


#
# Devuelve los campos de un resultado que ubican sus filas en el tiempo.
#
def period_fields(step, steps, periods) -> list[str]:
    texts = {(item.dataset_id, item.column) for item in periods}
    origin = step
    while origin is not None and origin.operation in {"sort", "latest"} and origin.depends_on:
        origin = steps.get(origin.depends_on[0])
    grains = {
        item.output_name for item in (origin.group_by if origin else []) if item.time_grain != "raw"
    }

    return [
        field.name
        for field in step.output_fields
        if field.sql_type in _TEMPORAL_TYPES
        or field.name in grains
        or (
            field.sql_type == "TEXT"
            and len(field.sources) == 1
            and (field.sources[0].dataset_id, field.sources[0].column) in texts
        )
    ]


#
# Indica si las fuentes del análisis conservan las versiones verificadas del catálogo.
#
def _same_versions(sources, catalog) -> bool:
    versions = {item.dataset_id: item.verified_version for item in catalog}
    return bool(sources) and all(
        versions.get(UUID(str(item.get("dataset_id")))) == item.get("version") for item in sources
    )


#
# Devuelve el paso indicado y todos los pasos de los que depende.
#
def _ancestry(key, steps) -> set[str]:
    found, pending = set(), [key]
    while pending:
        item = pending.pop()
        if item in steps and item not in found:
            found.add(item)
            pending.extend(steps[item].get("depends_on") or [])
    return found


#
# Copia un paso verificado como propuesta de repetición con su propósito.
#
def _proposed(step) -> dict:
    proposed = {key: step[key] for key in _STEP_FIELDS if key in step}
    proposed["purpose"] = "Repite el paso verificado del resultado anterior."
    return proposed


#
# Construye la selección de datasets y columnas que usan las lecturas de la cadena.
#
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


#
# Construye el paso de gráfico para el resultado, o None si no se puede graficar.
#
def _chart(target, steps, question, catalog):
    fields = target.get("output_fields") or []
    origin = _origin(target, steps)
    if origin is None or (origin["operation"] == "aggregate" and not origin.get("group_by")):
        return None

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


#
# Devuelve el paso de origen de un resultado, saltando los sort y latest.
#
def _origin(step, steps):
    while step is not None and step["operation"] in {"sort", "latest"} and step.get("depends_on"):
        step = steps.get(step["depends_on"][0])
    return step


#
# Devuelve el campo que sirve de eje x: agrupación, fecha o primera categoría de texto.
#
def _category(fields, grouped):
    by_name = {item["name"]: item for item in fields}
    candidates = [
        *(by_name[name] for name in grouped if name in by_name),
        *(item for item in fields if item["sql_type"] in _TEMPORAL_TYPES),
        *(item for item in fields if item["sql_type"] == "TEXT"),
    ]
    return candidates[0] if candidates else None


#
# Devuelve el tipo de gráfico pedido si los ejes lo admiten, o uno por defecto.
#
def _chart_type(question, x):
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
        return None
    return asked


#
# Devuelve el nombre original de cada columna por dataset y columna.
#
def _labels(catalog) -> dict[tuple, str]:
    return {
        (str(dataset.dataset_id), column.sql_name): column.original_name
        for dataset in catalog
        for column in dataset.columns_metadata
    }


#
# Devuelve el dataset y la columna de origen de un campo, si es único.
#
def _source(field) -> tuple | None:
    sources = field.get("sources") or []
    if len(sources) != 1:
        return None
    return str(sources[0].get("dataset_id")), sources[0].get("column")


#
# Devuelve un identificador de paso que no esté en uso.
#
def _unused_id(base, steps) -> str:
    key, index = base, 1
    while key in steps:
        index += 1
        key = f"{base}_{index}"
    return key
