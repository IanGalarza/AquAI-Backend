from datetime import date

from aquai.application.answering.presentation import format_unit
from aquai.application.answering.wording import dataset_label, file_stem, join_words
from aquai.application.contracts.planning import MAX_FILTER_VALUES
from aquai.application.conversations.context import cited_analysis
from aquai.application.matching import variable_name
from aquai.application.planning.context import numeric_comparison, period_phrase
from aquai.application.planning.temporalText import within_period
from aquai.application.planning.validation import period_columns
from aquai.application.units import comparable_unit

_UNPLANNED_STATUSES = {"insufficient_coverage", "failed"}


#
# Devuelve el aviso de archivos del resultado citado cuyo período no se pudo leer.
#
def period_notes(state, language) -> list[str]:
    routing, plan = state.routing, state.planning
    if routing is None or plan is None or not routing.requested_period:
        return []

    analysis = cited_analysis(state.conversation_history, routing.history_message_ids)
    if analysis is None:
        return []

    used = {str(item.dataset_id) for item in plan.selected_datasets}
    missing = [
        file_stem(item["provenance"]["source_name"])
        for item in analysis.sources
        if str(item["dataset_id"]) not in used
    ]
    if not missing:
        return []

    files = join_words(list(dict.fromkeys(missing)), language)
    if language == "es":
        return [
            f"No pude indicar el período del resultado de {files}: sus registros no tienen una única "
            "columna de fecha que se pueda leer."
        ]

    return [
        f"I could not tell the period of the result from {files}: "
        "its records have no single readable date column."
    ]


#
# Devuelve avisos de variables registradas en unidades o bases distintas entre archivos.
#
def unit_notes(state, language) -> list[str]:
    retrieval, plan, validation = state.retrieval, state.planning, state.validation
    if retrieval is None or plan is None or validation is None:
        return []
    if state.routing is not None and state.routing.requested_period:
        return []

    mentions = {
        (item.dataset_id, column): mention
        for item in retrieval.evidence
        for mention, columns in item.variable_matches.items()
        for column in columns
    }
    steps = {step.id: step for step in plan.operations}
    datasets = {item.dataset_id: item for item in plan.selected_datasets}
    units = {}
    for evidence in validation.usable_evidence:
        if evidence.kind != "table" or not evidence.is_requested_output:
            continue
        for field in steps[evidence.operation_id].output_fields:
            if not field.unit or field.unit_status != "documented":
                continue
            for ref in field.sources:
                mention = mentions.get((ref.dataset_id, ref.column))
                if mention is not None:
                    key = comparable_unit(_original_name(datasets, ref), field.unit)
                    found = units.setdefault(mention, {}).setdefault(key, {})
                    found.setdefault(format_unit(field.unit), set()).add(ref.dataset_id)

    return [
        _different_units(variable_name(mention), by_unit, datasets, language)
        for mention, by_unit in units.items()
        if len(by_unit) > 1
    ]


#
# Devuelve el nombre original de una columna a partir de su referencia.
#
def _original_name(datasets, ref) -> str:
    dataset = datasets.get(ref.dataset_id)
    columns = dataset.columns_metadata if dataset else []
    return next((item.original_name for item in columns if item.sql_name == ref.column), ref.column)


#
# Devuelve el texto de la base química de una unidad, o que no está declarada.
#
def _basis_text(basis, language) -> str:
    if basis is None:
        return "sin base declarada" if language == "es" else "basis not declared"
    return ("como " if language == "es" else "as ") + basis


#
# Redacta el aviso de que los archivos registran una variable en unidades distintas.
#
def _different_units(mention, by_unit, datasets, language) -> str:
    bases = {basis for _, basis in by_unit}
    places = []
    for (_, basis), spellings in by_unit.items():
        unit = next(iter(spellings))
        if len(bases) > 1:
            unit += f" {_basis_text(basis, language)}"
        names = {
            file_stem(datasets[key].provenance.source_name)
            for keys in spellings.values()
            for key in keys
        }
        connector = " en " if language == "es" else " in "
        places.append(unit + connector + join_words(sorted(names), language))

    listed = "; ".join(places)
    if language == "es":
        kind = "unidades o bases químicas distintas" if len(bases) > 1 else "unidades distintas"
        return (
            f"Los archivos registran {mention} en {kind} ({listed}): no convertí los valores, "
            "así que los resultados de distintos archivos no son directamente comparables."
        )

    kind = "different units or chemical bases" if len(bases) > 1 else "different units"
    return (
        f"The files record {mention} in {kind} ({listed}): I did not convert the values, "
        "so results from different files are not directly comparable."
    )


#
# Devuelve avisos de datasets con lo pedido que el resultado no incluye y por qué.
#
def coverage_notes(state, language) -> list[str]:
    retrieval, plan, routing = state.retrieval, state.planning, state.routing
    if retrieval is None or plan is None or routing is None:
        return []
    if retrieval.retrieval_method != "semantic":
        return []

    variables, values = _asked(routing, retrieval)
    names = list(dict.fromkeys(variable_name(item) for item in variables))
    if not variables and not values:
        return []

    sources = state.validation.sources if state.validation is not None else plan.selected_datasets
    used = {item.dataset_id for item in sources}
    evidence = {item.dataset_id: item for item in retrieval.evidence}
    unselected = [
        candidate
        for candidate in retrieval.candidates
        if candidate.dataset_id not in used
        and _covers(evidence.get(candidate.dataset_id), variables, values)
    ]
    textual = {
        item.dataset_id: columns
        for item in unselected
        if (columns := _text_measurements(item, evidence.get(item.dataset_id), variables))
    }
    unselected = [item for item in unselected if item.dataset_id not in textual]
    requested = plan.context.evidence.get("requested_periods", [])
    gaps = {
        item.dataset_id: gap
        for item in unselected
        if (gap := _period_gap(item, plan.temporal_text_columns, requested))
    }
    unselected = [item for item in unselected if item.dataset_id not in gaps]
    purpose = _dated_purpose(plan.context.evidence, language)
    undated = [item for item in unselected if purpose and not _dated(item, plan.context.evidence)]
    omitted = retrieval.omitted_candidates
    unanswered = [
        (item.name, item.source_name) for item in omitted if item.reason == "missing_dimension"
    ]
    skipped = [
        *((item.name, item.provenance.source_name) for item in unselected if item not in undated),
        *((item.name, item.source_name) for item in omitted if item.reason == "beyond_limit"),
    ]
    what = _requested(names, values, language)

    notes = [
        _stored_as_text(what, _files([(item.name, item.provenance.source_name)], language), columns, language)
        for item in retrieval.candidates
        if (columns := textual.get(item.dataset_id))
    ]
    notes += [
        _outside_period(what, _files([(item.name, item.provenance.source_name)], language), gap, requested, language)
        for item in retrieval.candidates
        if (gap := gaps.get(item.dataset_id))
    ]
    if unanswered:
        dimensions = join_words(retrieval.dimension_mentions, language)
        notes.append(_without_dimension(what, _files(unanswered, language), dimensions, language))
    if undated:
        files = _files([(item.name, item.provenance.source_name) for item in undated], language)
        notes.append(_without_dates(what, files, purpose, language))
    if skipped:
        notes.append(_not_included(what, _files(skipped, language), language))
    return notes


#
# Devuelve los avisos de mediciones pedidas que sólo están como texto y por eso no se calcularon.
#
def text_measurement_notes(state, language) -> list[str]:
    retrieval, plan, routing = state.retrieval, state.planning, state.routing
    if retrieval is None or plan is None or routing is None:
        return []
    if plan.status not in _UNPLANNED_STATUSES:
        return []
    if not numeric_comparison(plan.context.evidence):
        return []

    variables, values = _asked(routing, retrieval)
    evidence = {item.dataset_id: item for item in retrieval.evidence}
    covering = [
        item
        for item in retrieval.candidates
        if _covers(evidence.get(item.dataset_id), variables, values)
    ]
    textual = [
        (item, columns)
        for item in covering
        if (columns := _text_measurements(item, evidence.get(item.dataset_id), variables))
    ]
    if len(textual) < len(covering):
        return []

    return [
        _not_calculated(_files([(item.name, item.provenance.source_name)], language), columns, language)
        for item, columns in textual
    ]


#
# Devuelve las variables y los valores que pide la consulta.
#
def _asked(routing, retrieval) -> tuple[list[str], list[str]]:
    mentions = [item for item in routing.data_mentions if not isinstance(item, str)]
    variables = list(dict.fromkeys(item.text for item in mentions if item.kind == "variable"))
    values = [
        *dict.fromkeys(
            item.text
            for item in mentions
            if item.kind == "value" and item.text not in retrieval.dimension_mentions
        ),
        *retrieval.confirmed_values,
    ]
    return variables, values


#
# Devuelve para qué hacía falta una fecha: acotar un período o agrupar por tiempo.
#
def _dated_purpose(evidence, language) -> str | None:
    periods = evidence.get("requested_periods", [])
    if periods:
        phrases = [period_phrase(item, language) for item in periods]
        verb = "acotarlos " if language == "es" else "restrict it "
        return verb + join_words(list(dict.fromkeys(phrases)), language)

    grain = evidence.get("requested_time_grain")
    if grain is None:
        return None
    if language == "es":
        return "agruparlos por " + {"day": "día", "month": "mes", "year": "año"}[grain]
    return "group it by " + grain


#
# Indica si un dataset tiene la fecha o el año que exige la consulta.
#
def _dated(dataset, evidence) -> bool:
    kinds = set(period_columns([dataset]).values())
    if evidence.get("requested_periods"):
        return bool(kinds)
    return "date" in kinds


#
# Devuelve las columnas pedidas guardadas como texto sin versión numérica.
#
def _text_measurements(dataset, evidence, variables) -> list[str]:
    if evidence is None or not variables:
        return []

    columns = {column.sql_name: column for column in dataset.columns_metadata}
    projected = {column.derived_from for column in dataset.columns_metadata if column.numeric_eligible}
    matched = [
        columns[name]
        for variable in variables
        for name in evidence.variable_matches.get(variable, [])
        if name in columns
    ]
    textual = [
        column
        for column in matched
        if column.sql_type == "TEXT" and not column.numeric_eligible and column.sql_name not in projected
    ]
    return [column.original_name for column in textual] if matched and len(textual) == len(matched) else []


#
# Redacta el aviso de datos guardados como texto sin versión numérica.
#
def _stored_as_text(what, files, columns, language) -> str:
    storage = _text_storage(columns, language)
    if language == "es":
        return f"También hay datos de {what} en {files}, pero no los incluí porque {storage}."

    return f"There is also {what} data in {files}, but I did not include it because {storage}."


#
# Redacta el aviso de mediciones guardadas como texto con las que no se pudo calcular lo pedido.
#
def _not_calculated(files, columns, language) -> str:
    storage = _text_storage(columns, language)
    if language == "es":
        return f"En {files}, {storage}, así que no pude calcular lo pedido."

    return f"In {files}, {storage}, so I could not calculate what you asked for."


#
# Redacta que las columnas están guardadas como texto, sin versión numérica.
#
def _text_storage(columns, language) -> str:
    names = list(dict.fromkeys(columns))
    listed, single = join_words(names, language), len(names) == 1
    if language == "es":
        stored = "está guardada" if single else "están guardadas"
        version = "no tiene una versión numérica" if single else "no tienen una versión numérica"
        return f"{listed} {stored} como texto y {version}"

    stored = "is stored" if single else "are stored"
    return f"{listed} {stored} as text without a numeric version"


#
# Indica por qué un dataset con fechas en texto no se acotó al período pedido.
#
def _period_gap(dataset, text_periods, requested) -> str | None:
    columns = [
        item
        for item in text_periods
        if item.dataset_id == dataset.dataset_id and item.values and item.complete
    ]
    if not requested or len(columns) != 1:
        return None

    column = columns[0]
    counts = [
        sum(within_period(value, column.day_first, *_bounds(period)) for value in column.values)
        for period in requested
    ]
    if not any(counts):
        return "outside"
    return "crowded" if max(counts) > MAX_FILTER_VALUES else None


#
# Devuelve las fechas de inicio y fin de un período pedido.
#
def _bounds(period) -> tuple[date | None, date | None]:
    return tuple(date.fromisoformat(period[side]) if period.get(side) else None for side in ("start", "end"))


#
# Redacta el aviso de datos que no se incluyeron por el período pedido.
#
def _outside_period(what, files, gap, requested, language) -> str:
    periods = join_words(list(dict.fromkeys(item["text"] for item in requested)), language)
    if language == "es":
        reason = (
            f"no tienen registros en {periods}"
            if gap == "outside"
            else f"sus fechas, guardadas como texto, tienen demasiados valores distintos en {periods} para acotarlas"
        )
        return f"También hay datos de {what} en {files}, pero no los incluí porque {reason}."

    reason = (
        f"it has no records in {periods}"
        if gap == "outside"
        else f"its dates, stored as text, have too many distinct values in {periods} to restrict them"
    )
    return f"There is also {what} data in {files}, but I did not include it because {reason}."


#
# Indica si la evidencia de un dataset cubre las variables y valores pedidos.
#
def _covers(evidence, variables, values) -> bool:
    return (
        evidence is not None
        and all(item in evidence.variable_matches for item in variables)
        and all(item in evidence.value_matches for item in values)
    )


#
# Devuelve el texto de lo pedido: variables y valores.
#
def _requested(variables, values, language) -> str:
    if variables and values:
        connector = " de " if language == "es" else " in "
        return join_words(variables, language) + connector + join_words(values, language)

    return join_words(variables or values, language)


#
# Devuelve los archivos con su hoja como texto.
#
def _files(pairs, language) -> str:
    sheet = "hoja" if language == "es" else "sheet"
    names = [f"{file_stem(source)} ({sheet} {dataset_label(name)})" for name, source in pairs]

    return join_words(list(dict.fromkeys(names)), language)


#
# Redacta el aviso de datos sin la dimensión pedida.
#
def _without_dimension(what, files, dimensions, language) -> str:
    if language == "es":
        return (
            f"También hay datos de {what} en {files}, pero no los incluí porque no tienen "
            f"una columna de {dimensions}."
        )

    return (
        f"There is also {what} data in {files}, but I did not include it "
        f"because it has no {dimensions} column."
    )


#
# Redacta el aviso de datos sin fecha para el período o agrupación pedidos.
#
def _without_dates(what, files, purpose, language) -> str:
    if language == "es":
        return (
            f"También hay datos de {what} en {files}, pero no los incluí porque no tienen una fecha "
            f"con la que {purpose}."
        )

    return (
        f"There is also {what} data in {files}, but I did not include it "
        f"because it has no date to {purpose}."
    )


#
# Redacta el aviso de datos que el resultado no incluye.
#
def _not_included(what, files, language) -> str:
    if language == "es":
        return (
            f"También hay datos de {what} en {files}, que este resultado no incluye; "
            "podés consultarlos nombrando el archivo."
        )

    return (
        f"There is also {what} data in {files}, which this result does not include; "
        "you can query it by naming the file."
    )
