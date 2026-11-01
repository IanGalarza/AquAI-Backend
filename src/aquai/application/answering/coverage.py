"""Cobertura de un resultado: datasets que registran lo pedido y no participan de la respuesta, y unidades que
impiden comparar los resultados de distintos archivos."""

from datetime import date

from aquai.application.answering.presentation import format_unit
from aquai.application.answering.wording import dataset_label, file_stem, join_words
from aquai.application.contracts.planning import MAX_FILTER_VALUES
from aquai.application.conversations.context import cited_analysis
from aquai.application.matching import variable_name
from aquai.application.planning.context import period_phrase
from aquai.application.planning.temporalText import within_period
from aquai.application.planning.validation import period_columns
from aquai.application.units import comparable_unit


def period_notes(state, language) -> list[str]:
    """«¿Y de qué año es?»: los archivos del resultado citado cuyo período no se pudo leer (fechas guardadas como
    texto, varias columnas de fecha) se nombran, para no presentar como completa una respuesta parcial."""
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


def unit_notes(state, language) -> list[str]:
    """Una misma variable registrada en unidades o bases químicas distintas según el archivo (mg N/L frente a mg/l sin
    base declarada) no se convierte ni se compara: los resultados de cada archivo quedan en su unidad y se advierte que
    no son directamente comparables."""
    retrieval, plan, validation = state.retrieval, state.planning, state.validation
    if retrieval is None or plan is None or validation is None:
        return []
    if state.routing is not None and state.routing.requested_period:
        return []  # La respuesta sólo informa períodos, no compara valores.

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


def _original_name(datasets, ref) -> str:
    dataset = datasets.get(ref.dataset_id)
    columns = dataset.columns_metadata if dataset else []
    return next((item.original_name for item in columns if item.sql_name == ref.column), ref.column)


def _basis_text(basis, language) -> str:
    if basis is None:
        return "sin base declarada" if language == "es" else "basis not declared"
    return ("como " if language == "es" else "as ") + basis


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


def coverage_notes(state, language) -> list[str]:
    """Una cobertura parcial no se presenta como completa: se nombran los datasets con las variables y los valores
    pedidos que el resultado no incluye y, si se conoce, por qué. Una fuente elegida por el usuario o heredada de
    un resultado anterior no se compara con el resto del catálogo."""
    retrieval, plan, routing = state.retrieval, state.planning, state.routing
    if retrieval is None or plan is None or routing is None:
        return []
    if retrieval.retrieval_method != "semantic":
        return []

    mentions = [item for item in routing.data_mentions if not isinstance(item, str)]
    variables = list(dict.fromkeys(item.text for item in mentions if item.kind == "variable"))
    names = list(dict.fromkeys(variable_name(item) for item in variables))
    values = [
        *dict.fromkeys(
            item.text
            for item in mentions
            if item.kind == "value" and item.text not in retrieval.dimension_mentions
        ),
        *retrieval.confirmed_values,
    ]
    if not variables and not values:
        return []

    # Un dataset elegido cuya rama no se publicó tampoco forma parte de lo que la respuesta muestra.
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


def _dated_purpose(evidence, language) -> str | None:
    """Para qué hacía falta una fecha: acotar un período pedido o agrupar por día, mes o año."""
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


def _dated(dataset, evidence) -> bool:
    """Un período se acota con una fecha o un año; una agrupación por mes o día exige una fecha."""
    kinds = set(period_columns([dataset]).values())
    if evidence.get("requested_periods"):
        return bool(kinds)
    return "date" in kinds


def _text_measurements(dataset, evidence, variables) -> list[str]:
    """Columnas de las variables pedidas que el dataset guarda como texto sin representación numérica: no se pueden
    calcular («N-NO3» con valores como «<0,001»). Vacío si alguna variable tiene una columna numérica."""
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


def _stored_as_text(what, files, columns, language) -> str:
    names = list(dict.fromkeys(columns))
    listed, single = join_words(names, language), len(names) == 1
    if language == "es":
        stored = "está guardada" if single else "están guardadas"
        version = "no tiene una versión numérica" if single else "no tienen una versión numérica"
        return (
            f"También hay datos de {what} en {files}, pero no los incluí porque {listed} {stored} como texto "
            f"y {version}."
        )

    stored = "is stored" if single else "are stored"
    return (
        f"There is also {what} data in {files}, but I did not include it because {listed} {stored} as text "
        "without a numeric version."
    )


def _period_gap(dataset, text_periods, requested) -> str | None:
    """Por qué un dataset con fechas guardadas como texto no se acotó al período pedido: no tiene registros en él
    («outside») o tiene más fechas distintas en él de las que admite un filtro («crowded»)."""
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


def _bounds(period) -> tuple[date | None, date | None]:
    return tuple(date.fromisoformat(period[side]) if period.get(side) else None for side in ("start", "end"))


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


def _covers(evidence, variables, values) -> bool:
    return (
        evidence is not None
        and all(item in evidence.variable_matches for item in variables)
        and all(item in evidence.value_matches for item in values)
    )


def _requested(variables, values, language) -> str:
    if variables and values:
        connector = " de " if language == "es" else " in "
        return join_words(variables, language) + connector + join_words(values, language)

    return join_words(variables or values, language)


def _files(pairs, language) -> str:
    """Archivo y hoja: un mismo archivo puede tener varias tablas con la misma variable."""
    sheet = "hoja" if language == "es" else "sheet"
    names = [f"{file_stem(source)} ({sheet} {dataset_label(name)})" for name, source in pairs]

    return join_words(list(dict.fromkeys(names)), language)


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
