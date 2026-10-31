from aquai.application.answering.presentation import format_unit
from aquai.application.answering.wording import (
    column_label,
    count_text,
    dataset_label,
    join_words,
    sentence,
)
from aquai.application.contracts.answering import AnswerSegment

_PROCESSING_STATUSES = {"uploaded", "processing"}
_SPANISH_STATES = (
    ("fuente está desactivada", "fuentes están desactivadas"),
    ("fuente sigue en procesamiento", "fuentes siguen en procesamiento"),
    ("fuente no se pudo procesar", "fuentes no se pudieron procesar"),
)
_ENGLISH_STATES = (
    ("source is inactive", "sources are inactive"),
    ("source is still processing", "sources are still processing"),
    ("source could not be processed", "sources could not be processed"),
)


def catalog_segments(catalog, focus_dataset_ids=(), language="es") -> list[AnswerSegment]:
    focus = set(focus_dataset_ids)
    datasets = [dataset for source in catalog for dataset in source["datasets"]]
    available = [dataset for dataset in datasets if dataset["available"]]

    if not catalog:
        return [AnswerSegment(text=_no_sources(language))]
    if focus:
        return _focused_segments(catalog, focus, language)

    return _overview_segments(catalog, datasets, available, language)


def _no_sources(language):
    if language == "es":
        return (
            "Todavía no hay fuentes cargadas en AquAI. "
            "Podés subir archivos CSV o XLSX desde el panel de fuentes."
        )

    return (
        "There are no sources loaded in AquAI yet. "
        "You can upload CSV or XLSX files from the sources panel."
    )


def _focused_segments(catalog, focus, language):
    segments = []
    for source in catalog:
        labels = _tables(source["datasets"], language)
        for dataset, label in zip(source["datasets"], labels):
            if dataset["dataset_id"] in focus:
                segments.append(_table_segment(source, dataset, label, language))

    return segments


def _table_segment(source, dataset, label, language):
    es = language == "es"
    rows = _rows_text(dataset["row_count"], language)
    columns = count_text(
        len(dataset["columns"]),
        "columna" if es else "column",
        "columnas" if es else "columns",
        language,
    )
    if es:
        text = f"La tabla {label} del archivo {source['name']} tiene {rows} y {columns}:"
    else:
        text = f"The table {label} in the file {source['name']} has {rows} and {columns}:"

    return AnswerSegment(text=text, items=[_column(column) for column in dataset["columns"]])


def _overview_segments(catalog, datasets, available, language):
    header = _inventory(catalog, datasets, available, language)
    notes = _state_notes(catalog, language)
    if notes:
        header += " " + sentence(join_words(notes, language)) + "."

    items = [_source_item(source, language) for source in catalog]

    return [AnswerSegment(text=header, items=items), AnswerSegment(text=_closing(language))]


def _inventory(catalog, datasets, available, language):
    partial = len(available) != len(datasets)
    if language == "es":
        sources = count_text(len(catalog), "fuente cargada", "fuentes cargadas")
        tables = count_text(len(available), "tabla disponible", "tablas disponibles")
        prepared = f", de {len(datasets)} preparadas" if partial else ""
        return f"Tenés {sources} con {tables} para consultar{prepared}."

    sources = count_text(len(catalog), "loaded source", "loaded sources", "en")
    tables = count_text(len(available), "queryable table", "queryable tables", "en")
    prepared = f" out of {len(datasets)} prepared" if partial else ""

    return f"You have {sources} with {tables}{prepared}."


def _state_notes(catalog, language):
    inactive = sum(not source["is_active"] for source in catalog)
    pending = sum(source["processing_status"] in _PROCESSING_STATUSES for source in catalog)
    failed = sum(source["processing_status"] == "failed" for source in catalog)
    states = _SPANISH_STATES if language == "es" else _ENGLISH_STATES
    notes = []
    for total, (singular, plural) in zip((inactive, pending, failed), states):
        if total:
            notes.append(count_text(total, singular, plural, language))

    return notes


def _source_item(source, language):
    labels = _tables(source["datasets"], language)
    summaries = [
        _table_summary(dataset, label, language)
        for dataset, label in zip(source["datasets"], labels)
    ]
    tables = join_words(summaries, language)
    if not tables:
        tables = "sin tablas" if language == "es" else "no tables"

    return f"{source['name']}{_source_state(source, language)}: " + tables


def _table_summary(dataset, label, language):
    text = f"{label} ({_rows_text(dataset['row_count'], language)})"
    if dataset["available"]:
        return text

    return text + (", no disponible para consultas" if language == "es" else ", not queryable")


def _source_state(source, language):
    es = language == "es"
    state = "" if source["is_active"] else (" (desactivada)" if es else " (inactive)")
    if source["processing_status"] in _PROCESSING_STATUSES:
        return state + (" (en procesamiento)" if es else " (processing)")
    if source["processing_status"] == "failed":
        return state + (" (no se pudo procesar)" if es else " (processing failed)")

    return state


def _closing(language):
    if language == "es":
        return (
            "Podés preguntarme por las variables de una fuente o pedirme cálculos sobre sus datos."
        )

    return "You can ask about the variables in a source or request calculations on its data."


def _rows_text(total, language):
    es = language == "es"

    return count_text(total, "fila" if es else "row", "filas" if es else "rows", language)


def _column(column):
    unit = column.get("unit")
    label = column_label(column["original_name"], unit)
    if not unit or label == "pH":
        return label

    return f"{label} ({format_unit(unit)})"


def _tables(datasets, language):
    labels = [dataset_label(item["name"]) for item in datasets]
    word = "tabla" if language == "es" else "table"
    numbers = {}
    result = []
    for label in labels:
        if labels.count(label) > 1:
            numbers[label] = numbers.get(label, 0) + 1
            label += f" · {word} {numbers[label]}"
        result.append(label)

    return result
