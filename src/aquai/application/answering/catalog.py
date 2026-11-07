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


#
# Construye los segmentos de respuesta que describen las fuentes, tablas y documentos cargados.
#
def catalog_segments(
    catalog, focus_dataset_ids=(), language="es", documents=()
) -> list[AnswerSegment]:
    focus = set(focus_dataset_ids)
    datasets = [dataset for source in catalog for dataset in source["datasets"]]
    available = [dataset for dataset in datasets if dataset["available"]]

    if not catalog and not documents:
        return [AnswerSegment(text=_no_sources(language))]
    if focus:
        return _focused_segments(catalog, focus, language)

    return _overview_segments(catalog, datasets, available, language, documents)


#
# Devuelve el aviso de que todavía no hay fuentes cargadas.
#
def _no_sources(language):
    if language == "es":
        return (
            "Todavía no hay fuentes cargadas en AquAI. "
            "Podés subir archivos CSV, XLSX, PDF o DOCX desde el panel de fuentes."
        )

    return (
        "There are no sources loaded in AquAI yet. "
        "You can upload CSV, XLSX, PDF or DOCX files from the sources panel."
    )


#
# Construye los segmentos de las tablas del catálogo que el usuario pidió ver.
#
def _focused_segments(catalog, focus, language):
    segments = []
    for source in catalog:
        labels = _tables(source["datasets"], language)
        for dataset, label in zip(source["datasets"], labels):
            if dataset["dataset_id"] in focus:
                segments.append(_table_segment(source, dataset, label, language))

    return segments


#
# Construye el segmento que describe una tabla: cantidad de filas y lista de columnas.
#
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


#
# Construye el resumen general de las fuentes cargadas y su cierre.
#
def _overview_segments(catalog, datasets, available, language, documents):
    header = _inventory(catalog, datasets, available, language, documents)
    notes = _state_notes(catalog, language)
    if notes:
        header += " " + sentence(join_words(notes, language)) + "."

    items = [_source_item(source, language) for source in catalog]
    segments = [AnswerSegment(text=header, items=items)]
    if documents:
        segments.append(_documents_segment(documents, language))
    if catalog:
        segments.append(AnswerSegment(text=_closing(language, documents)))

    return segments


#
# Construye la lista de documentos que se consultan desde el chat.
#
def _documents_segment(documents, language):
    if language == "es":
        text = "Documentos PDF y DOCX, que uso para responder e interpretar resultados:"
    else:
        text = "PDF and DOCX documents, which I use to answer and interpret results:"

    items = [f"{document['name']}{_source_state(document, language)}" for document in documents]

    return AnswerSegment(text=text, items=items)


#
# Devuelve la frase con el total de fuentes, tablas disponibles y documentos.
#
def _inventory(catalog, datasets, available, language, documents=()):
    partial = len(available) != len(datasets)
    if language == "es":
        sources = count_text(len(catalog), "fuente cargada", "fuentes cargadas")
        tables = count_text(len(available), "tabla disponible", "tablas disponibles")
        prepared = f", de {len(datasets)} preparadas" if partial else ""
        if documents:
            return _spanish_mixed_inventory(catalog, documents, tables, prepared)
        return f"Tenés {sources} con {tables} para consultar{prepared}."

    sources = count_text(len(catalog), "loaded source", "loaded sources", "en")
    tables = count_text(len(available), "queryable table", "queryable tables", "en")
    prepared = f" out of {len(datasets)} prepared" if partial else ""
    if documents:
        return _english_mixed_inventory(catalog, documents, tables, prepared)

    return f"You have {sources} with {tables}{prepared}."


#
# Devuelve la frase en español con el total de fuentes, separando planillas y documentos.
#
def _spanish_mixed_inventory(catalog, documents, tables, prepared):
    total = count_text(len(catalog) + len(documents), "fuente cargada", "fuentes cargadas")
    texts = count_text(len(documents), "documento", "documentos")
    if not catalog:
        return f"Tenés {total}: {texts}."

    sheets = count_text(len(catalog), "planilla", "planillas")

    return f"Tenés {total}: {sheets} con {tables} para consultar{prepared} y {texts}."


#
# Devuelve la frase en inglés con el total de fuentes, separando planillas y documentos.
#
def _english_mixed_inventory(catalog, documents, tables, prepared):
    total = count_text(len(catalog) + len(documents), "loaded source", "loaded sources", "en")
    texts = count_text(len(documents), "document", "documents", "en")
    if not catalog:
        return f"You have {total}: {texts}."

    sheets = count_text(len(catalog), "spreadsheet", "spreadsheets", "en")

    return f"You have {total}: {sheets} with {tables}{prepared} and {texts}."


#
# Devuelve las notas sobre fuentes desactivadas, en procesamiento o con error.
#
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


#
# Devuelve la línea de una fuente con su estado y el resumen de sus tablas.
#
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


#
# Devuelve el resumen de una tabla: su nombre, sus filas y si no está disponible.
#
def _table_summary(dataset, label, language):
    text = f"{label} ({_rows_text(dataset['row_count'], language)})"
    if dataset["available"]:
        return text

    return text + (", no disponible para consultas" if language == "es" else ", not queryable")


#
# Devuelve el estado de una fuente: desactivada, en procesamiento o con error.
#
def _source_state(source, language):
    es = language == "es"
    state = "" if source["is_active"] else (" (desactivada)" if es else " (inactive)")
    if source["processing_status"] in _PROCESSING_STATUSES:
        return state + (" (en procesamiento)" if es else " (processing)")
    if source["processing_status"] == "failed":
        return state + (" (no se pudo procesar)" if es else " (processing failed)")

    return state


#
# Devuelve la invitación final a preguntar por las variables o pedir cálculos.
#
def _closing(language, documents=()):
    if language == "es":
        target = "planilla" if documents else "fuente"
        return f"Podés preguntarme por las variables de una {target} o pedirme cálculos sobre sus datos."

    target = "spreadsheet" if documents else "source"
    return f"You can ask about the variables in a {target} or request calculations on its data."


#
# Devuelve el texto con la cantidad de filas, en singular o plural.
#
def _rows_text(total, language):
    es = language == "es"

    return count_text(total, "fila" if es else "row", "filas" if es else "rows", language)


#
# Devuelve la etiqueta de una columna con su unidad formateada.
#
def _column(column):
    unit = column.get("unit")
    label = column_label(column["original_name"], unit)
    if not unit or label == "pH":
        return label

    return f"{label} ({format_unit(unit)})"


#
# Devuelve las etiquetas de las tablas, numerando las que se repiten.
#
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
