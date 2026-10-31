"""Inventario del catálogo redactado desde PostgreSQL; el LLM no aporta hechos."""

from aquai.application.answering.presentation import format_unit
from aquai.application.answering.wording import column_label, count_text, dataset_label, join_words, sentence
from aquai.application.contracts.answering import AnswerSegment


def _column(column):
    unit = column.get("unit")
    label = column_label(column["original_name"], unit)
    return label + (f" ({format_unit(unit)})" if unit and label != "pH" else "")


def _tables(datasets, language):
    """Hoja de cada tabla, sin rangos de celdas; si una hoja tiene varias, se numeran."""
    labels = [dataset_label(item["name"]) for item in datasets]
    seen = {}
    result = []
    for label in labels:
        if labels.count(label) > 1:
            seen[label] = seen.get(label, 0) + 1
            label += f" · {'tabla' if language == 'es' else 'table'} {seen[label]}"
        result.append(label)
    return result


def catalog_segments(catalog, focus_dataset_ids=(), language="es") -> list[AnswerSegment]:
    """Resume fuentes, estados y tablas; con foco, enumera también sus columnas."""
    focus = set(focus_dataset_ids)
    es = language == "es"
    datasets = [dataset for source in catalog for dataset in source["datasets"]]
    available = [dataset for dataset in datasets if dataset["available"]]
    if not catalog:
        return [AnswerSegment(text="Todavía no hay fuentes cargadas en AquAI. Podés subir archivos CSV o XLSX desde el panel de fuentes."
                              if es else "There are no sources loaded in AquAI yet. You can upload CSV or XLSX files from the sources panel.")]
    if focus:
        segments = []
        for source in catalog:
            for dataset, label in zip(source["datasets"], _tables(source["datasets"], language)):
                if dataset["dataset_id"] not in focus:
                    continue
                rows = count_text(dataset["row_count"], "fila" if es else "row", "filas" if es else "rows", language)
                columns = count_text(len(dataset["columns"]), "columna" if es else "column", "columnas" if es else "columns", language)
                segments.append(AnswerSegment(
                    text=f"La tabla {label} del archivo {source['name']} tiene {rows} y {columns}:" if es else
                         f"The table {label} in the file {source['name']} has {rows} and {columns}:",
                    items=[_column(column) for column in dataset["columns"]]))
        return segments
    inactive = sum(not source["is_active"] for source in catalog)
    pending = sum(source["processing_status"] in {"uploaded", "processing"} for source in catalog)
    failed = sum(source["processing_status"] == "failed" for source in catalog)
    if es:
        header = (f"Tenés {count_text(len(catalog), 'fuente cargada', 'fuentes cargadas')} con "
                  f"{count_text(len(available), 'tabla disponible', 'tablas disponibles')} para consultar"
                  + (f", de {len(datasets)} preparadas" if len(available) != len(datasets) else "") + ".")
        states = (("fuente está desactivada", "fuentes están desactivadas"),
                  ("fuente sigue en procesamiento", "fuentes siguen en procesamiento"),
                  ("fuente no se pudo procesar", "fuentes no se pudieron procesar"))
    else:
        header = (f"You have {count_text(len(catalog), 'loaded source', 'loaded sources', 'en')} with "
                  f"{count_text(len(available), 'queryable table', 'queryable tables', 'en')}"
                  + (f" out of {len(datasets)} prepared" if len(available) != len(datasets) else "") + ".")
        states = (("source is inactive", "sources are inactive"), ("source is still processing", "sources are still processing"),
                  ("source could not be processed", "sources could not be processed"))
    notes = [count_text(total, singular, plural, language)
             for total, (singular, plural) in zip((inactive, pending, failed), states) if total]
    if notes:
        header += " " + sentence(join_words(notes, language)) + "."
    items = []
    for source in catalog:
        tables = join_words([
            f"{label} ({count_text(item['row_count'], 'fila' if es else 'row', 'filas' if es else 'rows', language)})"
            + ("" if item["available"] else (", no disponible para consultas" if es else ", not queryable"))
            for item, label in zip(source["datasets"], _tables(source["datasets"], language))], language)
        state = "" if source["is_active"] else (" (desactivada)" if es else " (inactive)")
        if source["processing_status"] in {"uploaded", "processing"}:
            state += " (en procesamiento)" if es else " (processing)"
        elif source["processing_status"] == "failed":
            state += " (no se pudo procesar)" if es else " (processing failed)"
        items.append(f"{source['name']}{state}: " + (tables or ("sin tablas" if es else "no tables")))
    closing = ("Podés preguntarme por las variables de una fuente o pedirme cálculos sobre sus datos."
               if es else "You can ask about the variables in a source or request calculations on its data.")
    return [AnswerSegment(text=header, items=items), AnswerSegment(text=closing)]
