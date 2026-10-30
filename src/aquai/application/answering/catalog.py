"""Inventario del catálogo redactado desde PostgreSQL; el LLM no aporta hechos."""

from aquai.application.units import is_ph_name


def _plural(count, singular, plural):
    return f"{count} {singular if count == 1 else plural}"


def _sheet(name):
    parts = [part.strip() for part in name.split(" / ")]
    return " / ".join(parts[1:]) if len(parts) >= 3 else name


def _column(column):
    name = " ".join(str(column["original_name"]).split())
    if is_ph_name(name):
        return "pH"
    return name + (f" [{column['unit']}]" if column.get("unit") and f"[{column['unit']}]" not in name else "")


def catalog_answer(catalog, focus_dataset_ids=(), language="es"):
    """Resume fuentes, estados y datasets; con foco, enumera también sus columnas."""
    focus = set(focus_dataset_ids)
    es = language == "es"
    datasets = [dataset for source in catalog for dataset in source["datasets"]]
    available = [dataset for dataset in datasets if dataset["available"]]
    if not catalog:
        return ("Todavía no hay fuentes cargadas en AquAI. Podés subir archivos CSV o XLSX desde el panel de fuentes."
                if es else "There are no sources loaded in AquAI yet. You can upload CSV or XLSX files from the sources panel.")
    if focus:
        lines = []
        for source in catalog:
            selected = [item for item in source["datasets"] if item["dataset_id"] in focus]
            for dataset in selected:
                columns = ", ".join(_column(column) for column in dataset["columns"])
                lines.append(
                    (f"«{_sheet(dataset['name'])}» de «{source['name']}» tiene {_plural(dataset['row_count'], 'fila', 'filas')} "
                     f"y {_plural(len(dataset['columns']), 'columna', 'columnas')}: {columns}.")
                    if es else
                    (f"'{_sheet(dataset['name'])}' in '{source['name']}' has {_plural(dataset['row_count'], 'row', 'rows')} "
                     f"and {_plural(len(dataset['columns']), 'column', 'columns')}: {columns}.")
                )
        return "\n\n".join(lines)
    active = [source for source in catalog if source["is_active"]]
    inactive = len(catalog) - len(active)
    pending = [source for source in catalog if source["processing_status"] in {"uploaded", "processing"}]
    failed = [source for source in catalog if source["processing_status"] == "failed"]
    if es:
        header = (f"Tenés {_plural(len(catalog), 'fuente cargada', 'fuentes cargadas')}"
                  f" con {_plural(len(available), 'dataset consultable', 'datasets consultables')}"
                  + (f" de {len(datasets)} preparados" if len(available) != len(datasets) else "") + ".")
        notes = []
        if inactive:
            notes.append(f"{_plural(inactive, 'fuente está desactivada', 'fuentes están desactivadas')}")
        if pending:
            notes.append(f"{_plural(len(pending), 'fuente sigue en procesamiento', 'fuentes siguen en procesamiento')}")
        if failed:
            notes.append(f"{_plural(len(failed), 'fuente no se pudo procesar', 'fuentes no se pudieron procesar')}")
        if notes:
            header += " " + "; ".join(notes).capitalize() + "."
    else:
        header = (f"You have {_plural(len(catalog), 'loaded source', 'loaded sources')} with "
                  f"{_plural(len(available), 'queryable dataset', 'queryable datasets')}.")
    lines = [header, ""]
    for source in catalog:
        items = "; ".join(
            f"{_sheet(item['name'])} ({_plural(item['row_count'], 'fila' if es else 'row', 'filas' if es else 'rows')})"
            + ("" if item["available"] else (" — no consultable" if es else " — not queryable"))
            for item in source["datasets"]
        )
        state = "" if source["is_active"] else (" (desactivada)" if es else " (inactive)")
        if source["processing_status"] in {"uploaded", "processing"}:
            state += " (en procesamiento)" if es else " (processing)"
        elif source["processing_status"] == "failed":
            state += " (falló el procesamiento)" if es else " (processing failed)"
        lines.append(f"• {source['name']}{state}: " + (items or ("sin datasets" if es else "no datasets")))
    lines.append("")
    lines.append("Podés preguntarme por las variables de una fuente o pedirme cálculos sobre sus datos."
                 if es else "You can ask about the variables in a source or request calculations on its data.")
    return "\n".join(lines)
