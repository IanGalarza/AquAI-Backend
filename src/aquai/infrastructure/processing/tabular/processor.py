import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from hashlib import sha256
from pathlib import PurePath

from openpyxl.utils.cell import get_column_letter

from aquai.application.contracts.ingestion import (
    PreparationResult,
    PreparedTable,
    StructureSuggestion,
)
from aquai.application.errors import FileTooLargeError
from aquai.application.Rules.ingestion import NOTE_ROW, TOTAL_ROW_LABELS
from aquai.application.Rules.units import STACKED_HEADER_SEPARATOR
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.normalization import (
    column_names,
    header_unit,
    infer_column,
    numeric_projection,
    text_value,
)
from aquai.infrastructure.processing.tabular.reader import (
    LoadedFile,
    SheetData,
    TableReader,
    is_empty,
)

_Bounds = tuple[int, int, int, int]

_PLAIN_NUMBER = re.compile(r"[+-]?\d+(?:[.,]\d+)?")


@dataclass
class _RowScan:
    rows: list[list[object]] = field(default_factory=list)
    source_rows: list[int] = field(default_factory=list)
    omitted_notes: int = 0
    omitted_totals: int = 0
    formula_only_rows: int = 0


#
# Agrupa índices ordenados en bloques de posiciones consecutivas.
#
def _consecutive_groups(indices: list[int]) -> list[list[int]]:
    groups = []

    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(index)

    return groups


#
# Devuelve el rango en notación de hoja de cálculo para unos límites de región.
#
def _range_reference(bounds: _Bounds) -> str:
    left, top, right, bottom = bounds
    return f"{get_column_letter(left)}{top}:{get_column_letter(right)}{bottom}"


#
# Construye el nombre de una tabla a partir del archivo, la hoja y el rango.
#
def _table_name(filename: str, sheet_name: str, range_reference: str) -> str:
    return f"{PurePath(filename).stem} / {sheet_name} / {range_reference}"


#
# Devuelve el nombre provisional de una columna sin encabezado.
#
def _placeholder_name(column: int) -> str:
    return f"Columna {get_column_letter(column + 1)}"


#
# Calcula los límites de una región, base 1, a partir de sus columnas y filas.
#
def _region_bounds(columns: list[int], rows: list[int]) -> _Bounds:
    return (columns[0] + 1, rows[0] + 1, columns[-1] + 1, rows[-1] + 1)


#
# Devuelve los valores de una fila de la hoja para las columnas indicadas.
#
def _row_values(sheet: SheetData, row: int, columns: list[int]) -> list[object]:
    return [sheet.frame.iat[row, column] for column in columns]


#
# Indica si alguno de los valores no está vacío.
#
def _has_value(values) -> bool:
    return any(not is_empty(value) for value in values)


#
# Indica si el valor es una fórmula de hoja de cálculo.
#
def _is_formula(value: object) -> bool:
    return isinstance(value, str) and value.startswith("=")


#
# Indica si el valor es un número o una fecha, excluyendo booleanos.
#
def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float, Decimal, date)) and not isinstance(value, bool)


#
# Indica si una celda combinada queda totalmente dentro de los límites.
#
def _merge_inside(merge: _Bounds, bounds: _Bounds) -> bool:
    min_col, min_row, max_col, max_row = merge
    left, top, right, bottom = bounds
    return left <= min_col <= max_col <= right and top <= min_row <= max_row <= bottom


#
# Indica si una celda combinada se superpone con los límites.
#
def _merge_overlaps(merge: _Bounds, bounds: _Bounds) -> bool:
    min_col, min_row, max_col, max_row = merge
    left, top, right, bottom = bounds
    return min_col <= right and max_col >= left and min_row <= bottom and max_row >= top


#
# Indica si el valor es un texto utilizable como etiqueta de encabezado.
#
def _is_header_text(value: object) -> bool:
    if not isinstance(value, str) or not value.strip() or value.startswith("="):
        return False

    return not _looks_like_number_or_date(value)


#
# Indica si un texto tiene la forma de un número o una fecha.
#
def _looks_like_number_or_date(value: str) -> bool:
    return bool(re.search(r"\d", value)) and re.fullmatch(r"[+-]?[\d.,/ :T-]+", value) is not None


#
# Indica si la fila parece un título, con a lo sumo una etiqueta de texto.
#
def _is_title_row(values) -> bool:
    present = [value for value in values if not is_empty(value)]
    return len(present) <= 1 and all(_is_header_text(value) for value in present)


#
# Detecta las regiones tabulares de una hoja, declaradas o inferidas por bloques ocupados.
#
def _regions(sheet: SheetData, file_type: str) -> list[_Bounds]:
    if sheet.has_declared_tables:
        return sheet.table_ranges

    values = sheet.frame.values.tolist()
    occupied_rows = [index for index, row in enumerate(values) if _has_value(row)]

    if not occupied_rows:
        return []

    if file_type == "csv":
        return [(1, occupied_rows[0] + 1, sheet.frame.shape[1], occupied_rows[-1] + 1)]

    regions = []

    for rows in _consecutive_groups(occupied_rows):
        for columns in _column_blocks(sheet, values, rows):
            _add_region(sheet, regions, values, rows, columns)

    return regions


#
# Separa las columnas ocupadas en bloques, uniéndolos si comparten un mismo encabezado.
#
def _column_blocks(sheet: SheetData, values: list[list], rows: list[int]) -> list[list[int]]:
    columns = [
        column
        for column in range(sheet.frame.shape[1])
        if any(not is_empty(values[row][column]) for row in rows)
    ]
    blocks = _consecutive_groups(columns)

    if len(blocks) > 1 and _blocks_share_header(sheet, values, rows, columns, blocks):
        return [columns]

    return blocks


#
# Indica si varios bloques de columnas separados comparten en realidad un único encabezado.
#
def _blocks_share_header(
    sheet: SheetData,
    values: list[list],
    rows: list[int],
    columns: list[int],
    blocks: list[list[int]],
) -> bool:
    independent = all(
        len(block) >= 2 and _header(sheet, _region_bounds(block, rows)) for block in blocks
    )
    formula_block = any(_only_formulas(values, rows[1:], block) for block in blocks)
    first_row = [values[rows[0]][column] for column in columns]
    labels = [value for value in first_row if not is_empty(value)]
    shared_header = (
        len(labels) >= 2
        and len(labels) * 2 >= len(columns)
        and all(_is_header_text(value) for value in labels)
    )

    return not independent and not formula_block and shared_header


#
# Indica si las celdas indicadas están vacías o contienen solo fórmulas.
#
def _only_formulas(values: list[list], rows: list[int], columns: list[int]) -> bool:
    return all(
        is_empty(values[row][column]) or _is_formula(values[row][column])
        for row in rows
        for column in columns
    )


#
# Agrega una región a la lista o la fusiona con la anterior si no tiene encabezado propio.
#
def _add_region(
    sheet: SheetData,
    regions: list[_Bounds],
    values: list[list],
    rows: list[int],
    columns: list[int],
) -> None:
    bounds = _region_bounds(columns, rows)
    previous = _previous_region_with_header(sheet, regions, bounds)
    first_row = values[rows[0]][columns[0] : columns[-1] + 1]
    new_header = _header(sheet, bounds)
    continues_previous = (
        previous is not None
        and new_header is None
        and not all(_is_header_text(value) for value in first_row)
    )

    if not continues_previous:
        regions.append(bounds)
        return

    prior = regions[previous]
    regions[previous] = (prior[0], prior[1], prior[2], bounds[3])


#
# Busca la región previa con las mismas columnas y un encabezado válido.
#
def _previous_region_with_header(
    sheet: SheetData, regions: list[_Bounds], bounds: _Bounds
) -> int | None:
    for index in reversed(range(len(regions))):
        region = regions[index]
        same_columns = region[0] == bounds[0] and region[2] == bounds[2]

        if same_columns and _header(sheet, region) is not None:
            return index

    return None


#
# Detecta el encabezado de una región y devuelve sus nombres y su cantidad de filas.
#
def _header(
    sheet: SheetData,
    bounds: _Bounds,
    columns: list[int] | None = None,
    depth: int | None = None,
) -> tuple[list[str], int] | None:
    left, top, right, bottom = bounds

    if top >= bottom:
        return None

    columns = columns if columns is not None else list(range(left - 1, right))
    first_row = _row_values(sheet, top - 1, columns)
    merges = _header_merges(sheet, bounds)
    requested_depth = depth or _merged_depth(merges, top)

    if 1 < requested_depth <= 4 and top + requested_depth <= bottom:
        stacked = _stacked_header(sheet, top, columns, merges, requested_depth)

        if stacked is not None:
            return stacked

    if top + 1 < bottom:
        with_units = _unit_row_header(sheet, top, columns, first_row, depth)

        if with_units is not None:
            return with_units

    if depth not in (None, 1):
        return None

    return _single_row_header(sheet, bounds, columns, first_row)


#
# Devuelve las celdas combinadas que forman parte del encabezado de una región.
#
def _header_merges(sheet: SheetData, bounds: _Bounds) -> list[_Bounds]:
    left, top, right, bottom = bounds
    return [
        merged
        for merged in sheet.merged_ranges
        if top <= merged[1] <= top + 2
        and merged[3] < min(bottom, top + 4)
        and merged[0] >= left
        and merged[2] <= right
    ]


#
# Calcula cuántas filas de encabezado implican las celdas combinadas.
#
def _merged_depth(merges: list[_Bounds], top: int) -> int:
    return max((merged[3] - top + 1 + (merged[2] > merged[0]) for merged in merges), default=1)


#
# Construye el encabezado de varias filas apiladas y devuelve sus nombres si es consistente.
#
def _stacked_header(
    sheet: SheetData, top: int, columns: list[int], merges: list[_Bounds], depth: int
) -> tuple[list[str], int] | None:
    grid = _header_grid(sheet, top, columns, merges, depth)
    values = [value for row in grid for value in row if not is_empty(value)]
    sparse_context = any(
        sum(not is_empty(value) for value in row) < len(columns) for row in grid[:-1]
    )
    textual = all(_is_header_text(value) for value in values)
    labelled = _labelled_groups(grid, merges)

    if not values or not (textual or labelled) or not (merges or sparse_context):
        return None

    return _stacked_names(grid, columns), depth


#
# Arma la cuadrícula de valores del encabezado, propagando el contenido de las celdas combinadas.
#
def _header_grid(
    sheet: SheetData, top: int, columns: list[int], merges: list[_Bounds], depth: int
) -> list[list[object]]:
    grid = [_row_values(sheet, row, columns) for row in range(top - 1, top + depth - 1)]

    for min_col, min_row, max_col, max_row in merges:
        if max_row >= top + depth:
            continue

        for row in range(min_row, max_row + 1):
            for index, column in enumerate(columns):
                if min_col <= column + 1 <= max_col:
                    grid[row - top][index] = sheet.frame.iat[min_row - 1, min_col - 1]

    return grid


#
# Indica si las filas superiores son etiquetas de grupo combinadas sobre una fila de etiquetas.
#
def _labelled_groups(grid: list[list[object]], merges: list[_Bounds]) -> bool:
    last_row = grid[-1]

    return (
        bool(merges)
        and any(_is_header_text(value) for value in last_row)
        and all(is_empty(value) or _is_header_text(value) for value in last_row)
        and all(_is_group_label(value) for row in grid[:-1] for value in row if not is_empty(value))
    )


#
# Indica si el valor sirve como etiqueta de grupo en un encabezado apilado.
#
def _is_group_label(value: object) -> bool:
    return (
        _is_header_text(value)
        or _is_scalar(value)
        or (isinstance(value, str) and _PLAIN_NUMBER.fullmatch(value.strip()) is not None)
    )


#
# Une las etiquetas apiladas de cada columna en un único nombre.
#
def _stacked_names(grid: list[list[object]], columns: list[int]) -> list[str]:
    names = []

    for index, column in enumerate(columns):
        parts = dict.fromkeys(str(row[index]) for row in grid if not is_empty(row[index]))
        names.append(STACKED_HEADER_SEPARATOR.join(parts) or _placeholder_name(column))

    return names


#
# Detecta un encabezado de dos filas donde la segunda documenta las unidades.
#
def _unit_row_header(
    sheet: SheetData,
    top: int,
    columns: list[int],
    first_row: list[object],
    depth: int | None,
) -> tuple[list[str], int] | None:
    second_row = _row_values(sheet, top, columns)
    labels = [value for value in second_row if not is_empty(value)]
    documented_units = sum(
        isinstance(value, str) and _label_unit(value) is not None for value in labels
    )
    complete = all(
        _is_header_text(parent) or _is_header_text(child)
        for parent, child in zip(first_row, second_row)
    )
    textual = all(is_empty(value) or _is_header_text(value) for value in first_row + second_row)
    numeric_body = any(_is_numeric_cell(value) for value in _row_values(sheet, top + 1, columns))
    units_documented = len(labels) >= 2 and documented_units * 2 >= len(labels)

    if depth == 1 or not (complete and textual and numeric_body and units_documented):
        return None

    names = [_unit_row_name(parent, child) for parent, child in zip(first_row, second_row)]

    return names, 2


#
# Devuelve la unidad que documenta una etiqueta, si la tiene.
#
def _label_unit(value: object) -> str | None:
    return header_unit(f"unit [{value}]")


#
# Indica si la celda contiene un número o un texto numérico simple.
#
def _is_numeric_cell(value: object) -> bool:
    return _is_scalar(value) or (isinstance(value, str) and bool(_PLAIN_NUMBER.fullmatch(value)))


#
# Compone el nombre de una columna a partir de su etiqueta superior y su etiqueta de unidad.
#
def _unit_row_name(parent: object, child: object) -> str:
    if is_empty(parent):
        return str(child)

    if is_empty(child):
        return str(parent)

    if _label_unit(child):
        return f"{parent} [{child}]"

    return f"{parent}{STACKED_HEADER_SEPARATOR}{child}"


#
# Detecta un encabezado de una sola fila y devuelve sus nombres si es confiable.
#
def _single_row_header(
    sheet: SheetData, bounds: _Bounds, columns: list[int], first_row: list[object]
) -> tuple[list[str], int] | None:
    left, top, right, bottom = bounds
    labels = [value for value in first_row if not is_empty(value)]
    partial = len(labels) != len(columns)
    next_row = _row_values(sheet, top, columns)
    labelled_next_row = sum(_is_header_text(value) for value in next_row) * 2 > len(columns)

    if _mostly_codes(labels) and labelled_next_row:
        return None

    evidence = _has_data_evidence(sheet, top, bottom, columns)
    enough_labels = len(labels) >= 2 and len(labels) * 2 >= len(columns)

    if not labels or not all(_is_header_text(value) for value in labels):
        return None

    if partial and not (enough_labels and evidence):
        return None

    names = [
        _placeholder_name(column) if is_empty(value) else str(value)
        for value, column in zip(first_row, columns)
    ]

    return names, 1


#
# Indica si la mayoría de las etiquetas empieza con un dígito, como códigos.
#
def _mostly_codes(labels: list[object]) -> bool:
    codes = sum(isinstance(value, str) and re.match(r"^\d", value) is not None for value in labels)
    return codes * 2 > len(labels)


#
# Indica si las primeras filas bajo el encabezado contienen valores propios de datos.
#
def _has_data_evidence(sheet: SheetData, top: int, bottom: int, columns: list[int]) -> bool:
    return any(
        _is_data_like(value)
        for row in range(top, min(bottom, top + 3))
        for value in sheet.frame.iloc[row, columns]
    )


#
# Indica si el valor parece un dato: número, fecha, fórmula o texto que empieza con número.
#
def _is_data_like(value: object) -> bool:
    return _is_scalar(value) or (
        isinstance(value, str) and (value.startswith("=") or bool(re.match(r"^[+-]?\d", value)))
    )


#
# Devuelve las columnas de la región que tienen al menos una celda con valor.
#
def _occupied_columns(sheet: SheetData, bounds: _Bounds) -> list[int]:
    left, top, right, bottom = bounds
    return [
        column
        for column in range(left - 1, right)
        if any(not is_empty(sheet.frame.iat[row, column]) for row in range(top - 1, bottom))
    ]


#
# Localiza el encabezado de una región, saltando títulos previos, y cuenta las filas omitidas.
#
def _locate_header(
    sheet: SheetData,
    bounds: _Bounds,
    selected: list[int],
    declared: bool,
    header_depth: int | None,
) -> tuple[tuple[list[str], int] | None, int]:
    header = _header(sheet, bounds, selected, depth=header_depth)
    skipped = 0

    if header is not None or declared or header_depth is not None:
        return header, skipped

    left, top, right, bottom = bounds

    while top + skipped < min(bottom, top + 5):
        if not _is_title_row(sheet.frame.iloc[top + skipped - 1, left - 1 : right]):
            break

        skipped += 1
        header = _header(sheet, (left, top + skipped, right, bottom), selected)

        if header is not None:
            break

    return header, skipped


#
# Indica si hay celdas combinadas dentro de la zona de datos de la región.
#
def _merges_in_data(sheet: SheetData, bounds: _Bounds, data_start: int) -> bool:
    left, top, right, bottom = bounds
    return any(
        min_row < bottom + 1 and max_row >= data_start and min_col <= right and max_col >= left
        for min_col, min_row, max_col, max_row in sheet.merged_ranges
    )


#
# Recorre las filas de datos separando registros de totales, leyendas y filas solo de fórmulas.
#
def _scan_rows(
    sheet: SheetData, selected: list[int], data_start: int, bottom: int, wide_region: bool
) -> _RowScan:
    scan = _RowScan()

    for row_index in range(data_start - 1, bottom):
        values = [sheet.frame.iat[row_index, column] for column in selected]

        if all(is_empty(value) for value in values):
            continue

        occupied = [value for value in values if not is_empty(value)]

        if len(selected) > 1 and all(_is_formula(value) for value in occupied):
            scan.formula_only_rows += 1
            continue

        if _is_total_row(occupied):
            scan.omitted_totals += 1
            continue

        if wide_region and _is_note_row(occupied):
            scan.omitted_notes += 1
            continue

        scan.rows.append(values)
        scan.source_rows.append(sheet.source_rows[row_index])

    return scan


#
# Indica si la fila es un total, con etiqueta de total e importes en el resto.
#
def _is_total_row(occupied: list[object]) -> bool:
    if len(occupied) <= 1:
        return False

    label = occupied[0]

    if not isinstance(label, str) or label.strip().lower() not in TOTAL_ROW_LABELS:
        return False

    return all(_is_total_amount(value) for value in occupied[1:])


#
# Indica si el valor puede ser un importe de total: número, fórmula o texto numérico.
#
def _is_total_amount(value: object) -> bool:
    if isinstance(value, (int, float, Decimal)):
        return True

    return isinstance(value, str) and (
        value.startswith("=") or _PLAIN_NUMBER.fullmatch(value) is not None
    )


#
# Indica si la fila es una leyenda de una sola celda con etiqueta de nota explícita.
#
def _is_note_row(occupied: list[object]) -> bool:
    if len(occupied) != 1:
        return False

    note = occupied[0]

    return isinstance(note, str) and NOTE_ROW.match(note.strip()) is not None


#
# Genera las advertencias sobre filas de título, leyendas, totales y fórmulas que quedan sin interpretar.
#
def _omission_warnings(skipped: int, scan: _RowScan) -> list[str]:
    warnings = []

    if skipped:
        warnings.append(
            f"{skipped} filas de título anteriores al encabezado permanecen en el original."
        )

    if scan.omitted_notes:
        warnings.append(f"{scan.omitted_notes} leyendas de una celda permanecen en el original.")

    if scan.omitted_totals:
        warnings.append(
            f"{scan.omitted_totals} filas etiquetadas como totales permanecen en el original."
        )

    if scan.formula_only_rows:
        warnings.append(
            f"{scan.formula_only_rows} filas compuestas sólo por fórmulas permanecen en el original porque no aportan registros interpretados."
        )

    return warnings


#
# Construye los metadatos de una columna: nombre, tipo, origen, unidad y encabezados originales.
#
def _column_metadata(
    sheet: SheetData,
    original: str,
    sql_name: str,
    sql_type: str,
    column: int,
    header_rows: range,
) -> dict:
    metadata = {
        "original_name": original,
        "sql_name": sql_name,
        "sql_type": sql_type,
        "source_column": column + 1,
        "header_values": [text_value(sheet.frame.iat[row, column]) for row in header_rows],
        "numeric_eligible": sql_type in {"BIGINT", "NUMERIC"},
        "unit": None,
        "meaning": None,
    }
    unit = header_unit(original)

    if unit:
        metadata["unit"] = unit
        metadata["unit_evidence"] = original

    if all(is_empty(value) for value in metadata["header_values"]):
        metadata["header_missing"] = True

    return metadata


#
# Genera un nombre SQL libre para la columna de proyección numérica de una columna de texto.
#
def projection_name(sql_name: str, columns: list[dict]) -> str:
    base = sql_name[:53] + "_numeric"
    taken = {column["sql_name"] for column in columns}
    name, suffix = base, 2

    while name in taken:
        name = base[:59] + f"_{suffix}"
        suffix += 1

    return name


#
# Construye los metadatos de la columna de proyección numérica derivada de una de texto.
#
def projection_column(metadata: dict, name: str, values: list, excluded: int) -> dict:
    return {
        **metadata,
        "original_name": metadata["original_name"] + " (representación numérica)",
        "sql_name": name,
        "sql_type": "NUMERIC",
        "numeric_eligible": True,
        "derived_from": metadata["sql_name"],
        "representation": "numeric_projection",
        "excluded_count": excluded,
        "nonempty_count": sum(not is_empty(value) for value in values) + excluded,
    }


#
# Arma los registros de la tabla combinando valores convertidos, número de fila y fila de origen.
#
def _records(columns: list[dict], converted: list[list], source_rows: list[int]) -> list[dict]:
    sql_names = [column["sql_name"] for column in columns]
    records = []

    for index, (values, source_row) in enumerate(zip(zip(*converted), source_rows), 1):
        records.append(dict(zip(sql_names, values), _aquai_row=index, _aquai_source_row=source_row))

    return records


#
# Calcula una huella por fila a partir de las primeras columnas comparadas.
#
def _row_fingerprints(table: PreparedTable, compared_columns: int) -> list[str]:
    names = [column["sql_name"] for column in table.columns[:compared_columns]]
    fingerprints = []

    for row in table.rows:
        payload = json.dumps([row[name] for name in names], default=str, ensure_ascii=False)
        fingerprints.append(sha256(payload.encode()).hexdigest())

    return fingerprints


#
# Cuenta las filas coincidentes entre dos tablas con el mismo esquema inicial.
#
def _row_overlap(
    table: PreparedTable, schema: tuple, other: PreparedTable, other_schema: tuple
) -> dict | None:
    common = min(len(schema), len(other_schema))

    if common < 2 or schema[:common] != other_schema[:common]:
        return None

    known = set(_row_fingerprints(other, common))
    count = sum(item in known for item in _row_fingerprints(table, common))

    if not count:
        return None

    return {"with": other.name, "matching_rows": count, "compared_columns": common}


#
# Cuenta las celdas con valor que quedan fuera de las tablas declaradas de la hoja.
#
def _cells_outside_tables(sheet: SheetData) -> int:
    outside = 0

    for row_index, row in enumerate(sheet.frame.values.tolist(), 1):
        for column_index, value in enumerate(row, 1):
            inside = any(
                left <= column_index <= right and top <= row_index <= bottom
                for left, top, right, bottom in sheet.table_ranges
            )

            if not inside and not is_empty(value):
                outside += 1

    return outside


#
# Construye el identificador de una región a partir de su hoja y sus límites.
#
def _region_id(sheet_index: int, bounds: _Bounds) -> str:
    left, top, right, bottom = bounds
    return f"{sheet_index}:{left}:{top}:{right}:{bottom}"


#
# Indica si una sugerencia de estructura tiene orientación y posiciones coherentes.
#
def _is_valid_hint(hint: StructureSuggestion) -> bool:
    positions = (hint.header_start, hint.header_depth, hint.data_start)
    integers = all(type(value) is int for value in positions)

    if hint.orientation not in {"rows", "columns"} or not integers:
        return False

    return (
        1 <= hint.header_depth <= 4
        and hint.header_start >= 1
        and hint.data_start == hint.header_start + hint.header_depth
    )


#
# Devuelve una hoja con el fragmento de la región transpuesto, para leer matrices por columnas.
#
def _transposed_sheet(sheet: SheetData, bounds: _Bounds) -> SheetData:
    left, top, right, bottom = bounds
    fragment = sheet.frame.iloc[top - 1 : bottom, left - 1 : right].T
    merges = [
        (min_row - top + 1, min_col - left + 1, max_row - top + 1, max_col - left + 1)
        for min_col, min_row, max_col, max_row in sheet.merged_ranges
        if _merge_inside((min_col, min_row, max_col, max_row), bounds)
    ]

    return SheetData(
        sheet.name,
        fragment,
        list(range(left, right + 1)),
        merges,
        warnings=list(sheet.warnings),
    )


#
# Devuelve las celdas combinadas dentro de la región con posiciones relativas a ella.
#
def _relative_merges(sheet: SheetData, bounds: _Bounds) -> list[list[int]]:
    left, top, right, bottom = bounds
    return [
        [min_col - left + 1, min_row - top + 1, max_col - left + 1, max_row - top + 1]
        for min_col, min_row, max_col, max_row in sheet.merged_ranges
        if _merge_inside((min_col, min_row, max_col, max_row), bounds)
    ]


#
# Toma una fila de muestra de la región con celdas de texto acotado.
#
def _sample_row(sheet: SheetData, row: int, columns: list[int], bounds: _Bounds) -> dict:
    left, top, right, bottom = bounds
    cells = {}

    for column in columns:
        value = sheet.frame.iat[row, column]
        cells[str(column - left + 2)] = None if is_empty(value) else str(value)[:100]

    return {"row": row - top + 2, "cells": cells}


class SimpleTabularProcessor:
    #
    # Guarda la configuración y crea el lector de tablas que usa el procesador.
    #
    def __init__(self, settings: TabularSettings) -> None:
        self.settings = settings
        self.reader = TableReader(settings)

    #
    # Valida el archivo subido y devuelve su nombre seguro.
    #
    def validate_upload(self, filename: str, content: bytes) -> str:
        return self.reader.validate(filename, content)

    #
    # Lee el archivo y lo convierte en tablas preparadas, usando sugerencias de estructura si existen.
    #
    def prepare(
        self, filename: str, content: bytes, suggestions: list[StructureSuggestion] | None = None
    ) -> PreparationResult:
        loaded = self.reader.read(filename, content)
        result = PreparationResult(warnings=list(loaded.warnings))
        hints = self._unique_hints(suggestions or [], result.warnings)

        for sheet_index, sheet in enumerate(loaded.sheets):
            self._prepare_sheet(filename, loaded, sheet_index, sheet, hints, result)

        if not result.tables:
            result.warnings.append(
                "No se encontraron tablas con encabezados y registros confiables."
            )

        self._mark_overlap(result.tables)

        return result

    #
    # Indexa las sugerencias por región y las descarta si hay regiones repetidas.
    #
    @staticmethod
    def _unique_hints(
        suggestions: list[StructureSuggestion], warnings: list[str]
    ) -> dict[str, StructureSuggestion]:
        hints = {hint.region_id: hint for hint in suggestions}

        if len(hints) == len(suggestions):
            return hints

        warnings.append("La asistencia repitió regiones y no se aplicó.")

        return {}

    #
    # Prepara las tablas de una hoja y advierte sobre celdas fuera de las tablas declaradas.
    #
    def _prepare_sheet(
        self,
        filename: str,
        loaded: LoadedFile,
        sheet_index: int,
        sheet: SheetData,
        hints: dict[str, StructureSuggestion],
        result: PreparationResult,
    ) -> None:
        result.warnings.extend(f"{sheet.name}: {warning}" for warning in sheet.warnings)
        regions = _regions(sheet, loaded.file_type)

        if not regions and not sheet.has_declared_tables:
            result.warnings.append(f"La hoja '{sheet.name}' está vacía.")

        for bounds in regions:
            self._prepare_detected_region(
                filename, loaded, sheet_index, sheet, bounds, hints, result
            )

        if not sheet.has_declared_tables:
            return

        outside = _cells_outside_tables(sheet)

        if outside:
            result.warnings.append(
                f"{sheet.name}: {outside} celdas fuera de las tablas declaradas permanecen en el original."
            )

    #
    # Prepara una región detectada, recurriendo a una sugerencia verificada o recogiendo un candidato.
    #
    def _prepare_detected_region(
        self,
        filename: str,
        loaded: LoadedFile,
        sheet_index: int,
        sheet: SheetData,
        bounds: _Bounds,
        hints: dict[str, StructureSuggestion],
        result: PreparationResult,
    ) -> None:
        warning_start = len(result.warnings)
        table = self._prepare_region(filename, loaded, sheet, bounds, result.warnings)
        region_id = _region_id(sheet_index, bounds)
        accepts_candidates = (
            self.settings.structure_assistance and len(result.structure_candidates) < 3
        )

        if table is None and region_id in hints:
            table = self._verified_hint(
                filename, loaded, sheet, bounds, hints[region_id], result.warnings, warning_start
            )
        elif table is None and accepts_candidates:
            self._collect_candidate(sheet, bounds, region_id, result)

        if table is None:
            return

        result.tables.append(table)

        if len(result.tables) > self.settings.max_datasets:
            raise FileTooLargeError("El archivo supera el límite de datasets permitido.")

    #
    # Aplica una sugerencia de estructura y descarta las advertencias del intento automático si se verifica.
    #
    def _verified_hint(
        self,
        filename: str,
        loaded: LoadedFile,
        sheet: SheetData,
        bounds: _Bounds,
        hint: StructureSuggestion,
        warnings: list[str],
        warning_start: int,
    ) -> PreparedTable | None:
        table = self._assisted_region(filename, loaded, sheet, bounds, hint)

        if table is None:
            warnings.append(
                f"La sugerencia para {sheet.name} no pudo verificarse; se conserva el original."
            )
            return None

        del warnings[warning_start:]

        return table

    #
    # Registra una región no resuelta como candidata para recibir asistencia de estructura.
    #
    def _collect_candidate(
        self, sheet: SheetData, bounds: _Bounds, region_id: str, result: PreparationResult
    ) -> None:
        left, top, right, bottom = bounds
        evidence = any(
            _is_header_text(value)
            for row in sheet.frame.iloc[top:bottom, left - 1 : right].values
            for value in row
        )

        if bottom - top >= 2 and right > left and evidence:
            result.structure_candidates.append(self._structure_sample(sheet, bounds, region_id))

    #
    # Arma una muestra acotada de la región para pedir asistencia de estructura.
    #
    @staticmethod
    def _structure_sample(sheet: SheetData, bounds: _Bounds, region_id: str) -> dict:
        left, top, right, bottom = bounds
        row_indices = sorted(
            set(range(top - 1, min(bottom, top + 7))) | {bottom - 1, (top + bottom) // 2 - 1}
        )
        column_indices = sorted(set(range(left - 1, min(right, left + 9))) | {right - 1})

        return {
            "region_id": region_id,
            "sheet": sheet.name,
            "range": _range_reference(bounds),
            "shape": [bottom - top + 1, right - left + 1],
            "sample_is_partial": True,
            "sample": [_sample_row(sheet, row, column_indices, bounds) for row in row_indices],
            "merged_ranges": _relative_merges(sheet, bounds)[:20],
        }

    #
    # Prepara una región según una sugerencia de estructura, verificándola contra el archivo.
    #
    def _assisted_region(
        self,
        filename: str,
        loaded: LoadedFile,
        sheet: SheetData,
        bounds: _Bounds,
        hint: StructureSuggestion,
    ) -> PreparedTable | None:
        left, top, right, bottom = bounds

        if not _is_valid_hint(hint):
            return None

        if any(
            _merge_overlaps(merge, bounds) and not _merge_inside(merge, bounds)
            for merge in sheet.merged_ranges
        ):
            return None

        by_columns = hint.orientation == "columns"

        if by_columns:
            working = _transposed_sheet(sheet, bounds)
            region = (1, hint.header_start, working.frame.shape[1], working.frame.shape[0])
        else:
            working, region = sheet, (left, top + hint.header_start - 1, right, bottom)

        region_left, header_top, region_right, region_bottom = region

        if (
            header_top + hint.header_depth > region_bottom
            or region_right - region_left + 1 > self.settings.max_columns
        ):
            return None

        prefix_top = 0 if by_columns else top - 1
        prefix = working.frame.iloc[prefix_top : header_top - 1, region_left - 1 : region_right]

        if not all(_is_title_row(row) for row in prefix.values.tolist()):
            return None

        header = _header(working, region, depth=hint.header_depth) or _header(working, region)

        if header is None:
            return None

        verified_depth = header[1]
        table = self._prepare_region(
            filename, loaded, working, region, [], header_depth=verified_depth
        )

        if table is None:
            return None

        table.source_locator["structure_assistance"] = {
            "verified": True,
            "orientation": hint.orientation,
            "header_start": hint.header_start,
            "header_depth": verified_depth,
            "suggested_header_depth": hint.header_depth,
        }

        if hint.header_start > 1:
            table.warnings.append(
                f"{hint.header_start - 1} posiciones anteriores al encabezado asistido quedan en el original como contexto no interpretado."
            )

        if by_columns:
            self._restore_column_orientation(table, filename, sheet, bounds, hint, verified_depth)

        table.warnings.append(
            "La orientación y los encabezados sugeridos por Qwen se verificaron contra el archivo completo."
        )

        return table

    #
    # Restaura la referencia a filas y columnas de origen en tablas leídas por columnas.
    #
    @staticmethod
    def _restore_column_orientation(
        table: PreparedTable,
        filename: str,
        sheet: SheetData,
        bounds: _Bounds,
        hint: StructureSuggestion,
        verified_depth: int,
    ) -> None:
        left, top, right, bottom = bounds

        for column in table.columns:
            column["source_row"] = top + column.pop("source_column") - 1

        for row in table.rows:
            row["_aquai_source_column"] = row["_aquai_source_row"]
            row["_aquai_source_row"] = top

        first_header_column = left + hint.header_start - 1
        table.source_locator.update(
            range=_range_reference(bounds),
            orientation="columns",
            header_columns=list(range(first_header_column, first_header_column + verified_depth)),
        )
        table.source_locator.pop("header_rows", None)
        table.name = _table_name(filename, sheet.name, table.source_locator["range"])
        table.description = f"Matriz de {sheet.name}, rango {table.source_locator['range']}, con {len(table.rows)} columnas de origen almacenadas como registros."

    #
    # Marca las tablas que comparten filas con otras tablas del mismo archivo.
    #
    @staticmethod
    def _mark_overlap(tables: list[PreparedTable]) -> None:
        previous: list[tuple[PreparedTable, tuple]] = []

        for table in tables:
            schema = tuple((column["original_name"], column["unit"]) for column in table.columns)
            overlaps = []

            for other, other_schema in previous:
                overlap = _row_overlap(table, schema, other, other_schema)

                if overlap is not None:
                    overlaps.append(overlap)

            if overlaps:
                table.source_locator["possible_overlap"] = overlaps
                table.warnings.append(
                    "Hay filas coincidentes en otros datasets del archivo; revisar possible_overlap antes de combinar observaciones."
                )

            previous.append((table, schema))

    #
    # Prepara una región como tabla tipada o devuelve None con advertencias si no es confiable.
    #
    def _prepare_region(
        self,
        filename: str,
        loaded: LoadedFile,
        sheet: SheetData,
        bounds: _Bounds,
        warnings: list[str],
        header_depth: int | None = None,
    ) -> PreparedTable | None:
        left, top, right, bottom = bounds
        label = f"{sheet.name}!{_range_reference(bounds)}"
        declared = bounds in sheet.table_ranges
        selected = _occupied_columns(sheet, bounds)

        if len(selected) > self.settings.max_columns:
            warnings.append(
                f"La región {label} supera el límite de columnas por dataset y permanece en el original."
            )
            return None

        header, skipped = _locate_header(sheet, bounds, selected, declared, header_depth)
        single_column_sheet = right == left and loaded.file_type == "xlsx" and not declared

        if header is None or single_column_sheet:
            warnings.append(
                f"La región {label} no tiene estructura tabular confiable y permanece en el original."
            )
            return None

        originals, depth = header
        header_rows = range(top + skipped - 1, top + skipped + depth - 1)
        data_start = top + skipped + depth

        if _merges_in_data(sheet, bounds, data_start):
            warnings.append(
                f"La región {label} contiene celdas combinadas en los datos y permanece en el original."
            )
            return None

        scan = _scan_rows(sheet, selected, data_start, bottom, wide_region=right > left)

        if not scan.rows:
            warnings.append(f"La región {label} contiene encabezados sin registros.")
            return None

        table_warnings = list(sheet.warnings) + _omission_warnings(skipped, scan)
        columns, converted = self._typed_columns(
            sheet, selected, originals, scan.rows, header_rows, table_warnings
        )
        self._add_numeric_projections(columns, converted, scan.rows, table_warnings)
        records = _records(columns, converted, scan.source_rows)

        return self._prepared_table(
            filename,
            loaded,
            sheet,
            bounds,
            originals,
            columns,
            records,
            header_rows,
            table_warnings,
        )

    #
    # Infiere el tipo y los metadatos de cada columna y registra advertencias de encabezado o tipo.
    #
    @staticmethod
    def _typed_columns(
        sheet: SheetData,
        selected: list[int],
        originals: list[str],
        rows: list[list[object]],
        header_rows: range,
        table_warnings: list[str],
    ) -> tuple[list[dict], list[list]]:
        columns, converted = [], []

        for index, (original, sql_name) in enumerate(zip(originals, column_names(originals))):
            sql_type, values, warning = infer_column(original, [row[index] for row in rows])
            metadata = _column_metadata(
                sheet, original, sql_name, sql_type, selected[index], header_rows
            )
            columns.append(metadata)
            converted.append(values)

            if metadata.get("header_missing"):
                table_warnings.append(
                    f"La columna {get_column_letter(selected[index] + 1)} no tiene encabezado; su nombre es provisional."
                )

            if warning:
                table_warnings.append(warning)

        return columns, converted

    #
    # Agrega columnas de proyección numérica para columnas de texto con mediciones.
    #
    def _add_numeric_projections(
        self,
        columns: list[dict],
        converted: list[list],
        rows: list[list[object]],
        table_warnings: list[str],
    ) -> None:
        for index, metadata in enumerate(list(columns)):
            projectable = (
                metadata["sql_type"] == "TEXT"
                and not metadata.get("header_missing")
                and len(columns) < self.settings.max_columns
            )

            if not projectable:
                continue

            projection = numeric_projection(metadata["original_name"], [row[index] for row in rows])

            if projection is None:
                continue

            values, excluded = projection
            name = projection_name(metadata["sql_name"], columns)
            columns.append(projection_column(metadata, name, values, excluded))
            converted.append(values)
            table_warnings.append(
                f"{metadata['original_name']}: representación numérica auxiliar; {excluded} valores no convertibles conservados en el texto original y excluidos de cálculos."
            )

    #
    # Construye la tabla preparada con su localizador de origen, descripción y advertencias.
    #
    @staticmethod
    def _prepared_table(
        filename: str,
        loaded: LoadedFile,
        sheet: SheetData,
        bounds: _Bounds,
        originals: list[str],
        columns: list[dict],
        records: list[dict],
        header_rows: range,
        table_warnings: list[str],
    ) -> PreparedTable:
        locator = {
            "file_type": loaded.file_type,
            "sheet": sheet.name,
            "range": _range_reference(bounds),
            "header_rows": [sheet.source_rows[index] for index in header_rows],
        }

        if loaded.file_type == "csv":
            locator.update(encoding=loaded.encoding, delimiter=loaded.delimiter)

        visible_columns = ", ".join(originals[:8])[:400]
        description = f"Tabla de {sheet.name}, rango {locator['range']}, con {len(records)} registros. Columnas: {visible_columns}."

        return PreparedTable(
            name=_table_name(filename, sheet.name, locator["range"]),
            source_locator=locator,
            columns=columns,
            rows=records,
            description=description,
            warnings=table_warnings,
        )
