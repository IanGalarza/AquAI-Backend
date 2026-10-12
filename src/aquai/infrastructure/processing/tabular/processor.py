"""Preparación determinística de tablas convencionales, sin planes del modelo."""

from pathlib import PurePath
import re
from decimal import Decimal
from datetime import date

from openpyxl.utils.cell import get_column_letter

from aquai.application.contracts.ingestion import PreparationResult, PreparedTable
from aquai.application.errors import FileTooLargeError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.normalization import column_names, header_unit, infer_column
from aquai.infrastructure.processing.tabular.reader import LoadedFile, SheetData, TableReader, is_empty


def _groups(indices: list[int]) -> list[list[int]]:
    groups = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(index)
    return groups


def _regions(sheet: SheetData, file_type: str) -> list[tuple[int, int, int, int]]:
    if sheet.has_declared_tables:
        return sheet.table_ranges
    values = sheet.frame.values.tolist()
    occupied_rows = [index for index, row in enumerate(values) if any(not is_empty(value) for value in row)]
    if not occupied_rows:
        return []
    if file_type == "csv":
        return [(1, occupied_rows[0] + 1, sheet.frame.shape[1], occupied_rows[-1] + 1)]
    regions = []
    for rows in _groups(occupied_rows):
        columns = [column for column in range(sheet.frame.shape[1])
                   if any(not is_empty(values[row][column]) for row in rows)]
        for cols in _groups(columns):
            bounds = (cols[0] + 1, rows[0] + 1, cols[-1] + 1, rows[-1] + 1)
            previous = next((index for index, region in reversed(list(enumerate(regions)))
                             if region[0] == bounds[0] and region[2] == bounds[2]
                             and _header(sheet, region) is not None), None)
            first_row = values[rows[0]][cols[0]:cols[-1] + 1]
            if previous is not None and not all(_header_text(value) for value in first_row):
                prior = regions[previous]
                regions[previous] = (prior[0], prior[1], prior[2], bounds[3])
            else:
                regions.append(bounds)
    return regions


def _header_text(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and not value.startswith("=")
            and not (re.search(r"\d", value) and re.fullmatch(r"[+-]?[\d.,/ :T-]+", value)))


def _header(sheet: SheetData, bounds: tuple[int, int, int, int], columns: list[int] | None = None) -> tuple[list[str], int] | None:
    left, top, right, bottom = bounds
    if top >= bottom:
        return None
    columns = columns if columns is not None else list(range(left - 1, right))
    original_row = [sheet.frame.iat[top - 1, column] for column in columns]
    row = list(original_row)
    relevant_merges = [merged for merged in sheet.merged_ranges
                       if merged[1] == top and merged[3] <= top + 1
                       and merged[0] >= left and merged[2] <= right]
    if relevant_merges and top + 1 < bottom:
        second = [sheet.frame.iat[top, column] for column in columns]
        for min_col, min_row, max_col, max_row in relevant_merges:
            parent = sheet.frame.iat[min_row - 1, min_col - 1]
            for column in range(min_col, max_col + 1):
                if column - 1 not in columns:
                    continue
                index = columns.index(column - 1)
                row[index] = parent
                if max_row == top + 1:
                    second[index] = parent
        if all(_header_text(value) for value in row + second):
            names = [str(parent) if parent == child else f"{parent} | {child}"
                     for parent, child in zip(row, second)]
            return names, 2
    if top + 1 < bottom:
        second = [sheet.frame.iat[top, column] for column in columns]
        labels = [value for value in second if not is_empty(value)]
        documented_units = sum(isinstance(value, str) and header_unit(f"unit [{value}]") is not None for value in labels)
        complete = all(_header_text(parent) or _header_text(child) for parent, child in zip(original_row, second))
        textual = all(is_empty(value) or _header_text(value) for value in original_row + second)
        body = [sheet.frame.iat[top + 1, column] for column in columns]
        numeric_body = any(isinstance(value, (int, float, Decimal, date)) and not isinstance(value, bool)
                           or isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value) for value in body)
        if complete and textual and numeric_body and len(labels) >= 2 and documented_units * 2 >= len(labels):
            names = []
            for parent, child in zip(original_row, second):
                if is_empty(parent):
                    names.append(str(child))
                elif is_empty(child):
                    names.append(str(parent))
                elif header_unit(f"unit [{child}]"):
                    names.append(f"{parent} [{child}]")
                else:
                    names.append(f"{parent} | {child}")
            return names, 2
    if all(_header_text(value) for value in original_row):
        return [str(value) for value in original_row], 1
    return None


class SimpleTabularProcessor:
    def __init__(self, settings: TabularSettings) -> None:
        self.settings = settings
        self.reader = TableReader(settings)

    def validate_upload(self, filename: str, content: bytes) -> str:
        return self.reader.validate(filename, content)

    def prepare(self, filename: str, content: bytes) -> PreparationResult:
        loaded = self.reader.read(filename, content)
        result = PreparationResult(warnings=list(loaded.warnings))
        for sheet in loaded.sheets:
            result.warnings.extend(f"{sheet.name}: {warning}" for warning in sheet.warnings)
            regions = _regions(sheet, loaded.file_type)
            if not regions and not sheet.has_declared_tables:
                result.warnings.append(f"La hoja '{sheet.name}' está vacía.")
            for bounds in regions:
                table = self._prepare_region(filename, loaded, sheet, bounds, result.warnings)
                if table is not None:
                    result.tables.append(table)
                    if len(result.tables) > self.settings.max_datasets:
                        raise FileTooLargeError("El archivo supera el límite de datasets permitido.")
            if sheet.has_declared_tables:
                outside = sum(not is_empty(value) for row_index, row in enumerate(sheet.frame.values.tolist(), 1)
                              for column_index, value in enumerate(row, 1)
                              if not any(left <= column_index <= right and top <= row_index <= bottom
                                         for left, top, right, bottom in sheet.table_ranges))
                if outside:
                    result.warnings.append(f"{sheet.name}: {outside} celdas fuera de las tablas declaradas permanecen en el original.")
        if not result.tables:
            result.warnings.append("No se encontraron tablas con encabezados y registros confiables.")
        return result

    def _prepare_region(self, filename: str, loaded: LoadedFile, sheet: SheetData,
                        bounds: tuple[int, int, int, int], warnings: list[str]) -> PreparedTable | None:
        left, top, right, bottom = bounds
        range_ref = f"{get_column_letter(left)}{top}:{get_column_letter(right)}{bottom}"
        label = f"{sheet.name}!{range_ref}"
        declared = bounds in sheet.table_ranges
        selected = [column for column in range(left - 1, right)
                    if any(not is_empty(sheet.frame.iat[row, column]) for row in range(top - 1, bottom))]
        if len(selected) > self.settings.max_columns:
            warnings.append(f"La región {label} supera el límite de columnas por dataset y permanece en el original.")
            return None
        header = _header(sheet, bounds, selected)
        skipped = 0
        if header is None and not declared:
            # Un título o una leyenda breve puede preceder a un encabezado completo.
            while top + skipped < min(bottom, top + 5):
                row = sheet.frame.iloc[top + skipped - 1, left - 1:right]
                if sum(not is_empty(value) for value in row) != 1:
                    break
                skipped += 1
                header = _header(sheet, (left, top + skipped, right, bottom), selected)
                if header is not None:
                    break
        if header is None or (right == left and loaded.file_type == "xlsx" and not declared):
            warnings.append(f"La región {label} no tiene estructura tabular confiable y permanece en el original.")
            return None
        originals, depth = header
        start = top + skipped + depth
        if any(min_row < bottom + 1 and max_row >= start and min_col <= right and max_col >= left
               for min_col, min_row, max_col, max_row in sheet.merged_ranges):
            warnings.append(f"La región {label} contiene celdas combinadas en los datos y permanece en el original.")
            return None
        rows, source_rows = [], []
        omitted_notes, omitted_totals, formula_only_rows = 0, 0, 0
        for row_index in range(start - 1, bottom):
            values = [sheet.frame.iat[row_index, column] for column in selected]
            if all(is_empty(value) for value in values):
                continue
            occupied = [value for value in values if not is_empty(value)]
            if len(selected) > 1 and all(isinstance(value, str) and value.startswith("=") for value in occupied):
                formula_only_rows += 1
                continue
            if len(occupied) > 1 and isinstance(occupied[0], str) and occupied[0].strip().lower() in {
                "total", "totales", "subtotal", "total general",
            } and all(isinstance(value, (int, float, Decimal)) or
                      (isinstance(value, str) and (value.startswith("=") or re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value)))
                      for value in occupied[1:]):
                omitted_totals += 1
                continue
            # Sólo etiquetas explícitas identifican una leyenda; el texto largo puede ser un dato.
            if right > left and sum(not is_empty(value) for value in values) == 1:
                nonempty = occupied[0]
                if isinstance(nonempty, str) and re.match(r"^(?:nota|notas|fuente|leyenda|note|notes|source):", nonempty.strip(), re.IGNORECASE):
                    omitted_notes += 1
                    continue
            rows.append(values)
            source_rows.append(sheet.source_rows[row_index])
        if not rows:
            warnings.append(f"La región {label} contiene encabezados sin registros.")
            return None
        table_warnings = list(sheet.warnings)
        if skipped:
            table_warnings.append(f"{skipped} filas de título anteriores al encabezado permanecen en el original.")
        if omitted_notes:
            table_warnings.append(f"{omitted_notes} leyendas de una celda permanecen en el original.")
        if omitted_totals:
            table_warnings.append(f"{omitted_totals} filas etiquetadas como totales permanecen en el original.")
        if formula_only_rows:
            table_warnings.append(f"{formula_only_rows} filas compuestas sólo por fórmulas permanecen en el original porque no aportan registros interpretados.")
        sql_names = column_names(originals)
        columns, converted = [], []
        for index, (original, sql_name) in enumerate(zip(originals, sql_names)):
            sql_type, values, warning = infer_column(original, [row[index] for row in rows])
            metadata = {"original_name": original, "sql_name": sql_name, "sql_type": sql_type,
                        "source_column": selected[index] + 1,
                        "header_values": [sheet.frame.iat[row, selected[index]] for row in range(top + skipped - 1, start - 1)],
                        "numeric_eligible": sql_type in {"BIGINT", "NUMERIC"}, "unit": None, "meaning": None}
            unit = header_unit(original)
            if unit:
                metadata["unit"] = unit
                metadata["unit_evidence"] = original
            columns.append(metadata)
            converted.append(values)
            if warning:
                table_warnings.append(warning)
        records = [dict(zip(sql_names, values), _aquai_row=index, _aquai_source_row=source_row)
                   for index, (values, source_row) in enumerate(zip(zip(*converted), source_rows), 1)]
        locator = {"file_type": loaded.file_type, "sheet": sheet.name,
                   "range": range_ref,
                   "header_rows": [sheet.source_rows[index] for index in range(top + skipped - 1, start - 1)]}
        if loaded.file_type == "csv":
            locator.update(encoding=loaded.encoding, delimiter=loaded.delimiter)
        name = f"{PurePath(filename).stem} / {sheet.name} / {locator['range']}"
        visible_columns = ", ".join(originals[:8])[:400]
        description = f"Tabla de {sheet.name}, rango {locator['range']}, con {len(records)} registros. Columnas: {visible_columns}."
        return PreparedTable(name=name, source_locator=locator, columns=columns,
                             rows=records, description=description, warnings=table_warnings)
