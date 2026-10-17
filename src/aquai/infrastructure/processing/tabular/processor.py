"""Preparación determinística de tablas convencionales, sin planes del modelo."""

from pathlib import PurePath
from hashlib import sha256
import json
import re
from decimal import Decimal
from datetime import date

from openpyxl.utils.cell import get_column_letter

from aquai.application.contracts.ingestion import PreparationResult, PreparedTable, StructureSuggestion
from aquai.application.errors import FileTooLargeError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.normalization import column_names, header_unit, infer_column, numeric_projection, text_value
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
        column_groups = _groups(columns)
        # Una columna separadora no convierte un atributo sin nombre en otra tabla.
        if len(column_groups) > 1:
            independent = all(len(cols) >= 2 and _header(sheet, (cols[0] + 1, rows[0] + 1,
                                                               cols[-1] + 1, rows[-1] + 1))
                              for cols in column_groups)
            formula_block = any(all(is_empty(values[row][col]) or
                                    isinstance(values[row][col], str) and values[row][col].startswith("=")
                                    for row in rows[1:] for col in cols)
                                for cols in column_groups)
            first = [values[rows[0]][column] for column in columns]
            labels = [value for value in first if not is_empty(value)]
            shared_header = len(labels) >= 2 and len(labels) * 2 >= len(columns) and all(_header_text(value) for value in labels)
            if not independent and not formula_block and shared_header:
                column_groups = [columns]
        for cols in column_groups:
            bounds = (cols[0] + 1, rows[0] + 1, cols[-1] + 1, rows[-1] + 1)
            previous = next((index for index, region in reversed(list(enumerate(regions)))
                             if region[0] == bounds[0] and region[2] == bounds[2]
                             and _header(sheet, region) is not None), None)
            first_row = values[rows[0]][cols[0]:cols[-1] + 1]
            new_header = _header(sheet, bounds)
            if previous is not None and new_header is None and not all(_header_text(value) for value in first_row):
                prior = regions[previous]
                regions[previous] = (prior[0], prior[1], prior[2], bounds[3])
            else:
                regions.append(bounds)
    return regions


def _header_text(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and not value.startswith("=")
            and not (re.search(r"\d", value) and re.fullmatch(r"[+-]?[\d.,/ :T-]+", value)))


def _header(sheet: SheetData, bounds: tuple[int, int, int, int], columns: list[int] | None = None,
            depth: int | None = None) -> tuple[list[str], int] | None:
    left, top, right, bottom = bounds
    if top >= bottom:
        return None
    columns = columns if columns is not None else list(range(left - 1, right))
    original_row = [sheet.frame.iat[top - 1, column] for column in columns]
    relevant_merges = [merged for merged in sheet.merged_ranges
                       if top <= merged[1] <= top + 2 and merged[3] < min(bottom, top + 4)
                       and merged[0] >= left and merged[2] <= right]
    merged_depth = max((merged[3] - top + 1 + (merged[2] > merged[0])
                        for merged in relevant_merges), default=1)
    requested_depth = depth or merged_depth
    if 1 < requested_depth <= 4 and top + requested_depth <= bottom:
        grid = [[sheet.frame.iat[row, column] for column in columns]
                for row in range(top - 1, top + requested_depth - 1)]
        for min_col, min_row, max_col, max_row in relevant_merges:
            if max_row >= top + requested_depth:
                continue
            for row in range(min_row, max_row + 1):
                for index, column in enumerate(columns):
                    if min_col <= column + 1 <= max_col:
                        grid[row - top][index] = sheet.frame.iat[min_row - 1, min_col - 1]
        values = [value for row in grid for value in row if not is_empty(value)]
        sparse_context = any(sum(not is_empty(value) for value in row) < len(columns) for row in grid[:-1])
        labelled_groups = bool(relevant_merges) and any(_header_text(value) for value in grid[-1]) and all(
            is_empty(value) or _header_text(value) for value in grid[-1]
        ) and all(
            _header_text(value)
            or isinstance(value, (int, float, Decimal, date)) and not isinstance(value, bool)
            or isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value.strip()) is not None
            for row in grid[:-1] for value in row if not is_empty(value)
        )
        if values and (all(_header_text(value) for value in values) or labelled_groups) and (relevant_merges or sparse_context):
            names = [" | ".join(dict.fromkeys(str(row[index]) for row in grid if not is_empty(row[index])))
                     or f"Columna {get_column_letter(column + 1)}"
                     for index, column in enumerate(columns)]
            return names, requested_depth
    if top + 1 < bottom:
        second = [sheet.frame.iat[top, column] for column in columns]
        labels = [value for value in second if not is_empty(value)]
        documented_units = sum(isinstance(value, str) and header_unit(f"unit [{value}]") is not None for value in labels)
        complete = all(_header_text(parent) or _header_text(child) for parent, child in zip(original_row, second))
        textual = all(is_empty(value) or _header_text(value) for value in original_row + second)
        body = [sheet.frame.iat[top + 1, column] for column in columns]
        numeric_body = any(isinstance(value, (int, float, Decimal, date)) and not isinstance(value, bool)
                           or isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", value) for value in body)
        if depth != 1 and complete and textual and numeric_body and len(labels) >= 2 and documented_units * 2 >= len(labels):
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
    if depth not in (None, 1):
        return None
    labels = [value for value in original_row if not is_empty(value)]
    partial = len(labels) != len(columns)
    next_row = [sheet.frame.iat[top, column] for column in columns]
    # Códigos sobre columnas y otra fila de etiquetas sugieren una matriz con contexto.
    if sum(isinstance(value, str) and re.match(r"^\d", value) is not None for value in labels) * 2 > len(labels):
        if sum(_header_text(value) for value in next_row) * 2 > len(columns):
            return None
    evidence = any(isinstance(value, (int, float, Decimal, date)) and not isinstance(value, bool)
                   or isinstance(value, str) and (value.startswith("=") or re.match(r"^[+-]?\d", value))
                   for row in range(top, min(bottom, top + 3)) for value in sheet.frame.iloc[row, columns])
    if labels and all(_header_text(value) for value in labels) and (
        not partial or len(labels) >= 2 and len(labels) * 2 >= len(columns) and evidence
    ):
        return [str(value) if not is_empty(value) else f"Columna {get_column_letter(column + 1)}"
                for value, column in zip(original_row, columns)], 1
    return None


class SimpleTabularProcessor:
    def __init__(self, settings: TabularSettings) -> None:
        self.settings = settings
        self.reader = TableReader(settings)

    def validate_upload(self, filename: str, content: bytes) -> str:
        return self.reader.validate(filename, content)

    def prepare(self, filename: str, content: bytes,
                suggestions: list[StructureSuggestion] | None = None) -> PreparationResult:
        loaded = self.reader.read(filename, content)
        result = PreparationResult(warnings=list(loaded.warnings))
        suggestions = suggestions or []
        hints = {hint.region_id: hint for hint in suggestions}
        if len(hints) != len(suggestions):
            hints = {}
            result.warnings.append("La asistencia repitió regiones y no se aplicó.")
        for sheet_index, sheet in enumerate(loaded.sheets):
            result.warnings.extend(f"{sheet.name}: {warning}" for warning in sheet.warnings)
            regions = _regions(sheet, loaded.file_type)
            if not regions and not sheet.has_declared_tables:
                result.warnings.append(f"La hoja '{sheet.name}' está vacía.")
            for bounds in regions:
                warning_start = len(result.warnings)
                table = self._prepare_region(filename, loaded, sheet, bounds, result.warnings)
                left, top, right, bottom = bounds
                region_id = f"{sheet_index}:{left}:{top}:{right}:{bottom}"
                if table is None and region_id in hints:
                    table = self._assisted_region(filename, loaded, sheet, bounds, hints[region_id])
                    if table is not None:
                        del result.warnings[warning_start:]
                    else:
                        result.warnings.append(f"La sugerencia para {sheet.name} no pudo verificarse; se conserva el original.")
                elif table is None and self.settings.structure_assistance and len(result.structure_candidates) < 3:
                    evidence = any(_header_text(value) for row in sheet.frame.iloc[top:bottom, left - 1:right].values for value in row)
                    if bottom - top >= 2 and right > left and evidence:
                        result.structure_candidates.append(self._structure_sample(sheet, bounds, region_id))
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
        self._mark_overlap(result.tables)
        return result

    @staticmethod
    def _structure_sample(sheet: SheetData, bounds: tuple[int, int, int, int], region_id: str) -> dict:
        left, top, right, bottom = bounds
        row_indices = sorted(set(range(top - 1, min(bottom, top + 7))) | {bottom - 1, (top + bottom) // 2 - 1})
        column_indices = sorted(set(range(left - 1, min(right, left + 9))) | {right - 1})
        return {
            "region_id": region_id, "sheet": sheet.name, "range": f"{get_column_letter(left)}{top}:{get_column_letter(right)}{bottom}",
            "shape": [bottom - top + 1, right - left + 1], "sample_is_partial": True,
            "sample": [{"row": row - top + 2, "cells": {
                str(column - left + 2): None if is_empty(sheet.frame.iat[row, column]) else str(sheet.frame.iat[row, column])[:100]
                for column in column_indices}} for row in row_indices],
            "merged_ranges": [[a - left + 1, b - top + 1, c - left + 1, d - top + 1]
                              for a, b, c, d in sheet.merged_ranges
                              if left <= a <= c <= right and top <= b <= d <= bottom][:20],
        }

    def _assisted_region(self, filename: str, loaded: LoadedFile, sheet: SheetData,
                         bounds: tuple[int, int, int, int], hint: StructureSuggestion) -> PreparedTable | None:
        left, top, right, bottom = bounds
        if hint.orientation not in {"rows", "columns"} or any(type(value) is not int for value in
            (hint.header_start, hint.header_depth, hint.data_start)):
            return None
        if not 1 <= hint.header_depth <= 4 or hint.header_start < 1 or hint.data_start != hint.header_start + hint.header_depth:
            return None
        if any(a <= right and c >= left and b <= bottom and d >= top
               and not (left <= a <= c <= right and top <= b <= d <= bottom)
               for a, b, c, d in sheet.merged_ranges):
            return None
        if hint.orientation == "columns":
            fragment = sheet.frame.iloc[top - 1:bottom, left - 1:right].T
            merges = [(b - top + 1, a - left + 1, d - top + 1, c - left + 1)
                      for a, b, c, d in sheet.merged_ranges
                      if left <= a <= c <= right and top <= b <= d <= bottom]
            working = SheetData(sheet.name, fragment, list(range(left, right + 1)), merges,
                                warnings=list(sheet.warnings))
            region = (1, hint.header_start, fragment.shape[1], fragment.shape[0])
        else:
            working, region = sheet, (left, top + hint.header_start - 1, right, bottom)
        a, b, c, d = region
        if b + hint.header_depth > d or c - a + 1 > self.settings.max_columns:
            return None
        prefix = working.frame.iloc[(0 if hint.orientation == "columns" else top - 1):b - 1, a - 1:c]
        if any(len(present := [value for value in row if not is_empty(value)]) > 1
               or any(not _header_text(value) for value in present) for row in prefix.values.tolist()):
            return None
        header = _header(working, region, depth=hint.header_depth) or _header(working, region)
        if header is None:
            return None
        verified_depth = header[1]
        table = self._prepare_region(filename, loaded, working, region, [], header_depth=verified_depth)
        if table is None:
            return None
        table.source_locator["structure_assistance"] = {
            "verified": True, "orientation": hint.orientation,
            "header_start": hint.header_start, "header_depth": verified_depth,
            "suggested_header_depth": hint.header_depth,
        }
        if hint.header_start > 1:
            table.warnings.append(f"{hint.header_start - 1} posiciones anteriores al encabezado asistido quedan en el original como contexto no interpretado.")
        if hint.orientation == "columns":
            for column in table.columns:
                column["source_row"] = top + column.pop("source_column") - 1
            for row in table.rows:
                row["_aquai_source_column"] = row["_aquai_source_row"]
                row["_aquai_source_row"] = top
            table.source_locator.update(
                range=f"{get_column_letter(left)}{top}:{get_column_letter(right)}{bottom}",
                orientation="columns", header_columns=list(range(left + hint.header_start - 1, left + hint.header_start + verified_depth - 1)),
            )
            table.source_locator.pop("header_rows", None)
            table.name = f"{PurePath(filename).stem} / {sheet.name} / {table.source_locator['range']}"
            table.description = f"Matriz de {sheet.name}, rango {table.source_locator['range']}, con {len(table.rows)} columnas de origen almacenadas como registros."
        table.warnings.append("La orientación y los encabezados sugeridos por Qwen se verificaron contra el archivo completo.")
        return table

    @staticmethod
    def _mark_overlap(tables: list[PreparedTable]) -> None:
        previous: list[tuple[PreparedTable, tuple]] = []
        for table in tables:
            schema = tuple((column["original_name"], column["unit"]) for column in table.columns)
            overlaps = []
            for other, other_schema in previous:
                common = min(len(schema), len(other_schema))
                if common < 2 or schema[:common] != other_schema[:common]:
                    continue
                def fingerprints(prepared: PreparedTable) -> list[str]:
                    return [sha256(json.dumps([row[column["sql_name"]] for column in prepared.columns[:common]],
                                              default=str, ensure_ascii=False).encode()).hexdigest() for row in prepared.rows]
                known = set(fingerprints(other))
                count = sum(item in known for item in fingerprints(table))
                if count:
                    overlaps.append({"with": other.name, "matching_rows": count, "compared_columns": common})
            if overlaps:
                table.source_locator["possible_overlap"] = overlaps
                table.warnings.append("Hay filas coincidentes en otros datasets del archivo; revisar possible_overlap antes de combinar observaciones.")
            previous.append((table, schema))

    def _prepare_region(self, filename: str, loaded: LoadedFile, sheet: SheetData,
                        bounds: tuple[int, int, int, int], warnings: list[str],
                        header_depth: int | None = None) -> PreparedTable | None:
        left, top, right, bottom = bounds
        range_ref = f"{get_column_letter(left)}{top}:{get_column_letter(right)}{bottom}"
        label = f"{sheet.name}!{range_ref}"
        declared = bounds in sheet.table_ranges
        selected = [column for column in range(left - 1, right)
                    if any(not is_empty(sheet.frame.iat[row, column]) for row in range(top - 1, bottom))]
        if len(selected) > self.settings.max_columns:
            warnings.append(f"La región {label} supera el límite de columnas por dataset y permanece en el original.")
            return None
        header = _header(sheet, bounds, selected, depth=header_depth)
        skipped = 0
        if header is None and not declared and header_depth is None:
            # Un título o una leyenda breve puede preceder a un encabezado completo.
            while top + skipped < min(bottom, top + 5):
                row = sheet.frame.iloc[top + skipped - 1, left - 1:right]
                if sum(not is_empty(value) for value in row) > 1 or any(not _header_text(value) for value in row if not is_empty(value)):
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
                        "header_values": [text_value(sheet.frame.iat[row, selected[index]]) for row in range(top + skipped - 1, start - 1)],
                        "numeric_eligible": sql_type in {"BIGINT", "NUMERIC"}, "unit": None, "meaning": None}
            unit = header_unit(original)
            if unit:
                metadata["unit"] = unit
                metadata["unit_evidence"] = original
            columns.append(metadata)
            converted.append(values)
            if all(is_empty(value) for value in metadata["header_values"]):
                metadata["header_missing"] = True
                table_warnings.append(f"La columna {get_column_letter(selected[index] + 1)} no tiene encabezado; su nombre es provisional.")
            if warning:
                table_warnings.append(warning)
        for index, metadata in enumerate(list(columns)):
            if metadata["sql_type"] != "TEXT" or metadata.get("header_missing") or len(columns) >= self.settings.max_columns:
                continue
            projection = numeric_projection(metadata["original_name"], [row[index] for row in rows])
            if projection is None:
                continue
            values, excluded = projection
            base = metadata["sql_name"][:53] + "_numeric"
            name, suffix = base, 2
            while name in {column["sql_name"] for column in columns}:
                name, suffix = base[:59] + f"_{suffix}", suffix + 1
            columns.append({**metadata, "original_name": metadata["original_name"] + " (representación numérica)",
                            "sql_name": name, "sql_type": "NUMERIC", "numeric_eligible": True,
                            "derived_from": metadata["sql_name"], "representation": "numeric_projection",
                            "excluded_count": excluded, "nonempty_count": sum(not is_empty(value) for value in values) + excluded})
            converted.append(values)
            table_warnings.append(f"{metadata['original_name']}: representación numérica auxiliar; {excluded} valores no convertibles conservados en el texto original y excluidos de cálculos.")
        sql_names = [column["sql_name"] for column in columns]
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
