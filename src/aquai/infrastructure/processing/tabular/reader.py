"""Lectura acotada que conserva valores, fórmulas y filas originales."""

import csv
import io
import posixpath
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePath, PureWindowsPath
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.infrastructure.config.tabular import TabularSettings


class LiteralText(str):
    """Valor numérico que debe conservarse como texto por evidencia del Excel."""


@dataclass
class SheetData:
    name: str
    frame: pd.DataFrame
    source_rows: list[int]
    merged_ranges: list[tuple[int, int, int, int]] = field(default_factory=list)
    table_ranges: list[tuple[int, int, int, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_declared_tables: bool = False


@dataclass
class LoadedFile:
    file_type: str
    sheets: list[SheetData]
    warnings: list[str] = field(default_factory=list)
    encoding: str | None = None
    delimiter: str | None = None


def is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and value == "")


class TableReader:
    def __init__(self, settings: TabularSettings) -> None:
        self.settings = settings

    def validate(self, filename: str, content: bytes) -> str:
        if not isinstance(filename, str) or not filename.strip() or any(ord(char) < 32 for char in filename):
            raise TabularInputError("El archivo debe tener un nombre válido.")
        filename = PureWindowsPath(filename).name.strip()
        if not filename or len(filename) > 255:
            raise TabularInputError("El nombre del archivo supera el largo permitido.")
        suffix = PurePath(filename).suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            raise TabularInputError("Sólo se admiten archivos CSV y XLSX.")
        if not content:
            raise TabularInputError("El archivo está vacío.")
        if len(content) > self.settings.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño permitido.")
        try:
            if suffix == ".csv":
                self._csv_rows(content)
            else:
                self._check_archive(content)
        except (BadZipFile, KeyError, ElementTree.ParseError, UnicodeError, csv.Error,
                OSError, ValueError) as error:
            if isinstance(error, TabularInputError):
                raise
            raise TabularInputError("El archivo no tiene un formato tabular válido.") from error
        return filename

    def read(self, filename: str, content: bytes) -> LoadedFile:
        filename = self.validate(filename, content)
        file_type = PurePath(filename).suffix.lower()[1:]
        if file_type == "csv":
            rows, source_rows, encoding, delimiter, warnings = self._csv_rows(content)
            width = max(map(len, rows))
            padded = [[None if value == "" else value for value in row]
                      + [None] * (width - len(row)) for row in rows]
            sheet = SheetData("data", pd.DataFrame(padded, dtype=object), source_rows)
            return LoadedFile(file_type, [sheet], warnings, encoding, delimiter)
        try:
            return self._read_excel(content)
        except (ValueError, KeyError, OSError, ElementTree.ParseError, InvalidOperation) as error:
            raise TabularInputError("No se pudo leer el contenido del XLSX.") from error

    def _csv_rows(self, content: bytes) -> tuple[list[list[str]], list[int], str, str, list[str]]:
        warnings = []
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        else:
            encoding = "utf-8-sig"
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            encoding = "cp1252"
            text = content.decode(encoding)
            warnings.append("El CSV se leyó con CP1252 porque no contiene UTF-8 válido.")
        if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
            raise TabularInputError("El CSV contiene controles binarios y no es texto válido.")
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            candidates = {}
            for candidate in (",", ";", "\t", "|"):
                reader = csv.reader(io.StringIO(text[:8192]), delimiter=candidate)
                widths = [len(row) for _, row in zip(range(20), reader) if row]
                candidates[candidate] = sum(width > 1 for width in widths)
            delimiter = max(candidates, key=candidates.get)
            if candidates[delimiter]:
                warnings.append("El separador del CSV se infirió de filas con anchos distintos.")
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        rows, source_rows = [], []
        width, last_line = 0, 0
        for row in reader:
            source_rows.append(last_line + 1)
            last_line = reader.line_num
            width = max(width, len(row))
            rows.append(row)
            if width > self.settings.max_columns:
                raise FileTooLargeError("El CSV supera el límite de columnas permitido.")
            if len(rows) * max(width, 1) > self.settings.max_cells:
                raise FileTooLargeError("El CSV supera el límite de celdas permitido.")
        if not rows or not any(any(not is_empty(value) for value in row) for row in rows):
            raise TabularInputError("El CSV no contiene valores para procesar.")
        if len({len(row) for row in rows if row}) > 1:
            warnings.append("Las filas de distinto ancho se completaron con celdas vacías.")
        return rows, source_rows, encoding, delimiter, warnings

    def _check_archive(self, content: bytes) -> None:
        with ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(entries) > 10_000 or sum(entry.file_size for entry in entries) > self.settings.max_expanded_bytes:
                raise FileTooLargeError("El XLSX supera el tamaño descomprimido permitido.")
            if len(names) != len(set(names)):
                raise TabularInputError("El XLSX contiene partes duplicadas.")
            if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(names):
                raise TabularInputError("El archivo no es un libro XLSX válido.")
            if any(entry.flag_bits & 1 or entry.filename.startswith(("/", "\\"))
                   or ".." in entry.filename.replace("\\", "/").split("/") for entry in entries):
                raise TabularInputError("El XLSX contiene partes cifradas o rutas no válidas.")
            if any(name.lower().endswith("vbaproject.bin") for name in names):
                raise TabularInputError("Sólo se admiten XLSX sin macros.")
            total_cells, sheet_count = 0, 0
            for entry in entries:
                if not entry.filename.endswith((".xml", ".rels")):
                    continue
                data = archive.read(entry)
                if b"\x00" in data or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
                    raise TabularInputError("El XLSX contiene declaraciones XML no permitidas.")
                if not entry.filename.startswith("xl/worksheets/") or not entry.filename.endswith(".xml"):
                    continue
                sheet_count += 1
                if sheet_count > self.settings.max_sheets:
                    raise FileTooLargeError("El XLSX supera el límite de hojas permitido.")
                max_row, max_column, _, _, _ = self._worksheet_layout(archive, entry.filename)
                if total_cells + max_row * max_column > self.settings.max_cells:
                    raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")
                total_cells += max_row * max_column
            if not sheet_count:
                raise TabularInputError("El XLSX no contiene hojas de cálculo.")
            paths = self._worksheet_paths(archive)
            if len(paths) > self.settings.max_sheets or not paths:
                raise TabularInputError("El XLSX no declara un conjunto válido de hojas.")

    def _read_excel(self, content: bytes) -> LoadedFile:
        workbook = load_workbook(io.BytesIO(content), data_only=False, read_only=True, keep_links=False)
        try:
            with ZipFile(io.BytesIO(content)) as archive:
                paths = self._worksheet_paths(archive)
            sheets = []
            for worksheet in workbook.worksheets:
                with ZipFile(io.BytesIO(content)) as archive:
                    max_row, max_column, raw_numbers, merged_ranges, table_parts = self._worksheet_layout(
                        archive, paths[worksheet.title], include_values=True,
                    )
                    table_ranges, table_warnings = self._declared_tables(archive, paths[worksheet.title], table_parts,
                                                                         max_row, max_column)
                rows, formula_count, partial_date_count, formatted_code_count, error_count, time_count = [], 0, 0, 0, 0, 0
                for source_row in worksheet.iter_rows(max_row=max(max_row, 1), max_col=max(max_column, 1)):
                    values = []
                    for cell in source_row:
                        value = cell.value
                        raw = raw_numbers.get(getattr(cell, "coordinate", ""))
                        if cell.data_type == "f":
                            formula_count += 1
                            value = value if isinstance(value, str) else getattr(value, "text", None)
                            if value is None:
                                value = "[Fórmula Excel no textual conservada en el original]"
                        elif cell.data_type == "e":
                            error_count += 1
                        elif isinstance(value, (date, datetime)):
                            format_text = re.sub(r'"[^"]*"|\[[^\]]*\]', "", cell.number_format).lower()
                            if "d" not in format_text and raw is not None:
                                value = LiteralText(raw)
                                partial_date_count += 1
                            elif isinstance(value, datetime) and not any(token in format_text for token in ("h", "s")):
                                value = value.date()
                        elif isinstance(value, (time, timedelta)):
                            value = LiteralText(raw if raw is not None else str(value))
                            time_count += 1
                        elif raw is not None and not isinstance(value, bool):
                            numeric = Decimal(raw)
                            if re.fullmatch(r"0{2,}", cell.number_format) and numeric == numeric.to_integral_value() and numeric >= 0:
                                value = LiteralText(str(int(numeric)).zfill(len(cell.number_format)))
                                formatted_code_count += 1
                            else:
                                value = numeric
                        values.append(value)
                    rows.append(values)
                warnings = list(table_warnings)
                if formula_count:
                    warnings.append(f"{formula_count} fórmulas se conservan como expresiones de texto; no se recalcularon.")
                if error_count:
                    warnings.append(f"{error_count} errores Excel se conservan literalmente como texto.")
                if partial_date_count:
                    warnings.append(f"{partial_date_count} fechas con formato parcial conservan su serial original como texto.")
                if formatted_code_count:
                    warnings.append(f"{formatted_code_count} códigos conservan los ceros documentados por su formato Excel.")
                if time_count:
                    warnings.append(f"{time_count} horarios o duraciones conservan su valor de origen como texto.")
                sheets.append(SheetData(
                    worksheet.title, pd.DataFrame(rows, dtype=object), list(range(1, len(rows) + 1)),
                    merged_ranges, table_ranges, warnings, bool(table_parts),
                ))
            return LoadedFile("xlsx", sheets)
        finally:
            workbook.close()

    def _worksheet_paths(self, archive: ZipFile) -> dict[str, str]:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for relation in relationships:
            if not relation.attrib.get("Type", "").endswith("/worksheet"):
                continue
            target = relation.attrib.get("Target", "")
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            if relation.attrib.get("TargetMode") == "External" or not path.startswith("xl/worksheets/"):
                raise TabularInputError("El XLSX contiene una hoja externa o una ruta no válida.")
            if path not in archive.namelist():
                raise TabularInputError("El XLSX declara una hoja que no existe.")
            targets[relation.attrib["Id"]] = path
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        paths = {}
        for element in workbook.iter():
            if element.tag.rsplit("}", 1)[-1] == "sheet":
                relation_id = next((value for key, value in element.attrib.items()
                                    if key.rsplit("}", 1)[-1] == "id"), None)
                if relation_id in targets:
                    paths[element.attrib["name"]] = targets[relation_id]
        return paths

    def _worksheet_layout(self, archive: ZipFile, path: str, include_values: bool = False) -> tuple:
        """Sólo las celdas con contenido delimitan la grilla; el estilo vacío no la agranda."""
        max_row, max_column, values, merged_ranges, table_parts = 0, 0, {}, [], []
        with archive.open(path) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "c":
                    content = [child for child in element.iter()
                               if child.tag.rsplit("}", 1)[-1] in {"v", "t", "f"}]
                    if any(child.text not in {None, ""} or child.tag.rsplit("}", 1)[-1] == "f" for child in content):
                        row, column = coordinate_to_tuple(element.attrib["r"])
                        max_row, max_column = max(max_row, row), max(max_column, column)
                        if max_row * max_column > self.settings.max_cells:
                            raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")
                        if include_values and element.attrib.get("t", "n") == "n":
                            raw = next((child.text for child in content if child.tag.rsplit("}", 1)[-1] == "v"), None)
                            if raw is not None:
                                values[element.attrib["r"]] = raw
                    element.clear()
                elif tag == "mergeCell":
                    merged_ranges.append(range_boundaries(element.attrib["ref"]))
                    element.clear()
                elif tag == "tablePart":
                    table_parts.append(next(value for key, value in element.attrib.items()
                                            if key.rsplit("}", 1)[-1] == "id"))
                    element.clear()
                elif tag == "row":
                    element.clear()
        return max_row, max_column, values, merged_ranges, table_parts

    def _declared_tables(self, archive: ZipFile, sheet_path: str, table_parts: list[str],
                         max_row: int, max_column: int) -> tuple[list[tuple], list[str]]:
        if not table_parts:
            return [], []
        base = posixpath.dirname(sheet_path)
        relationship_path = posixpath.join(base, "_rels", posixpath.basename(sheet_path) + ".rels")
        relationships = ElementTree.fromstring(archive.read(relationship_path))
        targets = {}
        for relation in relationships:
            target = relation.attrib.get("Target", "")
            if relation.attrib.get("Id") in table_parts:
                if relation.attrib.get("TargetMode") == "External":
                    raise TabularInputError("El XLSX declara una tabla externa.")
                targets[relation.attrib["Id"]] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(base, target))
        ranges, warnings = [], []
        for part in table_parts:
            table = ElementTree.fromstring(archive.read(targets[part]))
            name = table.attrib.get("displayName", "tabla")
            if table.attrib.get("headerRowCount", "1") == "0":
                warnings.append(f"La tabla {name} no declara encabezados y se conserva en el original.")
                continue
            bounds = list(range_boundaries(table.attrib["ref"]))
            totals = int(table.attrib.get("totalsRowCount", "0"))
            if totals or table.attrib.get("totalsRowShown", "0") in {"1", "true"}:
                bounds[3] -= totals or 1
                warnings.append(f"La fila de totales declarada en {name} se conserva en el original.")
            bounds[2], bounds[3] = min(bounds[2], max_column), min(bounds[3], max_row)
            if bounds[0] <= bounds[2] and bounds[1] < bounds[3]:
                ranges.append(tuple(bounds))
        return ranges, warnings
