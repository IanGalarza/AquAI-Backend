"""Lectura de tablas en memoria y límites previos al procesamiento."""

import csv
import io
import posixpath
from dataclasses import dataclass, field
from pathlib import PurePath
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple

from aquai.application.errors import FileTooLargeError, TabularInputError


@dataclass
class FormulaData:
    expression: str
    cached_value: object


@dataclass
class SheetData:
    name: str
    frame: pd.DataFrame
    merged_cells: list[str] = field(default_factory=list)
    formulas: list[dict[str, object]] = field(default_factory=list)
    formula_count: int = 0
    formula_rows: set[int] = field(default_factory=set)
    formula_columns: set[int] = field(default_factory=set)
    formula_cells: dict[tuple[int, int], FormulaData] = field(default_factory=dict)
    missing_formula_cache_count: int = 0
    missing_formula_cells: list[str] = field(default_factory=list)
    error_cell_count: int = 0
    error_cells: list[str] = field(default_factory=list)
    error_positions: set[tuple[int, int]] = field(default_factory=set)


@dataclass
class LoadedTable:
    file_type: str
    sheets: list[SheetData]
    warnings: list[str] = field(default_factory=list)
    csv_encoding: str | None = None
    csv_delimiter: str | None = None


class TableReader:
    def __init__(
        self,
        max_cells: int = 2_000_000,
        max_sheets: int = 30,
        max_expanded_bytes: int = 100 * 1024 * 1024,
        sample_rows: int = 12,
    ) -> None:
        self.max_cells = max_cells
        self.max_sheets = max_sheets
        self.max_expanded_bytes = max_expanded_bytes
        self.sample_rows = sample_rows
        self.max_columns = 1_000

    def read(self, file_name: str, content: bytes) -> LoadedTable:
        suffix = PurePath(file_name).suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            raise TabularInputError("Sólo se admiten archivos CSV y XLSX.")
        if not content:
            raise TabularInputError("El archivo está vacío.")
        try:
            if suffix == ".csv":
                return self._read_csv(content)
            return self._read_excel(content)
        except TabularInputError:
            raise
        except (ValueError, OSError, BadZipFile, ElementTree.ParseError, csv.Error) as error:
            raise TabularInputError("No se pudo abrir el archivo tabular.") from error
        except Exception as error:
            raise TabularInputError("El archivo tabular no tiene un formato válido.") from error

    def _read_csv(self, content: bytes) -> LoadedTable:
        warnings = []
        encoding = "utf-8-sig"
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp1252")
            encoding = "cp1252"
            warnings.append("El CSV se leyó con codificación CP1252 porque no es UTF-8.")
        if "\x00" in text:
            raise TabularInputError("El CSV contiene bytes nulos y no es un archivo de texto válido.")
        try:
            delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            widths_by_delimiter = {}
            for candidate in (",", ";", "\t", "|"):
                sampled = list(csv.reader(io.StringIO(text[:8192]), delimiter=candidate))[:20]
                widths_by_delimiter[candidate] = sum(len(row) > 1 for row in sampled)
            delimiter = max(widths_by_delimiter, key=widths_by_delimiter.get)
            if widths_by_delimiter[delimiter]:
                warnings.append(
                    "El separador del CSV se infirió a partir de sus filas "
                    "porque sus anchos no son uniformes."
                )
        row_count = 0
        column_count = 0
        widths: set[int] = set()
        has_value = False
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True):
            row_count += 1
            column_count = max(column_count, len(row))
            widths.add(len(row))
            has_value = has_value or any(value.strip() for value in row)
            if row_count * max(column_count, 1) > self.max_cells:
                raise FileTooLargeError("El CSV supera el límite de celdas permitido.")
            if column_count > self.max_columns:
                raise FileTooLargeError("El CSV supera el límite de 1000 columnas permitido.")
        if not has_value:
            raise TabularInputError("El CSV no contiene valores para procesar.")
        if len(widths) > 1:
            warnings.append(
                "El CSV tiene filas de distinto ancho; "
                "las celdas faltantes se completaron con nulos."
            )
        frame = pd.read_csv(
            io.StringIO(text, newline=""),
            sep=delimiter,
            header=None,
            names=list(range(column_count)),
            dtype=object,
            engine="python",
            keep_default_na=False,
            na_values=[""],
            skip_blank_lines=False,
            on_bad_lines="error",
        )
        if frame.shape[0] != row_count:
            raise TabularInputError("No se pudo conservar la numeración original de las filas del CSV.")
        return LoadedTable("csv", [SheetData("data", frame)], warnings, encoding, delimiter)

    def _check_excel_archive(self, content: bytes) -> None:
        with ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > 10_000 or sum(entry.file_size for entry in entries) > self.max_expanded_bytes:
                raise FileTooLargeError("El XLSX supera el límite de tamaño descomprimido permitido.")
            sheet_names = self._worksheet_parts(archive)
            if len(sheet_names) > self.max_sheets:
                raise FileTooLargeError("El XLSX supera el límite de hojas permitido.")
            total_cells = 0
            for sheet_name in sheet_names:
                max_row = 0
                max_column = 0
                with archive.open(sheet_name) as stream:
                    for _, element in ElementTree.iterparse(stream, events=("end",)):
                        tag = element.tag.rsplit("}", 1)[-1]
                        if tag in {"dimension", "mergeCell"}:
                            coordinate = element.attrib.get("ref", "A1").split(":")[-1]
                        elif tag == "c" and "r" in element.attrib:
                            coordinate = element.attrib["r"]
                        else:
                            element.clear()
                            continue
                        row, column = coordinate_to_tuple(coordinate)
                        max_row = max(max_row, row)
                        max_column = max(max_column, column)
                        if max_row * max_column + total_cells > self.max_cells:
                            raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")
                        if max_column > self.max_columns:
                            raise FileTooLargeError("El XLSX supera el límite de 1000 columnas permitido.")
                        element.clear()
                total_cells += max_row * max_column

    def _worksheet_parts(self, archive: ZipFile) -> set[str]:
        content_types = ElementTree.fromstring(archive.read("[Content_Types].xml"))
        worksheet_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
        names = {
            element.attrib["PartName"].lstrip("/")
            for element in content_types
            if element.attrib.get("ContentType") == worksheet_type
        }
        for entry in archive.infolist():
            if not entry.filename.endswith(".rels"):
                continue
            base = posixpath.dirname(posixpath.dirname(entry.filename))
            relations = ElementTree.fromstring(archive.read(entry.filename))
            for relation in relations:
                if not relation.attrib.get("Type", "").endswith("/worksheet"):
                    continue
                if relation.attrib.get("TargetMode") == "External":
                    raise TabularInputError("El XLSX contiene una hoja externa que no se puede procesar.")
                target = relation.attrib.get("Target", "")
                if target.startswith("/"):
                    name = target.lstrip("/")
                else:
                    name = posixpath.normpath(posixpath.join(base, target))
                names.add(name)
        return names

    def _read_excel(self, content: bytes) -> LoadedTable:
        self._check_excel_archive(content)
        workbook = load_workbook(io.BytesIO(content), data_only=False, read_only=False, keep_links=False)
        cached_workbook = None
        try:
            cached_workbook = load_workbook(
                io.BytesIO(content), data_only=True, read_only=True, keep_links=False,
            )
            if len(workbook.worksheets) > self.max_sheets:
                raise FileTooLargeError("El XLSX supera el límite de hojas permitido.")
            total_cells = sum(sheet.max_row * sheet.max_column for sheet in workbook.worksheets)
            if total_cells > self.max_cells:
                raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")
            if any(sheet.max_column > self.max_columns for sheet in workbook.worksheets):
                raise FileTooLargeError("El XLSX supera el límite de 1000 columnas permitido.")
            sheets = []
            for worksheet in workbook.worksheets:
                cached_sheet = cached_workbook[worksheet.title]
                cached_sheet.reset_dimensions()
                cached_rows = cached_sheet.iter_rows(
                    min_row=1,
                    max_row=worksheet.max_row,
                    min_col=1,
                    max_col=worksheet.max_column,
                    values_only=False,
                )
                rows = []
                error_cells = []
                error_cell_count = 0
                error_positions = set()
                for cached_row in cached_rows:
                    rows.append([cell.value for cell in cached_row])
                    for cell in cached_row:
                        if cell.data_type == "e":
                            error_cell_count += 1
                            error_positions.add((cell.row, cell.column))
                            if len(error_cells) < 100:
                                error_cells.append(cell.coordinate)
                data = SheetData(
                    worksheet.title,
                    pd.DataFrame(rows, dtype=object),
                    [str(merged) for merged in sorted(
                        worksheet.merged_cells.ranges,
                        key=lambda merged: (merged.min_row, merged.min_col, merged.max_row, merged.max_col),
                    )],
                )
                data.error_cell_count = error_cell_count
                data.error_cells = error_cells
                data.error_positions = error_positions
                formula_cells = [
                    cell for row in worksheet.iter_rows() for cell in row if cell.data_type == "f"
                ]
                data.formula_count = len(formula_cells)
                sample_count = min(self.sample_rows, data.formula_count)
                head_count = min(3, sample_count)
                sample_positions = set(range(head_count))
                remaining = sample_count - head_count
                for offset in range(remaining):
                    sample_positions.add(
                        head_count
                        + (data.formula_count - head_count - 1) * (offset + 1) // remaining
                    )
                for index, cell in enumerate(formula_cells):
                    data.formula_rows.add(cell.row)
                    data.formula_columns.add(cell.column)
                    cached_value = data.frame.iat[cell.row - 1, cell.column - 1]
                    data.formula_cells[(cell.row, cell.column)] = FormulaData(
                        expression=str(cell.value), cached_value=cached_value,
                    )
                    if cached_value is None:
                        data.missing_formula_cache_count += 1
                        if len(data.missing_formula_cells) < 100:
                            data.missing_formula_cells.append(cell.coordinate)
                    if index in sample_positions:
                        data.formulas.append({
                            "cell": cell.coordinate,
                            "formula": str(cell.value),
                            "cached_value": cached_value,
                        })
                sheets.append(data)
            if not sheets:
                raise TabularInputError("El XLSX no contiene hojas de cálculo.")
            return LoadedTable("xlsx", sheets)
        finally:
            workbook.close()
            if cached_workbook is not None:
                cached_workbook.close()
