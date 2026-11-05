import csv
import io
import posixpath
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePath, PureWindowsPath
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile, ZipInfo

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.infrastructure.config.tabular import TabularSettings

_CSV_DELIMITERS = (",", ";", "\t", "|")
_UNFORMATTED_FORMULA = "[Fórmula Excel no textual conservada en el original]"


class LiteralText(str):
    pass


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


@dataclass
class _SheetLayout:
    max_row: int = 0
    max_column: int = 0
    values: dict[str, str] = field(default_factory=dict)
    merged_ranges: list[tuple] = field(default_factory=list)
    table_parts: list[str] = field(default_factory=list)


@dataclass
class _ConversionTally:
    formulas: int = 0
    errors: int = 0
    partial_dates: int = 0
    formatted_codes: int = 0
    times: int = 0


#
# Indica si un valor de celda está vacío.
#
def is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str) and value == "")


#
# Devuelve el nombre de una etiqueta XML sin su espacio de nombres.
#
def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


#
# Devuelve los valores de los atributos de identificador de un elemento XML.
#
def _attribute_ids(element):
    return (value for key, value in element.attrib.items() if _local_name(key) == "id")


#
# Devuelve el identificador de relación de una parte de tabla del XLSX.
#
def _table_part_id(element) -> str:
    relation_id = next(_attribute_ids(element), None)

    if relation_id is None:
        raise TabularInputError("El XLSX declara una tabla sin identificador de relación.")

    return relation_id


#
# Resuelve la ruta de una parte del XLSX a partir de su base y su destino.
#
def _part_path(base: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")

    return posixpath.normpath(posixpath.join(base, target))


#
# Indica si una entrada del ZIP está cifrada o tiene una ruta insegura.
#
def _is_unsafe_entry(entry: ZipInfo) -> bool:
    encrypted = entry.flag_bits & 1
    absolute = entry.filename.startswith(("/", "\\"))
    escapes_root = ".." in entry.filename.replace("\\", "/").split("/")

    return bool(encrypted or absolute or escapes_root)


#
# Indica si el nombre de archivo corresponde a una hoja de cálculo del XLSX.
#
def _is_worksheet_part(filename: str) -> bool:
    return filename.startswith("xl/worksheets/") and filename.endswith(".xml")


#
# Rechaza partes XML con bytes nulos o declaraciones DOCTYPE y ENTITY.
#
def _check_xml_part(data: bytes) -> None:
    upper = data.upper()

    if b"\x00" in data or b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise TabularInputError("El XLSX contiene declaraciones XML no permitidas.")


#
# Completa una fila CSV hasta el ancho indicado y convierte las cadenas vacías en None.
#
def _padded_row(row: list[str], width: int) -> list[str | None]:
    return [None if value == "" else value for value in row] + [None] * (width - len(row))


#
# Devuelve el texto de una fórmula, o un marcador si no tiene representación textual.
#
def _formula_text(value: object) -> str:
    if isinstance(value, str):
        return value

    text = getattr(value, "text", None)

    return _UNFORMATTED_FORMULA if text is None else text


#
# Convierte una fecha de celda según su formato, conservando como texto las fechas parciales.
#
def _date_value(value: date, number_format: str, raw: str | None, tally: _ConversionTally):
    format_text = re.sub(r'"[^"]*"|\[[^\]]*\]', "", number_format).lower()

    if "d" not in format_text and raw is not None:
        tally.partial_dates += 1
        return LiteralText(raw)

    shows_time = any(token in format_text for token in ("h", "s"))

    if isinstance(value, datetime) and not shows_time:
        return value.date()

    return value


#
# Convierte un número crudo de celda a decimal, conservando los códigos con ceros a la izquierda.
#
def _numeric_value(raw: str, number_format: str, tally: _ConversionTally):
    numeric = Decimal(raw)

    if not _is_zero_padded_code(numeric, number_format):
        return numeric

    tally.formatted_codes += 1

    return LiteralText(str(int(numeric)).zfill(len(number_format)))


#
# Indica si el formato exige ceros a la izquierda en un entero no negativo.
#
def _is_zero_padded_code(numeric: Decimal, number_format: str) -> bool:
    return bool(
        re.fullmatch(r"0{2,}", number_format)
        and numeric == numeric.to_integral_value()
        and numeric >= 0
    )


#
# Convierte una celda del Excel a su valor, contando fórmulas, errores y horarios.
#
def _cell_value(cell, raw: str | None, tally: _ConversionTally):
    value = cell.value

    if cell.data_type == "f":
        tally.formulas += 1
        return _formula_text(value)

    if cell.data_type == "e":
        tally.errors += 1
        return value

    if isinstance(value, (date, datetime)):
        return _date_value(value, cell.number_format, raw, tally)

    if isinstance(value, (time, timedelta)):
        tally.times += 1
        return LiteralText(raw if raw is not None else str(value))

    if raw is not None and not isinstance(value, bool):
        return _numeric_value(raw, cell.number_format, tally)

    return value


#
# Genera las advertencias sobre las conversiones especiales realizadas al leer el Excel.
#
def _conversion_warnings(tally: _ConversionTally) -> list[str]:
    warnings = []

    if tally.formulas:
        warnings.append(
            f"{tally.formulas} fórmulas se conservan como expresiones de texto; no se recalcularon."
        )

    if tally.errors:
        warnings.append(f"{tally.errors} errores Excel se conservan literalmente como texto.")

    if tally.partial_dates:
        warnings.append(
            f"{tally.partial_dates} fechas con formato parcial conservan su serial original como texto."
        )

    if tally.formatted_codes:
        warnings.append(
            f"{tally.formatted_codes} códigos conservan los ceros documentados por su formato Excel."
        )

    if tally.times:
        warnings.append(
            f"{tally.times} horarios o duraciones conservan su valor de origen como texto."
        )

    return warnings


class TableReader:
    #
    # Guarda la configuración con los límites de lectura de archivos tabulares.
    #
    def __init__(self, settings: TabularSettings) -> None:
        self.settings = settings

    #
    # Valida nombre, extensión, tamaño y estructura del archivo y devuelve el nombre seguro.
    #
    def validate(self, filename: str, content: bytes) -> str:
        filename = self.safe_filename(filename)
        suffix = PurePath(filename).suffix.lower()

        if suffix not in {".csv", ".xlsx"}:
            raise TabularInputError("Sólo se admiten archivos CSV, XLSX, PDF y DOCX.")

        if not content:
            raise TabularInputError("El archivo está vacío.")

        if len(content) > self.settings.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño permitido.")

        self._check_structure(suffix, content)

        return filename

    #
    # Lee un archivo CSV o XLSX validado y devuelve sus hojas con los datos.
    #
    def read(self, filename: str, content: bytes) -> LoadedFile:
        filename = self.validate(filename, content)
        file_type = PurePath(filename).suffix.lower()[1:]

        if file_type == "csv":
            return self._read_csv(content)

        try:
            return self._read_excel(content)
        except TabularInputError:
            raise
        except (ValueError, KeyError, OSError, ElementTree.ParseError, InvalidOperation) as error:
            raise TabularInputError("No se pudo leer el contenido del XLSX.") from error

    #
    # Valida el nombre de archivo y devuelve solo su nombre base.
    #
    @staticmethod
    def safe_filename(filename: str) -> str:
        has_valid_name = (
            isinstance(filename, str)
            and bool(filename.strip())
            and not any(ord(char) < 32 for char in filename)
        )

        if not has_valid_name:
            raise TabularInputError("El archivo debe tener un nombre válido.")

        filename = PureWindowsPath(filename).name.strip()

        if not filename or len(filename) > 255:
            raise TabularInputError("El nombre del archivo supera el largo permitido.")

        return filename

    #
    # Verifica que el contenido sea un CSV o un XLSX estructuralmente válido.
    #
    def _check_structure(self, suffix: str, content: bytes) -> None:
        try:
            if suffix == ".csv":
                self._csv_rows(content)
            else:
                self._check_archive(content)
        except (
            BadZipFile,
            KeyError,
            ElementTree.ParseError,
            UnicodeError,
            csv.Error,
            OSError,
            ValueError,
        ) as error:
            if isinstance(error, TabularInputError):
                raise
            raise TabularInputError("El archivo no tiene un formato tabular válido.") from error

    #
    # Lee un CSV y lo devuelve como una única hoja de datos.
    #
    def _read_csv(self, content: bytes) -> LoadedFile:
        rows, source_rows, encoding, delimiter, warnings = self._csv_rows(content)
        width = max(map(len, rows))
        padded = [_padded_row(row, width) for row in rows]
        sheet = SheetData("data", pd.DataFrame(padded, dtype=object), source_rows)

        return LoadedFile("csv", [sheet], warnings, encoding, delimiter)

    #
    # Decodifica y analiza el CSV, y devuelve sus filas, la codificación, el separador y las advertencias.
    #
    def _csv_rows(self, content: bytes) -> tuple[list[list[str]], list[int], str, str, list[str]]:
        warnings = []
        text, encoding = self._decode_csv(content, warnings)

        if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
            raise TabularInputError("El CSV contiene controles binarios y no es texto válido.")

        delimiter = self._csv_delimiter(text, warnings)
        rows, source_rows = self._parse_csv(text, delimiter)

        if not rows or not any(any(not is_empty(value) for value in row) for row in rows):
            raise TabularInputError("El CSV no contiene valores para procesar.")

        if len({len(row) for row in rows if row}) > 1:
            warnings.append("Las filas de distinto ancho se completaron con celdas vacías.")

        return rows, source_rows, encoding, delimiter, warnings

    #
    # Decodifica el CSV con UTF-8 o UTF-16 y, si falla, con CP1252 con una advertencia.
    #
    @staticmethod
    def _decode_csv(content: bytes, warnings: list[str]) -> tuple[str, str]:
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            encoding = "utf-16"
        else:
            encoding = "utf-8-sig"

        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            text = content.decode("cp1252")
            warnings.append("El CSV se leyó con CP1252 porque no contiene UTF-8 válido.")
            return text, "cp1252"

    #
    # Detecta el separador del CSV por análisis o, si falla, por el ancho de las filas.
    #
    @staticmethod
    def _csv_delimiter(text: str, warnings: list[str]) -> str:
        sample = text[:8192]

        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            pass

        multi_column_rows = {}

        for candidate in _CSV_DELIMITERS:
            reader = csv.reader(io.StringIO(sample), delimiter=candidate)
            widths = [len(row) for _, row in zip(range(20), reader) if row]
            multi_column_rows[candidate] = sum(width > 1 for width in widths)

        delimiter = max(multi_column_rows, key=multi_column_rows.get)

        if multi_column_rows[delimiter]:
            warnings.append("El separador del CSV se infirió de filas con anchos distintos.")

        return delimiter

    #
    # Analiza el texto CSV en filas y registra la línea de origen de cada una.
    #
    def _parse_csv(self, text: str, delimiter: str) -> tuple[list[list[str]], list[int]]:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        rows, source_rows = [], []
        width, last_line = 0, 0

        for row in reader:
            source_rows.append(last_line + 1)
            last_line = reader.line_num
            width = max(width, len(row))
            rows.append(row)
            self._check_csv_size(len(rows), width)

        return rows, source_rows

    #
    # Verifica que el CSV no supere los límites de columnas y de celdas.
    #
    def _check_csv_size(self, row_count: int, width: int) -> None:
        if width > self.settings.max_columns:
            raise FileTooLargeError("El CSV supera el límite de columnas permitido.")

        if row_count * max(width, 1) > self.settings.max_cells:
            raise FileTooLargeError("El CSV supera el límite de celdas permitido.")

    #
    # Verifica que el XLSX sea un archivo ZIP válido con hojas dentro de los límites.
    #
    def _check_archive(self, content: bytes) -> None:
        with ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            self._check_entries(entries)

            if not self._count_worksheets(archive, entries):
                raise TabularInputError("El XLSX no contiene hojas de cálculo.")

            paths = self._worksheet_paths(archive)

            if len(paths) > self.settings.max_sheets or not paths:
                raise TabularInputError("El XLSX no declara un conjunto válido de hojas.")

    #
    # Verifica tamaño, duplicados, partes obligatorias, rutas seguras y ausencia de macros en el XLSX.
    #
    def _check_entries(self, entries: list[ZipInfo]) -> None:
        names = [entry.filename for entry in entries]
        expanded_bytes = sum(entry.file_size for entry in entries)

        if len(entries) > 10_000 or expanded_bytes > self.settings.max_expanded_bytes:
            raise FileTooLargeError("El XLSX supera el tamaño descomprimido permitido.")

        if len(names) != len(set(names)):
            raise TabularInputError("El XLSX contiene partes duplicadas.")

        if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(names):
            raise TabularInputError("El archivo no es un libro XLSX válido.")

        if any(_is_unsafe_entry(entry) for entry in entries):
            raise TabularInputError("El XLSX contiene partes cifradas o rutas no válidas.")

        if any(name.lower().endswith("vbaproject.bin") for name in names):
            raise TabularInputError("Sólo se admiten XLSX sin macros.")

    #
    # Cuenta las hojas del XLSX y verifica sus límites de hojas y de celdas.
    #
    def _count_worksheets(self, archive: ZipFile, entries: list[ZipInfo]) -> int:
        total_cells, sheet_count = 0, 0

        for entry in entries:
            if not entry.filename.endswith((".xml", ".rels")):
                continue

            _check_xml_part(archive.read(entry))

            if not _is_worksheet_part(entry.filename):
                continue

            sheet_count += 1

            if sheet_count > self.settings.max_sheets:
                raise FileTooLargeError("El XLSX supera el límite de hojas permitido.")

            max_row, max_column, _, _, _ = self._worksheet_layout(archive, entry.filename)
            sheet_cells = max_row * max_column

            if total_cells + sheet_cells > self.settings.max_cells:
                raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")

            total_cells += sheet_cells

        return sheet_count

    #
    # Lee un XLSX y devuelve sus hojas con los datos.
    #
    def _read_excel(self, content: bytes) -> LoadedFile:
        workbook = load_workbook(
            io.BytesIO(content), data_only=False, read_only=True, keep_links=False
        )

        try:
            with ZipFile(io.BytesIO(content)) as archive:
                paths = self._worksheet_paths(archive)

            sheets = [
                self._read_sheet(content, worksheet, paths[worksheet.title])
                for worksheet in workbook.worksheets
            ]

            return LoadedFile("xlsx", sheets)
        finally:
            workbook.close()

    #
    # Lee una hoja del XLSX con sus valores, celdas combinadas, tablas declaradas y advertencias.
    #
    def _read_sheet(self, content: bytes, worksheet, path: str) -> SheetData:
        with ZipFile(io.BytesIO(content)) as archive:
            max_row, max_column, raw_numbers, merged_ranges, table_parts = self._worksheet_layout(
                archive, path, include_values=True
            )
            table_ranges, table_warnings = self._declared_tables(
                archive, path, table_parts, max_row, max_column
            )

        tally = _ConversionTally()
        rows = []

        for source_row in worksheet.iter_rows(max_row=max(max_row, 1), max_col=max(max_column, 1)):
            rows.append(
                [
                    _cell_value(cell, raw_numbers.get(getattr(cell, "coordinate", "")), tally)
                    for cell in source_row
                ]
            )

        warnings = list(table_warnings) + _conversion_warnings(tally)

        return SheetData(
            worksheet.title,
            pd.DataFrame(rows, dtype=object),
            list(range(1, len(rows) + 1)),
            merged_ranges,
            table_ranges,
            warnings,
            bool(table_parts),
        )

    #
    # Devuelve el mapa de nombre de hoja a ruta de su parte dentro del XLSX.
    #
    def _worksheet_paths(self, archive: ZipFile) -> dict[str, str]:
        targets = self._worksheet_targets(archive)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        paths = {}

        for element in workbook.iter():
            if _local_name(element.tag) != "sheet":
                continue

            relation_id = next(_attribute_ids(element), None)

            if relation_id in targets:
                paths[element.attrib["name"]] = targets[relation_id]

        return paths

    #
    # Devuelve el mapa de identificador de relación a ruta de hoja, validando que sean internas.
    #
    @staticmethod
    def _worksheet_targets(archive: ZipFile) -> dict[str, str]:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {}

        for relation in relationships:
            if not relation.attrib.get("Type", "").endswith("/worksheet"):
                continue

            path = _part_path("xl", relation.attrib.get("Target", ""))
            external = relation.attrib.get("TargetMode") == "External"

            if external or not path.startswith("xl/worksheets/"):
                raise TabularInputError("El XLSX contiene una hoja externa o una ruta no válida.")

            if path not in archive.namelist():
                raise TabularInputError("El XLSX declara una hoja que no existe.")

            targets[relation.attrib["Id"]] = path

        return targets

    #
    # Recorre una hoja para obtener sus dimensiones, valores crudos, celdas combinadas y tablas.
    #
    def _worksheet_layout(self, archive: ZipFile, path: str, include_values: bool = False) -> tuple:
        layout = _SheetLayout()

        with archive.open(path) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                tag = _local_name(element.tag)

                if tag == "c":
                    self._register_cell(element, layout, include_values)
                elif tag == "mergeCell":
                    layout.merged_ranges.append(range_boundaries(element.attrib["ref"]))
                elif tag == "tablePart":
                    layout.table_parts.append(_table_part_id(element))

                if tag in {"c", "mergeCell", "tablePart", "row"}:
                    element.clear()

        return (
            layout.max_row,
            layout.max_column,
            layout.values,
            layout.merged_ranges,
            layout.table_parts,
        )

    #
    # Registra una celda con contenido en el diseño de la hoja y valida el límite de celdas.
    #
    def _register_cell(self, element, layout: _SheetLayout, include_values: bool) -> None:
        content = [child for child in element.iter() if _local_name(child.tag) in {"v", "t", "f"}]
        has_content = any(
            child.text not in {None, ""} or _local_name(child.tag) == "f" for child in content
        )

        if not has_content:
            return

        row, column = coordinate_to_tuple(element.attrib["r"])
        layout.max_row = max(layout.max_row, row)
        layout.max_column = max(layout.max_column, column)

        if layout.max_row * layout.max_column > self.settings.max_cells:
            raise FileTooLargeError("El XLSX supera el límite de celdas permitido.")

        if not include_values or element.attrib.get("t", "n") != "n":
            return

        raw = next((child.text for child in content if _local_name(child.tag) == "v"), None)

        if raw is not None:
            layout.values[element.attrib["r"]] = raw

    #
    # Obtiene los rangos de las tablas declaradas en una hoja junto con sus advertencias.
    #
    def _declared_tables(
        self,
        archive: ZipFile,
        sheet_path: str,
        table_parts: list[str],
        max_row: int,
        max_column: int,
    ) -> tuple[list[tuple], list[str]]:
        if not table_parts:
            return [], []

        targets = self._table_targets(archive, sheet_path, table_parts)
        ranges, warnings = [], []

        for part in table_parts:
            table = ElementTree.fromstring(archive.read(targets[part]))
            bounds = self._declared_range(table, max_row, max_column, warnings)

            if bounds is not None:
                ranges.append(bounds)

        return ranges, warnings

    #
    # Devuelve el mapa de identificador de relación a ruta de las tablas declaradas en una hoja.
    #
    @staticmethod
    def _table_targets(archive: ZipFile, sheet_path: str, table_parts: list[str]) -> dict:
        base = posixpath.dirname(sheet_path)
        relationship_path = posixpath.join(base, "_rels", posixpath.basename(sheet_path) + ".rels")
        relationships = ElementTree.fromstring(archive.read(relationship_path))
        targets = {}

        for relation in relationships:
            target = relation.attrib.get("Target", "")

            if relation.attrib.get("Id") not in table_parts:
                continue

            if relation.attrib.get("TargetMode") == "External":
                raise TabularInputError("El XLSX declara una tabla externa.")

            targets[relation.attrib["Id"]] = _part_path(base, target)

        return targets

    #
    # Calcula el rango de una tabla declarada, excluyendo la fila de totales.
    #
    @staticmethod
    def _declared_range(table, max_row: int, max_column: int, warnings: list[str]) -> tuple | None:
        name = table.attrib.get("displayName", "tabla")

        if table.attrib.get("headerRowCount", "1") == "0":
            warnings.append(f"La tabla {name} no declara encabezados y se conserva en el original.")
            return None

        bounds = list(range_boundaries(table.attrib["ref"]))

        if None in bounds:
            raise TabularInputError(
                f"La tabla {name} declara un rango sin límites de filas o columnas."
            )

        totals = int(table.attrib.get("totalsRowCount", "0"))

        if totals or table.attrib.get("totalsRowShown", "0") in {"1", "true"}:
            bounds[3] -= totals or 1
            warnings.append(f"La fila de totales declarada en {name} se conserva en el original.")

        bounds[2], bounds[3] = min(bounds[2], max_column), min(bounds[3], max_row)

        if bounds[0] <= bounds[2] and bounds[1] < bounds[3]:
            return tuple(bounds)

        return None
