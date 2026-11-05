import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import groupby
from pathlib import Path
from statistics import mean, mode
from typing import Any

import pypdfium2

from aquai.application.contracts.documents import PreparedElement
from aquai.domain.enums.documentContent import DocumentElementKind
from aquai.infrastructure.processing.documents.pageText import (
    Glyph,
    Segment,
    aligned_rows,
    glyphs_in_box,
    is_rotated,
    is_value_cell,
    is_value_row,
    line_segments,
    page_glyphs,
    render_box,
    upright,
    visual_lines,
)
from aquai.infrastructure.processing.documents.structure import table_lines
from aquai.infrastructure.processing.documents.symbols import PageSymbols

_NOT_ALPHANUMERIC = re.compile(r"[\W_]+")
_NUMBER = re.compile(r"\d+(?:[.,·]\d+)*")
_MIN_LINE_CHARS = 3
_MIN_SHARED_CHARS = 0.9
_MIN_COVERED_CHARS = 0.9
_MAX_SHARED_WHEN_MISSING = 0.5
_MIN_COLLAPSED_LINES = 3
_MIN_COLLAPSED_SEGMENTS = 3
_MAX_COLLAPSED_WORDS = 3
_MAX_BRIEF_SEGMENT = 12
_MAX_HEADER_LINES = 4
_MIN_SPANNED_CELLS = 3
_WIDE_LINE_SHARE = 0.7
_MIN_EXTRA_IMAGE_LINES = 1.5
_MIN_READ_CHARS = 0.8
_TABLE_OCR_SCALE = 4
_WHOLE_IMAGE = [float("-inf"), float("-inf"), float("inf"), float("inf")]

ReadBoxes = Callable[[Any], list[Glyph]]


@dataclass
class _GridRow:
    letters: Counter
    numbers: list[str]


@dataclass(frozen=True)
class TableCheck:
    lines: list[list[Segment]]
    broken_lines: int
    collapsed_rows: int
    broken_rows: int = 0
    covered_share: float = 0.0
    grid_columns: int = 0
    header_lines: int = 0
    missing_lines: list[list[str]] = field(default_factory=list)
    verifiable: bool = True
    sideways: bool = False

    #
    # Indica si no se puede confiar en la grilla: sus filas contradicen lo que la página muestra
    # en una misma línea, o la tabla está de costado y Docling la leyó sin enderezar.
    #
    @property
    def misaligned(self) -> bool:
        contradicted = self.broken_lines > 0 or self.collapsed_rows > 0
        return self.verifiable and (contradicted or self.sideways)


#
# Contrasta cada tabla con su página: si la grilla la contradice y la tabla es de valores breves,
# la reemplaza por las líneas de la página, que no pueden cruzar filas; si no, deja la duda.
#
def verify_tables(
    path: Path,
    elements: list[PreparedElement],
    symbols: PageSymbols,
    read_boxes: ReadBoxes,
    scanned_pages: set[int],
) -> dict[str, list[dict[str, Any]]]:
    review: dict[str, list[dict[str, Any]]] = {
        "rebuilt": [],
        "read_by_ocr": [],
        "uncertain": [],
        "unverifiable": [],
    }
    tables = [
        element
        for element in elements
        if element.kind == DocumentElementKind.TABLE and element.cells and element.locations
    ]
    if not tables:
        return review

    pdf = pypdfium2.PdfDocument(str(path))
    glyphs: dict[int, list[Glyph]] = {}
    try:
        for element in tables:
            scanned = element.locations[0]["page"] in scanned_pages
            _review_table(pdf, glyphs, symbols, element, review, read_boxes if scanned else None)
    finally:
        pdf.close()

    return review


#
# Contrasta una tabla y registra el resultado, con el elemento al que corresponde.
#
def _review_table(
    pdf,
    glyphs: dict[int, list[Glyph]],
    symbols: PageSymbols,
    element: PreparedElement,
    review: dict[str, list[dict[str, Any]]],
    read_boxes: ReadBoxes | None,
) -> None:
    location = element.locations[0]
    number = location["page"]
    entry = {"page": number, "label": element.label, "element": element.position}
    page = pdf[number - 1]
    if number not in glyphs:
        glyphs[number] = symbols.applied(number, page_glyphs(page))

    check = check_table(glyphs[number], location["bbox"], element.cells, element.header_rows)
    if read_boxes and not check.verifiable:
        _review_against_image(page, element, entry, review, read_boxes)
    elif not check.verifiable or len(element.locations) > 1:
        review["unverifiable"].append(entry)
    elif read_boxes and _read_from_image(page, element, check, read_boxes):
        review["read_by_ocr"].append(entry)
    elif check.misaligned and _is_rebuildable(check):
        _rebuild(element, check)
        review["rebuilt"].append(entry)
    elif check.misaligned:
        _add_missing_lines(element, check)
        review["uncertain"].append(entry)


#
# Contrasta una tabla de una página sin capa de texto con lo que se lee en su imagen. La grilla
# también salió de un OCR y ninguna lectura vale más que la otra: sólo se registra el resultado.
#
def _review_against_image(
    page,
    element: PreparedElement,
    entry: dict[str, Any],
    review: dict[str, list[dict[str, Any]]],
    read_boxes: ReadBoxes,
) -> None:
    image = render_box(page, element.locations[0]["bbox"], _TABLE_OCR_SCALE)
    check = check_table(read_boxes(image), _WHOLE_IMAGE, element.cells, element.header_rows)
    if not check.verifiable or check.covered_share < _MIN_READ_CHARS:
        review["unverifiable"].append(entry)
    elif check.misaligned:
        review["uncertain"].append(entry)


#
# Lee una tabla de una página escaneada en la imagen, sólo si ésta muestra bastantes más líneas
# que la capa de texto: el OCR que la escribió puede haber dejado filas enteras afuera.
#
def _read_from_image(
    page, element: PreparedElement, check: TableCheck, read_boxes: ReadBoxes
) -> bool:
    image = render_box(page, element.locations[0]["bbox"], _TABLE_OCR_SCALE)
    lines = [line_segments(line) for line in visual_lines(read_boxes(image))]
    enough = len(lines) >= max(_MIN_EXTRA_IMAGE_LINES * len(check.lines), 1)
    if not enough or _covered_share(element.cells, lines) < _MIN_READ_CHARS:
        return False

    _rebuild(element, replace(check, lines=lines))
    return True


#
# Reemplaza las celdas de una tabla por las líneas de su página, conservando sus notas.
#
def _rebuild(element: PreparedElement, check: TableCheck) -> None:
    previous = table_lines(element.cells)
    notes = (element.content or "").split("\n")[len(previous) :]
    element.cells = aligned_rows(check.lines)
    element.header_rows = check.header_lines
    element.content = "\n".join([*table_lines(element.cells), *notes])


#
# Agrega a una tabla dudosa las líneas de su página que la grilla perdió, para no perder texto.
#
def _add_missing_lines(element: PreparedElement, check: TableCheck) -> None:
    previous = table_lines(element.cells)
    notes = (element.content or "").split("\n")[len(previous) :]
    width = max(len(row) for row in element.cells)
    missing = [[*line, *[""] * (width - len(line))] for line in check.missing_lines]
    element.cells = [*element.cells, *missing]
    element.content = "\n".join([*table_lines(element.cells), *notes])


#
# Indica si las líneas de la página pueden reemplazar a la grilla: una fila entera debe contradecir
# a la grilla, porque una celda partida en varias líneas no lo hace; además deben cubrir su texto,
# no perder columnas y tener celdas breves, porque en texto corrido se leerían peor.
#
def _is_rebuildable(check: TableCheck) -> bool:
    lengths = [len(segment.text) for line in check.lines for segment in line]
    contradicted = check.broken_rows > 0 or check.collapsed_rows > 0 or check.sideways
    covered = check.covered_share >= _MIN_COVERED_CHARS
    brief = mean(lengths) <= _MAX_BRIEF_SEGMENT
    return contradicted and covered and brief and _keeps_columns(check)


#
# Indica si la lectura por líneas conserva las columnas con contenido que tiene la grilla.
#
def _keeps_columns(check: TableCheck) -> bool:
    return len(aligned_rows(check.lines)[0]) >= check.grid_columns


#
# Devuelve la parte del texto de la grilla que aparece en las líneas; las celdas combinadas se
# repiten en la grilla y se cuentan una vez.
#
def _covered_share(cells: list[list[str]], lines: list[list[Segment]]) -> float:
    grid = sum((Counter(_compact("".join(dict.fromkeys(row)))) for row in cells), Counter())
    page = sum((_line_letters(line) for line in lines), Counter())
    return (grid & page).total() / max(grid.total(), 1)


#
# Cuenta las columnas de la grilla que tienen algún contenido.
#
def _filled_columns(cells: list[list[str]]) -> int:
    return sum(any(cell.strip() for cell in column) for column in zip(*cells, strict=False))


#
# Compara las filas de una tabla con las líneas visuales de su región en la página original, con
# el texto enderezado si la tabla está girada. Las líneas sueltas antes de la primera fila son
# títulos; una fila de encabezado puede ocupar varias.
#
def check_table(
    glyphs: list[Glyph], box: list[float], cells: list[list[str]], header_rows: int = 0
) -> TableCheck:
    inside = glyphs_in_box(glyphs, box)
    if not inside:
        return TableCheck([], 0, 0, verifiable=False)

    sideways = is_rotated(inside)
    readable = upright(inside) if sideways else inside
    lines = [line_segments(line) for line in visual_lines(readable)]
    rows = [_grid_row(row, position < header_rows) for position, row in enumerate(cells)]
    matches = [_matching_row(segments, rows) for segments in lines]
    unmatched = _unmatched_lines(lines, matches)
    usual_cells = mode(len(line) for line in lines)
    return TableCheck(
        lines,
        len(unmatched),
        _collapsed_rows(lines, matches, header_rows),
        sum(len(line) >= usual_cells for line in unmatched),
        _covered_share(cells, lines),
        _filled_columns(cells),
        _leading_lines(lines, [] if sideways else matches, header_rows),
        _missing_lines(unmatched, rows),
        sideways=sideways,
    )


#
# Prepara una fila de la grilla para compararla con las líneas de la página.
#
def _grid_row(cells: list[str], is_header: bool) -> _GridRow:
    return _GridRow(_row_letters(cells, is_header), _numbers(" ".join(cells)))


#
# Devuelve los números de un texto, con un solo separador decimal.
#
def _numbers(text: str) -> list[str]:
    return [number.replace(",", ".").replace("·", ".") for number in _NUMBER.findall(text)]


#
# Devuelve las letras de una fila. En un encabezado, varias celdas seguidas iguales son una celda
# combinada y cuentan una vez: repetidas, taparían que a la grilla le falta parte de la línea.
#
def _row_letters(row: list[str], is_header: bool) -> Counter:
    cells = []
    for cell, group in groupby(row):
        run = len(list(group))
        cells += [cell] if is_header and run >= _MIN_SPANNED_CELLS else [cell] * run

    return Counter(_compact("".join(cells)))


#
# Devuelve las líneas sin fila que cuentan: las muy breves no dicen nada y las de un solo texto
# antes de la primera fila son títulos, si es que alguna línea encontró su fila.
#
def _unmatched_lines(
    lines: list[list[Segment]], matches: list[int | None]
) -> list[list[Segment]]:
    first = next((position for position, row in enumerate(matches) if row is not None), None)
    unmatched = []
    for position, (segments, row) in enumerate(zip(lines, matches, strict=True)):
        is_title = first is not None and position < first and len(segments) == 1
        if row is None and not is_title and _line_letters(segments).total() >= _MIN_LINE_CHARS:
            unmatched.append(segments)

    return unmatched


#
# Devuelve las letras y dígitos de una línea, para comparar sin depender de espacios ni signos.
#
def _line_letters(segments: list[Segment]) -> Counter:
    return Counter(_compact("".join(segment.text for segment in segments)))


#
# Devuelve las líneas que la grilla no tiene: ninguna fila conserva buena parte de su texto, a
# diferencia de una línea cruzada, cuyos valores siguen repartidos entre dos filas.
#
def _missing_lines(unmatched: list[list[Segment]], rows: list[_GridRow]) -> list[list[str]]:
    missing = []
    for segments in unmatched:
        letters = _line_letters(segments)
        shared = max(((letters & row.letters).total() for row in rows), default=0)
        if shared < _MAX_SHARED_WHEN_MISSING * letters.total():
            missing.append([segment.text for segment in segments])

    return missing


#
# Devuelve la primera fila que contiene el texto de la línea y se lo descuenta, para que dos
# líneas iguales no se atribuyan a la misma; devuelve None si ninguna alcanza. Se comparan
# caracteres y no cadenas porque subíndices y superíndices se intercalan distinto, pero los
# dígitos deben estar y los valores, en orden: un nombre largo no puede tapar un valor ajeno, ni
# dos valores intercambiados pasar por iguales.
#
def _matching_row(segments: list[Segment], rows: list[_GridRow]) -> int | None:
    line = _line_letters(segments)
    if line.total() < _MIN_LINE_CHARS:
        return None

    digits = Counter({char: count for char, count in line.items() if char.isdigit()})
    values = [number for item in segments if is_value_cell(item.text) for number in _numbers(item.text)]
    for position, row in enumerate(rows):
        similar = (line & row.letters).total() >= _MIN_SHARED_CHARS * line.total()
        if similar and not digits - row.letters and _appear_in_order(values, row.numbers):
            row.letters = row.letters - line
            return position

    return None


#
# Indica si los valores de una línea aparecen en la fila en ese orden; una línea puede tener sólo
# parte de una fila, si sus celdas ocupan varias líneas.
#
def _appear_in_order(values: list[str], numbers: list[str]) -> bool:
    remaining = iter(numbers)
    return all(value in remaining for value in values)


#
# Cuenta las filas de datos que absorbieron varias líneas que en la página son filas breves.
#
def _collapsed_rows(
    lines: list[list[Segment]], matches: list[int | None], header_rows: int
) -> int:
    by_row: dict[int, list[list[Segment]]] = {}
    for segments, row in zip(lines, matches, strict=True):
        if row is not None and row >= header_rows:
            by_row.setdefault(row, []).append(segments)

    return sum(
        _is_collapsed(group) or _absorbs_the_table(group, lines) for group in by_row.values()
    )


#
# Devuelve cuántas líneas de la página preceden a la primera fila de datos: la primera línea con
# valores; en una tabla sin números, la primera que la grilla no tiene como encabezado o, sin
# esa referencia, la segunda línea ancha, porque la primera es la de los títulos de columna.
#
def _leading_lines(
    lines: list[list[Segment]], matches: list[int | None], header_rows: int
) -> int:
    for position, line in enumerate(lines):
        if is_value_row([segment.text for segment in line]):
            return min(position, _MAX_HEADER_LINES)

    for position, row in enumerate(matches):
        if row is not None and row >= header_rows:
            return min(position, _MAX_HEADER_LINES)

    widest = max(len(line) for line in lines)
    wide = [
        position
        for position, line in enumerate(lines)
        if len(line) >= _WIDE_LINE_SHARE * widest
    ]
    return min(wide[1], _MAX_HEADER_LINES) if len(wide) > 1 else 0


#
# Indica si una sola fila se quedó con media tabla: sus celdas ya no dicen qué va con qué.
#
def _absorbs_the_table(group: list[list[Segment]], lines: list[list[Segment]]) -> bool:
    return len(group) > _MIN_COLLAPSED_LINES and 2 * len(group) >= len(lines)


#
# Indica si una fila absorbió varias líneas que en la página son filas de celdas breves.
#
def _is_collapsed(lines: list[list[Segment]]) -> bool:
    tabular = [
        line
        for line in lines
        if len(line) >= _MIN_COLLAPSED_SEGMENTS
        and all(len(segment.text.split()) <= _MAX_COLLAPSED_WORDS for segment in line)
    ]
    return len(tabular) >= _MIN_COLLAPSED_LINES


#
# Reduce un texto a sus letras y dígitos, para comparar sin depender de espacios ni signos.
#
def _compact(text: str) -> str:
    return _NOT_ALPHANUMERIC.sub("", text).casefold()
