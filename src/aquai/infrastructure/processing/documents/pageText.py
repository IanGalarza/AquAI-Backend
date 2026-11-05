import ctypes
import re
from dataclasses import dataclass, replace
from math import degrees
from statistics import median, mode
from typing import Any

import numpy
import pypdfium2.raw as pdfium_raw

_FONT_NAME_BYTES = 256
_QUARTER_TURN = 90
_VALUE_CELL = re.compile(r"^[(<>≤≥~≈±+−–-]?\s*\.?\d")
_LINE_TOLERANCE = 0.55
_SEGMENT_GAP = 0.9
_WORD_GAP = 0.05
_USUAL_GAPS = 2
_SCRIPT_SHIFT = 0.2
_COLUMN_NOISE = 0.1
_MIN_BOX_SIDE = 1
_NO_TEXT = "\ufffd"
_HIGH_SURROGATES = range(0xD800, 0xDC00)
_LOW_SURROGATES = range(0xDC00, 0xE000)


@dataclass(frozen=True)
class Glyph:
    text: str
    left: float
    top: float
    right: float
    bottom: float
    font: str
    turn: int = 0

    #
    # Devuelve el centro vertical del carácter.
    #
    @property
    def middle(self) -> float:
        return (self.top + self.bottom) / 2

    #
    # Devuelve la altura del carácter.
    #
    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True)
class Segment:
    text: str
    left: float
    right: float


#
# Lee los caracteres de una página con su fuente, su giro y su recuadro, tal como la página se
# ve: con origen arriba a la izquierda del área visible, que puede no ser el origen del PDF, y
# con el giro que la página declara, el mismo en que Docling ubica sus elementos.
#
def page_glyphs(page) -> list[Glyph]:
    page_box = page.get_bbox()
    rotation = page.get_rotation()
    textpage = page.get_textpage()
    buffer = (ctypes.c_char * _FONT_NAME_BYTES)()
    flags = ctypes.c_int()
    glyphs = []
    count = textpage.count_chars()
    codes = [pdfium_raw.FPDFText_GetUnicode(textpage.raw, index) for index in range(count)]
    for index, code in enumerate(codes):
        if code in _LOW_SURROGATES and index and codes[index - 1] in _HIGH_SURROGATES:
            continue

        box = textpage.get_charbox(index, loose=True)
        left, top, right, bottom = _visible_box(box, page_box, rotation)
        size = pdfium_raw.FPDFText_GetFontInfo(
            textpage.raw, index, buffer, _FONT_NAME_BYTES, ctypes.byref(flags)
        )
        glyphs.append(
            Glyph(
                text=_glyph_text(code, codes[index + 1] if index + 1 < count else 0),
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                font=buffer.value.decode(errors="replace") if size > 0 else "",
                turn=_visible_turn(pdfium_raw.FPDFText_GetCharAngle(textpage.raw, index), rotation),
            )
        )

    return glyphs


#
# Lleva un recuadro del PDF (izquierda, abajo, derecha, arriba) a la página tal como se ve y lo
# devuelve como izquierda, arriba, derecha y abajo.
#
def _visible_box(
    box: tuple[float, float, float, float],
    page_box: tuple[float, float, float, float],
    rotation: int,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = box
    page_left, page_bottom, page_right, page_top = page_box
    width, height = page_right - page_left, page_top - page_bottom
    x0, x1 = left - page_left, right - page_left
    y0, y1 = bottom - page_bottom, top - page_bottom
    if rotation == 90:
        return y0, x0, y1, x1
    if rotation == 180:
        return width - x1, y0, width - x0, y1
    if rotation == 270:
        return height - y1, width - x1, height - y0, width - x0

    return x0, height - y1, x1, height - y0


#
# Devuelve el giro de un carácter en la página tal como se ve, en cuartos de vuelta en sentido
# horario; una inclinación leve, como la de un escaneo, no cuenta.
#
def _visible_turn(angle: float, rotation: int) -> int:
    quarters = round((degrees(max(angle, 0)) + rotation) / _QUARTER_TURN)
    return quarters % 4 * _QUARTER_TURN


#
# Devuelve el texto de un código del PDF, que llega en UTF-16: un par sustituto es un solo
# carácter. El cero y una mitad suelta no son texto válido y PostgreSQL no los aceptaría.
#
def _glyph_text(code: int, following: int = 0) -> str:
    if code in _HIGH_SURROGATES and following in _LOW_SURROGATES:
        return chr(0x10000 + ((code - 0xD800) << 10) + (following - 0xDC00))

    is_broken = code == 0 or code in _HIGH_SURROGATES or code in _LOW_SURROGATES
    return _NO_TEXT if is_broken else chr(code)


#
# Renderiza una región de la página, con origen arriba a la izquierda, como imagen para leerla.
# Lo que cae fuera de la página se acota a su borde: un PDF puede tener texto fuera del área.
# Si el texto está girado, la imagen se endereza para que se pueda leer.
#
def render_box(page, box: list[float], scale: float, turn: int = 0) -> Any:
    width, height = page.get_size()
    left = min(max(box[0], 0), width - _MIN_BOX_SIDE)
    top = min(max(box[1], 0), height - _MIN_BOX_SIDE)
    right = max(min(box[2], width), left + _MIN_BOX_SIDE)
    bottom = max(min(box[3], height), top + _MIN_BOX_SIDE)
    bitmap = page.render(scale=scale, crop=(left, height - bottom, width - right, top))
    image = numpy.array(bitmap.to_pil().convert("RGB"))
    return numpy.ascontiguousarray(numpy.rot90(image, turn // _QUARTER_TURN))


#
# Devuelve los caracteres visibles cuyo centro cae dentro de un recuadro.
#
def glyphs_in_box(glyphs: list[Glyph], box: list[float]) -> list[Glyph]:
    left, top, right, bottom = box
    return [
        glyph
        for glyph in glyphs
        if glyph.text.strip()
        and left - 1 <= (glyph.left + glyph.right) / 2 <= right + 1
        and top - 1 <= glyph.middle <= bottom + 1
    ]


#
# Indica si la mayoría del texto está girado, caso en el que las líneas visuales no son filas.
#
def is_rotated(glyphs: list[Glyph]) -> bool:
    return sum(bool(glyph.turn) for glyph in glyphs) > len(glyphs) / 2


#
# Endereza un texto girado, como quien gira la hoja para leerlo: sus líneas pasan a ser filas.
#
def upright(glyphs: list[Glyph]) -> list[Glyph]:
    turn = mode(glyph.turn for glyph in glyphs if glyph.turn)
    return [_turned(glyph, turn) for glyph in glyphs]


#
# Gira el recuadro de un carácter. Sólo importa la posición relativa, no el origen.
#
def _turned(glyph: Glyph, turn: int) -> Glyph:
    left, top, right, bottom = glyph.left, glyph.top, glyph.right, glyph.bottom
    if turn == 270:
        return replace(glyph, left=-bottom, top=left, right=-top, bottom=right, turn=0)
    if turn == 90:
        return replace(glyph, left=top, top=-right, right=bottom, bottom=-left, turn=0)

    return replace(glyph, left=-right, top=-bottom, right=-left, bottom=-top, turn=0)


#
# Agrupa caracteres en líneas visuales, de arriba hacia abajo.
#
def visual_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    lines: list[list[Glyph]] = []
    for glyph in sorted(glyphs, key=lambda item: item.middle):
        line = lines[-1] if lines else None
        if line and abs(glyph.middle - _line_middle(line)) < _LINE_TOLERANCE * _line_height(line):
            line.append(glyph)
        else:
            lines.append([glyph])

    return [sorted(line, key=lambda item: item.left) for line in lines]


#
# Devuelve el centro vertical de una línea.
#
def _line_middle(line: list[Glyph]) -> float:
    return median(glyph.middle for glyph in line)


#
# Devuelve la altura típica de los caracteres de una línea.
#
def _line_height(line: list[Glyph]) -> float:
    return max(median(glyph.height for glyph in line), 1.0)


#
# Separa una línea en los textos que la página muestra apartados entre sí, como celdas. Los
# espacios se deducen de la distancia, porque los del PDF pueden caer en medio de una palabra al
# ordenar los caracteres, y se comparan con la distancia habitual de la línea, que en una capa
# de OCR no es cero. Un salto de altura también separa, para que 10⁵ no se lea 105.
#
def line_segments(line: list[Glyph]) -> list[Segment]:
    height = _line_height(line)
    gaps = [max(after.left - before.right, 0) for before, after in zip(line, line[1:])]
    usual = median(gaps) if gaps else 0
    segments: list[Segment] = []
    previous = None
    for glyph in line:
        gap = glyph.left - previous.right if previous else float("inf")
        if gap > _SEGMENT_GAP * height:
            segments.append(Segment(glyph.text, glyph.left, glyph.right))
        else:
            shifted = abs(glyph.middle - previous.middle) > _SCRIPT_SHIFT * height
            space = " " if gap > _WORD_GAP * height + _USUAL_GAPS * usual or shifted else ""
            last = segments[-1]
            segments[-1] = Segment(
                last.text + space + glyph.text, last.left, max(last.right, glyph.right)
            )

        previous = glyph

    return segments


#
# Ordena las líneas de una tabla en columnas, dejando vacías las celdas que cada línea no ocupa:
# en muchas tablas la posición del valor es el dato, como una marca bajo una categoría.
#
def aligned_rows(lines: list[list[Segment]]) -> list[list[str]]:
    columns = _columns(lines)
    rows = []
    for line in lines:
        row = [""] * len(columns)
        for segment in line:
            position = _column_of(segment, columns)
            row[position] = f"{row[position]} {segment.text}".strip()
        rows.append(row)

    return _joined_wraps(rows)


#
# Une a su fila las líneas de un solo texto que continúan una celda: el nombre de una especie
# puede ocupar dos líneas y sus valores una sola, entre medio.
#
def _joined_wraps(rows: list[list[str]]) -> list[list[str]]:
    received: set[int] = set()
    for position, row in enumerate(rows):
        filled = [column for column, cell in enumerate(row) if cell]
        if len(filled) != 1:
            continue

        column = filled[0]
        above = rows[position - 1] if position else None
        below = rows[position + 1] if position + 1 < len(rows) else None
        if position - 1 in received:
            above[column] = f"{above[column]} {row[column]}"
        elif below and not below[column] and is_value_row(below):
            below[column] = row[column]
            received.add(position + 1)
        elif above and not above[column] and is_value_row(above):
            above[column] = row[column]
        else:
            continue

        row[column] = ""

    return [row for row in rows if any(row)]


#
# Indica si una fila tiene valores en la mitad o más de sus celdas, y no puede ser un encabezado.
#
def is_value_row(cells: list[str]) -> bool:
    filled = [cell.strip() for cell in cells if cell.strip()]
    values = sum(is_value_cell(cell) for cell in filled)
    return len(filled) >= 2 and 2 * values >= len(filled)


#
# Indica si un texto es un valor y no un rótulo: empieza con un número, quizá con su signo.
#
def is_value_cell(text: str) -> bool:
    return bool(_VALUE_CELL.match(text.strip()))


#
# Ubica las columnas: son los tramos horizontales que ocupan varias líneas con más de una celda.
# Lo que cruza pocas líneas, como el título de un grupo de columnas, no llega a unirlas.
#
def _columns(lines: list[list[Segment]]) -> list[tuple[float, float]]:
    every = [segment for line in lines for segment in line]
    segments = [segment for line in lines if len(line) > 1 for segment in line] or every
    start = int(min(segment.left for segment in segments))
    coverage = [0] * (int(max(segment.right for segment in segments)) - start + 2)
    for segment in segments:
        for point in range(int(segment.left) - start, int(segment.right) - start + 1):
            coverage[point] += 1

    limit = max(1, int(len(lines) * _COLUMN_NOISE))
    columns: list[tuple[float, float]] = []
    opened = None
    for point, count in enumerate(coverage):
        if count > limit and opened is None:
            opened = point
        elif count <= limit and opened is not None:
            columns.append((start + opened, start + point))
            opened = None

    return columns or [(start, start + len(coverage))]


#
# Devuelve la columna que más se solapa con un texto, o la más cercana si no toca ninguna.
#
def _column_of(segment: Segment, columns: list[tuple[float, float]]) -> int:
    overlaps = [min(segment.right, right) - max(segment.left, left) for left, right in columns]
    return overlaps.index(max(overlaps))
