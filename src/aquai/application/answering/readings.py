import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from aquai.application.Rules.evidence import ABBREVIATION_END, SENTENCE_END
from aquai.application.Rules.units import UNIT_WORD_STARTS
from aquai.application.units import comparable_unit, concentration_basis

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_POWER = rf"(?:[{_SUPERSCRIPT_DIGITS}]+|\d{{1,2}})(?!\d|[.,]\d)"
_NUMBER = re.compile(
    rf"(?P<scientific>(?<![\w.,/-])-?\d+(?:[.,]\d+)?\s?[×x·]\s?10\s?(?:\^\s?)?(?:[-⁻]\s?)?{_POWER}"
    rf"|(?<![\w.,/-])10(?:\^\s?-?\d{{1,2}}|⁻?[{_SUPERSCRIPT_DIGITS}]+)(?!\d)"
    r"|(?<![\w.,/-])10\s(?:-\s?)?\d{1,2}(?=\s?(?:cells?|c[eé]l\w*|ind\w*|UFC|CFU|NMP|MPN)\b))"
    r"|(?<![\w.,/-])-?\d+(?:[.,   ]\d{3})*(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?!\d)"
    r"|(?<![\w.,])\d+(?:[.,]\d+)?(?=[-–/]\d)"
    r"|(?<=[-–/])\d+(?:[.,]\d+)?(?!\d)"
)
_SCIENTIFIC_PARTS = re.compile(
    r"(?:(?P<mantissa>-?\d+(?:[.,]\d+)?)\s?[×x·]\s?)?10\s?(?:\^\s?)?(?P<sign>[-⁻]\s?)?"
    rf"(?P<power>[{_SUPERSCRIPT_DIGITS}]+|\d{{1,2}})"
)
_SUPERSCRIPTS = str.maketrans(_SUPERSCRIPT_DIGITS, "0123456789")
_BASIS = r"(?:\s?(?:N|P|NO3|NO2|NH4|PO4|O2)(?=\s?[/LlLl]))?"
_UNIT = re.compile(
    r"\s?(?P<unit>%|[°º◦]\s?C|"
    r"(?:[µμunmkK]?(?:g[Ll]|Sc?m)\s?[-−–⁻]\s?[\d¹]|"
    rf"[µμunmkK]?(?:g{_BASIS}|S|L|l|mol|m)|NTU|UNT|UTN|FNU|FTU|ppm|ppb|UFC|NMP|CFU|MPN|"
    r"[Cc][ée]l(?:ulas)?\.?|[Cc]ells?|[Ii]nd\.?)"
    r"(?:\s?[/·]\s?[\wµμ]+|\s[\wµμ]+\s?[-−–⁻]\s?[\d¹²³])*"
    r"(?:\s(?:como|as)\s(?:N|P|NO3|NO2|NH4|PO4)\b)?)(?![\wµμ/])"
)
_UNIT_WORD = re.compile(rf"\s?(?P<word>(?:{'|'.join(UNIT_WORD_STARTS)})\w*)", re.IGNORECASE)
_UNIT_EXPONENTS = {"-1", "-2", "-3"}
_UNIT_BEFORE = re.compile(r"(?<![^\W\d_])(?:[µμmk]?[Ll]|c?m|k?g|s|d|h)\s$")
_LIST_MARKER = re.compile(r"^\s*\d{1,2}[.)]\s+[^\W\d_]{4,}")
_GROUP_SPACES = re.compile(r"[   ]")
_COMPARATOR = re.compile(r"([<>≤≥≈~])\s?$")
_DIGIT_TOKEN = re.compile(r"\S*[\d⁰¹²³⁴-⁹]\S*")
_TOKEN_EDGES = " .,;:()[]{}«»\"'¿?¡!"
_NEARBY_CHARS = 60
_NEARBY_BEFORE = 30
_TABLE_CELL = "|"
_DEVIATION = "±"
_MINUS_SIGNS = str.maketrans({"−": "-", "–": "-"})
_UNIT_SPELLINGS = {
    "ntu": ("ntu", "unt", "utn"),
    "l": ("l", "litre", "litres", "liter", "liters", "litro", "litros"),
    "ml": ("ml", "millilitre", "millilitres", "milliliter", "milliliters", "mililitro", "mililitros"),
    "cel": ("cel", "cell", "cells", "celula", "celulas"),
}
HIDDEN_VALUE = "[valor no confirmado]"


@dataclass(frozen=True)
class Reading:
    """Número leído de un texto, con sus valores posibles y lo que lo acompaña."""

    text: str
    values: frozenset[Decimal]
    unit: str | None
    spelled_unit: str | None = None
    comparator: str | None = None
    start: int = 0
    end: int = 0
    stop: int = 0


#
# Devuelve los valores de un número: con idioma, la lectura de ese idioma; sin él, las posibles.
#
def number_values(token: str, language: str | None = None) -> frozenset[Decimal]:
    text = _GROUP_SPACES.sub("", token)
    if "," not in text and "." not in text:
        return _forms(text)
    if "," in text and "." in text:
        decimal = max(text.rfind("."), text.rfind(","))
        return _forms(f"{re.sub(r'[.,]', '', text[:decimal])}.{text[decimal + 1:]}")

    separator = "," if "," in text else "."
    parts = text.split(separator)
    whole = parts[0].lstrip("-")
    grouped = all(len(part) == 3 for part in parts[1:]) and 1 <= len(whole) <= 3 and whole[0] != "0"
    plain = "".join(parts) if grouped else None
    decimal = ".".join(parts) if len(parts) == 2 else None
    if language is None:
        return _forms(plain, decimal)

    thousands = "." if language == "es" else ","

    return _forms(plain if grouped and separator == thousands else decimal or plain)


#
# Devuelve el valor de un número en notación científica: «7,3 × 10⁵» y su forma aplanada «7.3 × 10 5».
#
def scientific_values(token: str, language: str | None = None) -> frozenset[Decimal]:
    parts = _SCIENTIFIC_PARTS.fullmatch(token.translate(_MINUS_SIGNS))
    if parts is None:
        return frozenset()

    mantissa = parts.group("mantissa")
    mantissas = number_values(mantissa, language) if mantissa else frozenset({Decimal(1)})
    power = int(parts.group("power").translate(_SUPERSCRIPTS))
    exponent = -power if parts.group("sign") else power

    return frozenset(value.scaleb(exponent).normalize() for value in mantissas)


#
# Convierte textos en números normalizados, omitiendo los que no lo son.
#
def _forms(*texts) -> frozenset[Decimal]:
    return frozenset(item for item in (_decimal(text) for text in texts) if item is not None)


#
# Convierte un texto en un número normalizado, o None si no lo es.
#
def _decimal(text: str | None) -> Decimal | None:
    if text is None:
        return None
    try:
        return Decimal(text).normalize()
    except ArithmeticError:
        return None


#
# Devuelve los números de un texto con la unidad y el comparador que los acompañan.
#
def number_readings(text: str, language: str | None = None) -> list[Reading]:
    source = text.translate(_MINUS_SIGNS)
    readings = []
    consumed = 0
    for match in _NUMBER.finditer(source):
        scientific = match.group("scientific") is not None
        values = scientific_values(match.group(), language) if scientific else number_values(match.group(), language)
        if not values or match.start() < consumed or _is_list_marker(source, match):
            continue
        if match.group() in _UNIT_EXPONENTS and _UNIT_BEFORE.search(source[: match.start()]):
            continue

        following = source[match.end() :]
        unit = _UNIT.match(text[match.end() :])
        word = _UNIT_WORD.match(following) if unit is None else None
        comparator = _COMPARATOR.search(source[: match.start()])
        consumed = match.end() + unit.end() if unit else match.end()
        readings.append(
            Reading(
                match.group(),
                values,
                unit.group("unit").strip() if unit else None,
                spelled_unit=word.group("word") if word else None,
                comparator=comparator.group(1) if comparator else None,
                start=match.start(),
                end=match.end(),
                stop=consumed,
            )
        )

    return readings


#
# Indica si el número es la marca de un elemento de una lista numerada.
#
def _is_list_marker(text: str, match) -> bool:
    line_start = text.rfind("\n", 0, match.start()) + 1
    return match.start() - line_start <= 2 and _LIST_MARKER.match(text[line_start:]) is not None


#
# Devuelve las palabras con dígitos que quedaron sin leer como números: «pH9,8», «x10⁴», «.5».
#
def unread_tokens(text: str, readings: list[Reading]) -> list[str]:
    blanked = list(text)
    for reading in readings:
        blanked[reading.start : reading.stop] = " " * (reading.stop - reading.start)

    return [match.group().strip(_TOKEN_EDGES) for match in _DIGIT_TOKEN.finditer("".join(blanked))]


#
# Separa un texto en oraciones sin cortar en abreviaturas como «et al.» o «p.».
#
def sentences(text: str) -> list[str]:
    found, pending = [], ""
    for piece in SENTENCE_END.split(text):
        pending = f"{pending} {piece}".strip()
        if pending and not ABBREVIATION_END.search(pending):
            found.append(pending)
            pending = ""

    return [*found, pending] if pending else found


#
# Devuelve el texto de un fragmento dudoso con sus números y unidades ocultos.
#
def without_numbers(text: str) -> str:
    pieces, position = [], 0
    for reading in number_readings(text):
        pieces.append(_DIGIT_TOKEN.sub(HIDDEN_VALUE, text[position : reading.start]))
        pieces.append(HIDDEN_VALUE)
        position = reading.stop

    return "".join([*pieces, _DIGIT_TOKEN.sub(HIDDEN_VALUE, text[position:])])


#
# Devuelve un texto en minúsculas, sin acentos y con los símbolos de micro y grado unificados.
#
def folded(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text.casefold().replace("µ", "μ").replace("◦", "°"))

    return "".join(char for char in plain if not unicodedata.combining(char))


#
# Indica si cada parte de una unidad figura en un texto como palabra completa, en alguna grafía.
#
def names_unit(unit: str, text: str) -> bool:
    plain = folded(text)

    return all(
        any(
            re.search(rf"(?<![^\W\d_]){re.escape(item)}(?![^\W\d_])", plain)
            for item in _UNIT_SPELLINGS.get(part, (part,))
        )
        for part in comparable_unit("", unit)[0].split("/")
    )


#
# Indica si una fuente escribe la unidad junto a ese número, o en la fila o columna de tabla que lo contiene.
#
def written_nearby(reading, source: str, language: str | None = None) -> bool:
    if _TABLE_CELL in source:
        return _in_labeled_cell(reading, source, language)

    found = number_readings(source, language)

    return any(
        reading.values & (item.values | group_values(item.text))
        and _same_unit_nearby(reading, item, source, _window_start(found, index, source), found[index + 1 :])
        for index, item in enumerate(found)
    )


#
# Devuelve dónde empieza lo que acompaña a un número: tras el anterior, o antes de su «media ±».
#
def _window_start(found, index, source) -> int:
    while index and _DEVIATION in source[found[index - 1].stop : found[index].start]:
        index -= 1

    return found[index - 1].stop if index else 0


#
# Compara la unidad con la del número en la fuente; sin una propia, mira entre sus números vecinos.
#
def _same_unit_nearby(reading, item, source, start, following) -> bool:
    if item.unit and comparable_unit("", item.unit) == comparable_unit("", reading.unit):
        return True
    if item.unit and _numerator(item.unit) != _numerator(reading.unit):
        return False

    stop = following[0].start if following else len(source)
    window = source[max(start, item.start - _NEARBY_BEFORE) : min(stop, item.end + _NEARBY_CHARS)]

    return names_unit(reading.unit, window)


#
# Indica si el número está en una celda que nombra la unidad en ella misma, en su columna o en su fila.
#
def _in_labeled_cell(reading, source: str, language) -> bool:
    rows = [line.split(_TABLE_CELL) for line in source.splitlines() if _TABLE_CELL in line]
    if not rows:
        return False

    header = next((row for row in rows if any(names_unit(reading.unit, cell) for cell in row)), rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            values = {value for item in number_readings(cell, language) for value in item.values}
            column = header[index] if index < len(header) else ""
            labels = (cell, column, _row_label(row))
            if reading.values & values and any(_labels_unit(reading, label) for label in labels):
                return True

    return False


#
# Devuelve la etiqueta de una fila de tabla: su primera celda con contenido.
#
def _row_label(row) -> str:
    return next((cell for cell in row if cell.strip()), "")


#
# Indica si una etiqueta nombra la unidad de la lectura y, si la lectura declara una base química, esa misma.
#
def _labels_unit(reading, label: str) -> bool:
    basis = comparable_unit("", reading.unit)[1]

    return names_unit(reading.unit, label) and basis in {None, concentration_basis(label, label)}


#
# Devuelve la parte de una unidad que indica la magnitud medida.
#
def _numerator(unit: str) -> str:
    return comparable_unit("", unit)[0].split("/")[0]


#
# Devuelve cada número de un token separado por espacios: en una tabla pueden ser celdas distintas.
#
def group_values(token: str) -> frozenset[Decimal]:
    if " " not in token or _SCIENTIFIC_PARTS.fullmatch(token.translate(_MINUS_SIGNS)):
        return frozenset()

    return frozenset(value for part in token.split() for value in number_values(part))
