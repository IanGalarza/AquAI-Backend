import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal

from aquai.application.answering.measures import shown_value
from aquai.application.answering.presentation import format_unit
from aquai.application.answering.readings import RANGE_JOINT, number_readings
from aquai.application.contracts.answering import UnitHypothesis
from aquai.application.Rules.evidence import GUIDELINE_CUES, MEAN_WITH_DEVIATION, ORGANIZATION_AGENT
from aquai.application.Rules.units import UNIT_BASES, UNIT_PREFIXES
from aquai.application.units import unit_key

_MENTION_WINDOW = 90
_BRACKET_UNIT = re.compile(r"^\s*[\[(]\s*(?P<unit>[^\[\]()]{1,16}?)\s*[\])]")
_VALUE_BOUNDARY = re.compile(r"[|\n;]")
_MAX_UNITS = 3
_MAX_QUOTE = 110
_NAME_SEPARATOR = re.compile(r"[\s\-‐–]+")
_DOUBT_RATIO = Decimal(100)
_SAME_ORDER = Decimal(10)
_DOUBT_TEXT = {
    "es": "La unidad declarada de {label} ({declared_unit}) podría ser incorrecta: leídos en esa unidad, sus valores "
    "(hasta {observed} {declared_unit}, es decir {converted} {unit}) quedan más de cien veces {side} los "
    "{documented} {unit} que informa la bibliografía recuperada para la misma variable, y leídos en {unit} quedan "
    "en su mismo orden. No cambié la unidad del dato y no hice comparaciones numéricas con esos valores.",
    "en": "The unit declared for {label} ({declared_unit}) may be wrong: read in that unit, its values (up to "
    "{observed} {declared_unit}, that is {converted} {unit}) are more than a hundred times {side} the {documented} "
    "{unit} that the retrieved literature reports for the same variable, while read in {unit} they are in the same "
    "range. I did not change the data's unit and made no numerical comparison with these values.",
}
_DOUBT_SIDES = {
    "es": {"above": "por encima de", "below": "por debajo de"},
    "en": {"above": "above", "below": "below"},
}
_POSITIONS = {
    "es": {
        "above": "quedaría por encima de",
        "below": "quedaría por debajo de",
        "within": "quedaría dentro de",
    },
    "en": {
        "above": "would be above",
        "below": "would be below",
        "within": "would be within",
    },
}


@dataclass(frozen=True)
class DeclaredUnitDoubt:
    """Indicio documental de que la unidad declarada de un resultado no corresponde a sus valores."""

    field: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class DocumentedValue:
    """Unidad y valores con que un fragmento expresa una variable."""

    unit: str | None
    low: Decimal
    high: Decimal
    evidence_id: str
    quote: str


#
# Devuelve las lecturas condicionales de los resultados sin unidad, según lo que dicen los fragmentos.
#
def unit_hypotheses(results, fragments, language) -> list[UnitHypothesis]:
    hypotheses = []
    for result in results:
        if not result.lacks_unit or not result.values:
            continue

        found = [item for item in documented_values(result, fragments) if item.unit]
        units = _ranked_units(found)
        hypotheses.extend(
            _hypothesis(result, unit, units, found, language) for unit in units.values()
        )

    return hypotheses


#
# Reúne las unidades y valores que los fragmentos escriben junto a la variable.
#
def documented_values(result, fragments) -> list[DocumentedValue]:
    pattern = re.compile(
        r"(?<![^\W\d_])(?:"
        + "|".join(_name_pattern(name) for name in result.names if name)
        + r")(?![^\W\d_])",
        re.IGNORECASE,
    )
    found = []
    for fragment in fragments:
        text = _plain(fragment.text)
        for match in pattern.finditer(text):
            following = text[match.start() : match.end() + _MENTION_WINDOW]
            value = _following_value(following, match.end() - match.start(), fragment, result)
            if value is not None:
                found.append(value)

    return found


#
# Devuelve el patrón de un nombre que acepta sus palabras separadas por espacio o guion, o juntas, como quedan
# al cortar una línea en un PDF: «microcystin-LR», «microcystin LR», «microcystinLR».
#
def _name_pattern(name) -> str:
    words = [word for word in _NAME_SEPARATOR.split(_plain(name)) if word]

    return r"[\s\-‐–]?".join(re.escape(word) for word in words)


#
# Devuelve los resultados cuya unidad declarada podría ser incorrecta: leídos en esa unidad quedan cien veces o
# más fuera de lo que los fragmentos informan de la misma variable en otra escala de la misma magnitud, y
# leídos en esa escala quedan en su mismo orden. Las guías y los límites citados no cuentan como observaciones.
#
def declared_unit_doubts(results, fragments, language) -> list[DeclaredUnitDoubt]:
    doubts = {}
    for result in results:
        if result.field in doubts or result.unit_status != "documented" or not result.unit or not result.values:
            continue

        observed = [
            item
            for item in documented_values(result, fragments)
            if item.unit and not (GUIDELINE_CUES.search(item.quote) or ORGANIZATION_AGENT.search(item.quote))
        ]
        for values in _by_unit(observed).values():
            doubt = _scale_doubt(result, values, language)
            if doubt is not None:
                doubts[result.field] = doubt
                break

    return list(doubts.values())


#
# Agrupa los valores documentados por unidad, juntando sus grafías equivalentes.
#
def _by_unit(found) -> dict[str, list[DocumentedValue]]:
    grouped = {}
    for item in found:
        grouped.setdefault(unit_key(item.unit), []).append(item)

    return grouped


#
# Devuelve el indicio de que el resultado está en la escala de los valores documentados y no en la declarada.
#
def _scale_doubt(result, values, language) -> DeclaredUnitDoubt | None:
    unit = values[0].unit
    factor = unit_factor(result.unit, unit)
    if factor is None or Decimal(1) / _DOUBT_RATIO < factor < _DOUBT_RATIO:
        return None

    low, high = min(item.low for item in values), max(item.high for item in values)
    observed = max(result.values)
    declared = observed * factor
    if low <= 0 or observed <= 0 or low / _DOUBT_RATIO <= declared <= high * _DOUBT_RATIO:
        return None
    if not low / _SAME_ORDER <= observed <= high * _SAME_ORDER:
        return None

    documented = shown_value(low, language) if low == high else f"{shown_value(low, language)}-{shown_value(high, language)}"
    text = _DOUBT_TEXT[language].format(
        label=result.label,
        declared_unit=format_unit(result.unit),
        observed=shown_value(observed, language),
        converted=shown_value(declared, language),
        unit=format_unit(unit),
        documented=documented,
        side=_DOUBT_SIDES[language]["above" if declared > high else "below"],
    )

    return DeclaredUnitDoubt(result.field, text, tuple(dict.fromkeys(item.evidence_id for item in values)))


#
# Lee la unidad y los valores que siguen a la mención de una variable, hasta el siguiente dato.
#
def _following_value(window, mention_length, fragment, result) -> DocumentedValue | None:
    text = _VALUE_BOUNDARY.split(window[mention_length:], maxsplit=1)[0]
    bracket = _BRACKET_UNIT.match(text)
    unit = bracket.group("unit") if bracket and _plausible(bracket.group("unit")) else None
    readings = number_readings(text[bracket.end() :] if unit else text)
    if unit is None:
        unit = next((item.unit for item in readings if item.unit), None)
    if not readings or (unit is None and result.unit_status != "dimensionless"):
        return None

    values = _values_of(readings, text[bracket.end() :] if bracket and unit else text, unit)
    if not values:
        return None

    quote = _VALUE_BOUNDARY.split(window, maxsplit=1)[0].strip()

    return DocumentedValue(unit, min(values), max(values), fragment.evidence_id, quote)


#
# Devuelve los valores de la variable: la media de «media ± desvío» o el valor o rango con esa unidad.
#
def _values_of(readings, text, unit) -> list[Decimal]:
    mean = MEAN_WITH_DEVIATION.search(text)
    if mean is not None:
        return _unambiguous(number_readings(mean.group("mean"))[:1])

    chosen = next((index for index, item in enumerate(readings) if item.unit == unit), 0)
    first = readings[chosen]
    previous = readings[chosen - 1] if chosen else None
    following = readings[chosen + 1] if chosen + 1 < len(readings) else None
    if previous is not None and not previous.unit and RANGE_JOINT.fullmatch(text[previous.end : first.start]):
        return _unambiguous([previous, first])
    if following is not None and following.unit == first.unit:
        if RANGE_JOINT.fullmatch(text[first.stop : following.start]):
            return _unambiguous([first, following])

    return _unambiguous([first])


#
# Devuelve los valores de lecturas que no admiten dos interpretaciones de su separador.
#
def _unambiguous(readings) -> list[Decimal]:
    if any(len(item.values) != 1 for item in readings):
        return []

    return [next(iter(item.values)) for item in readings]


#
# Indica si un texto entre corchetes se lee como una unidad y no como una cita.
#
def _plausible(text) -> bool:
    readings = number_readings(f"1 {text}")

    return bool(readings) and readings[0].unit is not None


#
# Devuelve las unidades documentadas más frecuentes, una por cada grafía equivalente.
#
def _ranked_units(found) -> dict[str, str]:
    counts, names = {}, {}
    for item in found:
        key = unit_key(item.unit)
        counts[key] = counts.get(key, 0) + 1
        names.setdefault(key, item.unit)

    ranked = sorted(counts, key=counts.get, reverse=True)[:_MAX_UNITS]

    return {key: names[key] for key in ranked}


#
# Construye una lectura condicional del valor observado bajo una unidad.
#
def _hypothesis(result, unit, units, found, language) -> UnitHypothesis:
    same = [item for item in found if unit_key(item.unit) == unit_key(unit)]

    return UnitHypothesis(
        field=result.field,
        label=result.label,
        column=result.column,
        observed=_observed(result, language),
        unit=format_unit(unit),
        equivalents=_equivalents(result, unit, units, language),
        documented=[f"{_position(result, item, language)} («{_cited(item)}»)" for item in same[:3]],
        evidence_ids=list(dict.fromkeys(item.evidence_id for item in same)),
    )


#
# Devuelve la cita de un valor documentado, sin cortar una palabra.
#
def _cited(value) -> str:
    quote = " ".join(value.quote.split())
    if len(quote) <= _MAX_QUOTE:
        return quote

    return f"{quote[:_MAX_QUOTE].rsplit(' ', 1)[0]}…"


#
# Devuelve el valor observado, o su rango si el resultado tiene varias filas.
#
def _observed(result, language) -> str:
    low, high = min(result.values), max(result.values)
    if low == high:
        return shown_value(low, language)

    return f"{shown_value(low, language)}-{shown_value(high, language)}"


#
# Devuelve el valor expresado en las otras unidades candidatas de la misma magnitud.
#
def _equivalents(result, unit, units, language) -> list[str]:
    value = max(result.values)
    converted = []
    for other in units.values():
        factor = unit_factor(unit, other)
        if factor is not None and factor != 1:
            converted.append(f"{shown_value(value * factor, language)} {format_unit(other)}")

    return converted


#
# Devuelve dónde quedaría el valor respecto de lo documentado en esa unidad.
#
def _position(result, item, language) -> str:
    if item.unit is None:
        return ""

    value = max(result.values)
    position = "above" if value > item.high else "below" if value < item.low else "within"
    bounds = shown_value(item.low, language)
    if item.low != item.high:
        bounds += f"-{shown_value(item.high, language)}"

    return f"{_POSITIONS[language][position]} {bounds} {format_unit(item.unit)}"


#
# Devuelve el factor que convierte una unidad en otra de la misma magnitud, o None.
#
def unit_factor(source: str, target: str) -> Decimal | None:
    left, right = _scaled(unit_key(source)), _scaled(unit_key(target))
    if left is None or right is None or left[1] != right[1]:
        return None

    return left[0] / right[0]


#
# Devuelve la escala de una unidad y su forma sin prefijos.
#
def _scaled(key: str) -> tuple[Decimal, tuple] | None:
    numerator, _, denominator = key.partition("/")
    top = _prefixed(numerator)
    bottom = _prefixed(denominator) if denominator else (Decimal(1), "")
    if top is None or bottom is None:
        return None

    return top[0] / bottom[0], (top[1], bottom[1])


#
# Separa el prefijo de una unidad simple y devuelve su factor y su base.
#
def _prefixed(part: str) -> tuple[Decimal, str] | None:
    if part in UNIT_BASES:
        return Decimal(1), part

    prefix, base = part[:1], part[1:]
    if prefix in UNIT_PREFIXES and base in UNIT_BASES:
        return Decimal(UNIT_PREFIXES[prefix]), base

    return None


#
# Devuelve un texto sin acentos, conservando mayúsculas y símbolos de unidad.
#
def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.replace("µ", "μ"))

    return "".join(char for char in decomposed if not unicodedata.combining(char))
