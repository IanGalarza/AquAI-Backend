import re
import unicodedata

from aquai.application.Rules.units import (
    ASCII_MICRO_PREFIX,
    ELEMENT_NAMES,
    HEADER_UNIT,
    INVERSE_UNIT,
    MAX_UNIT_LENGTH,
    MISDECODED_SYMBOLS,
    NAME_BASIS,
    ORGANISM_COUNTS,
    RATIO_UNIT,
    SIMPLE_UNIT,
    STACKED_HEADER_SEPARATOR,
    SYMBOL_VARIANTS,
    UNIT_BASIS,
    UNIT_EQUIVALENTS,
    UNIT_WORDS,
)
from aquai.application.Rules.variables import NAME_SEPARATORS, PH_ALIASES

_INVERSE_POWER = re.compile(r"\s+(\S+?)\^?-(\d)$")
_SPACED_POWER = re.compile(r"\s*-\s*(\d)$")
_GLUED_POWER = re.compile(r"(?<=[gGsS])(?=(?:[Ll]|c?m)-\d$)")
_POWER_SIGNS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻−–", "0123456789---")
_WORD = re.compile(r"[^\W\d_]+")


#
# Indica si un nombre de columna corresponde al pH.
#
def is_ph_name(name: str) -> bool:
    normalized = NAME_SEPARATORS.sub("", name).casefold()

    return normalized in PH_ALIASES


#
# Indica si todas las columnas de origen de un campo son de pH.
#
def is_ph_field(field) -> bool:
    return bool(field.sources) and all(is_ph_name(ref.column) for ref in field.sources)


#
# Devuelve el estado de unidad a presentar, tratando el pH como adimensional.
#
def presentation_unit_status(field):
    return "dimensionless" if is_ph_field(field) else field.unit_status


#
# Extrae la unidad documentada en un encabezado si es reconocible.
#
def header_unit(name: str) -> str | None:
    match = HEADER_UNIT.search(name)

    if not match:
        return None

    unit = (match.group(1) or match.group(2)).strip()

    if "�" in unit or len(unit) > MAX_UNIT_LENGTH:
        return None

    recognized = (
        SIMPLE_UNIT.fullmatch(unit) or INVERSE_UNIT.fullmatch(unit) or RATIO_UNIT.fullmatch(unit)
    )

    return unit if recognized else None


#
# Devuelve la unidad si una celda es por sí sola una unidad.
#
def label_unit(value: object) -> str | None:
    return header_unit(f"unit [{value}]")


#
# Corrige símbolos de unidad mal decodificados o con variantes.
#
def repaired_symbols(unit: str) -> str:
    for misdecoded, symbol in {**MISDECODED_SYMBOLS, **SYMBOL_VARIANTS}.items():
        unit = unit.replace(misdecoded, symbol)

    return unit


#
# Devuelve una clave normalizada que iguala las distintas grafías de una misma unidad.
#
def unit_key(unit: str) -> str:
    without_basis = UNIT_BASIS.sub(lambda match: match.group("unit") or "", repaired_symbols(unit))
    text = _SPACED_POWER.sub(r"-\1", without_basis.translate(_POWER_SIGNS).strip())
    text = _GLUED_POWER.sub(" ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(_INVERSE_POWER.sub(_division, text).split()).casefold()
    text = _WORD.sub(lambda match: UNIT_WORDS.get(match.group(), match.group()), text)
    text = ASCII_MICRO_PREFIX.sub("μ", text)

    return UNIT_EQUIVALENTS.get(text, text)


#
# Indica si una unidad cuenta organismos: dice qué se contó, aunque el nombre de la columna no lo diga.
#
def counts_organisms(unit: str | None) -> bool:
    return bool(unit) and unit_key(unit).split("/")[0] in ORGANISM_COUNTS


#
# Devuelve la especie química en que se expresa una concentración, si se declara.
#
def concentration_basis(name: str, unit: str | None) -> str | None:
    found = UNIT_BASIS.search(repaired_symbols(unit or ""))
    if found:
        return found.group("lead") or found.group("element") or found.group("tail")

    found = NAME_BASIS.search(name)
    if found:
        return found.group("lead") or found.group("tail") or found.group("word")

    text = unicodedata.normalize("NFKD", name.casefold())
    words = "".join(char for char in text if not unicodedata.combining(char)).split()
    return next((ELEMENT_NAMES[word] for word in words if word in ELEMENT_NAMES), None)


#
# Devuelve unidad y base química, que deben coincidir para comparar dos valores.
#
def comparable_unit(name: str, unit: str) -> tuple[str, str | None]:
    return unit_key(unit), concentration_basis(name, unit)


#
# Convierte una potencia negativa de unidad en una división.
#
def _division(match) -> str:
    power = match.group(2)
    return f"/{match.group(1)}{'' if power == '1' else power}"


#
# Devuelve el encabezado sin la unidad que documenta.
#
def unitless_name(original_name: str) -> str:
    if header_unit(original_name):
        return original_name[: HEADER_UNIT.search(original_name).start()]
    if STACKED_HEADER_SEPARATOR in original_name:
        name, unit = original_name.rsplit(STACKED_HEADER_SEPARATOR, 1)
        if label_unit(unit):
            return name

    return original_name


#
# Devuelve la unidad documentada en el propio encabezado, si existe.
#
def documented_unit(original_name: str) -> str | None:
    unit = header_unit(original_name)
    if unit is None and STACKED_HEADER_SEPARATOR in original_name:
        unit = label_unit(original_name.rsplit(STACKED_HEADER_SEPARATOR, 1)[1])

    return unit
