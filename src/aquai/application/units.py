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
    RATIO_UNIT,
    SIMPLE_UNIT,
    STACKED_HEADER_SEPARATOR,
    SYMBOL_VARIANTS,
    UNIT_BASIS,
    UNIT_EQUIVALENTS,
    UNIT_WORDS,
)
from aquai.application.Rules.variables import NAME_SEPARATORS, PH_ALIASES

# «mg L-1» se escribe también «mg/L» y «mg m-3», «mg/m3»: la potencia negativa de la última unidad es una división.
_INVERSE_POWER = re.compile(r"\s+(\S+?)\^?-(\d)$")
_WORD = re.compile(r"[^\W\d_]+")


def is_ph_name(name: str) -> bool:
    normalized = NAME_SEPARATORS.sub("", name).casefold()

    return normalized in PH_ALIASES


def is_ph_field(field) -> bool:
    return bool(field.sources) and all(is_ph_name(ref.column) for ref in field.sources)


def presentation_unit_status(field):
    return "dimensionless" if is_ph_field(field) else field.unit_status


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


def label_unit(value: object) -> str | None:
    """Una celda que por sí sola es una unidad, como la fila de unidades de un encabezado («mS cm-1»)."""
    return header_unit(f"unit [{value}]")


def repaired_symbols(unit: str) -> str:
    for misdecoded, symbol in {**MISDECODED_SYMBOLS, **SYMBOL_VARIANTS}.items():
        unit = unit.replace(misdecoded, symbol)

    return unit


def unit_key(unit: str) -> str:
    """Grafías de una misma unidad: mg/l, mg/L y mg L-1; cél/mL, Cel mL-1 y cells/mL; µS/cm con el signo micro, la
    letra griega o una «u»; NTU, UNT y UTN; mg/m3 y µg/L. La base química («mg N/L») no forma parte de la clave: se
    compara aparte (comparable_unit)."""
    without_basis = UNIT_BASIS.sub(lambda match: match.group("unit") or "", repaired_symbols(unit))
    text = unicodedata.normalize("NFKD", without_basis.strip())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = "".join(_INVERSE_POWER.sub(_division, text).split()).casefold()
    text = _WORD.sub(lambda match: UNIT_WORDS.get(match.group(), match.group()), text)
    text = ASCII_MICRO_PREFIX.sub("μ", text)

    return UNIT_EQUIVALENTS.get(text, text)


def concentration_basis(name: str, unit: str | None) -> str | None:
    """Especie en que se expresa la concentración de un nutriente: «N», «P», un ion («NO3») o None si no la declara."""
    found = UNIT_BASIS.search(repaired_symbols(unit or ""))
    if found:
        return found.group("lead") or found.group("tail")

    found = NAME_BASIS.search(name)
    if found:
        return found.group("lead") or found.group("tail") or found.group("word")

    text = unicodedata.normalize("NFKD", name.casefold())
    words = "".join(char for char in text if not unicodedata.combining(char)).split()
    return next((ELEMENT_NAMES[word] for word in words if word in ELEMENT_NAMES), None)


def comparable_unit(name: str, unit: str) -> tuple[str, str | None]:
    """Dos valores son directamente comparables sólo con la misma unidad y la misma base química."""
    return unit_key(unit), concentration_basis(name, unit)


def _division(match) -> str:
    power = match.group(2)
    return f"/{match.group(1)}{'' if power == '1' else power}"


def unitless_name(original_name: str) -> str:
    """Encabezado sin la unidad que documenta: «T | (°C)» → «T |», «Cond | mS cm-1» → «Cond»."""
    if header_unit(original_name):
        return original_name[: HEADER_UNIT.search(original_name).start()]
    if STACKED_HEADER_SEPARATOR in original_name:
        name, unit = original_name.rsplit(STACKED_HEADER_SEPARATOR, 1)
        if label_unit(unit):
            return name

    return original_name


def documented_unit(original_name: str) -> str | None:
    """Unidad que documenta el propio encabezado: al final entre corchetes o paréntesis, o como fila inferior de un
    encabezado apilado («Cond | mS cm-1»). Nunca se deduce del nombre de la variable."""
    unit = header_unit(original_name)
    if unit is None and STACKED_HEADER_SEPARATOR in original_name:
        unit = label_unit(original_name.rsplit(STACKED_HEADER_SEPARATOR, 1)[1])

    return unit
