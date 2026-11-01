import math
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from aquai.application.Rules.ingestion import (
    CENSORED_VALUE,
    DECIMAL_COMMA,
    IDENTIFIER_COLUMN,
    MEASUREMENT_COLUMN,
    MISSING_VALUE_MARKER,
    NON_MEASUREMENT_COLUMN,
)
from aquai.application.units import header_unit
from aquai.infrastructure.processing.tabular.reader import LiteralText, is_empty

RESERVED_COLUMNS = {"_aquai_row", "_aquai_source_row", "_aquai_source_column"}
INTEGER = re.compile(r"[+-]?(?:0|[1-9]\d*)\Z")
NUMBER = re.compile(r"[+-]?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?\Z")


def column_names(originals: list[str]) -> list[str]:
    used = set(RESERVED_COLUMNS)
    names = []

    for index, original in enumerate(originals, 1):
        name = _unused_name(_sql_base_name(original, index), used)
        used.add(name)
        names.append(name)

    return names


def _sql_base_name(original: str, index: int) -> str:
    ascii_name = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    base = re.sub(r"[^a-z0-9_]+", "_", ascii_name.lower()).rstrip("_") or f"column_{index}"

    if base[0].isdigit() or base.startswith("_aquai_"):
        base = "column_" + base.lstrip("_")

    return base[:63]


def _unused_name(base: str, used: set[str]) -> str:
    name, count = base, 2

    while name in used:
        suffix = f"_{count}"
        name = base[: 63 - len(suffix)] + suffix
        count += 1

    return name


def text_value(value: object) -> str | None:
    if is_empty(value):
        return None

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def _number(value: object) -> Decimal | None:
    numeric = _decimal_candidate(value)

    if numeric is None:
        return None

    return numeric if numeric.is_finite() and abs(numeric.adjusted()) <= 1000 else None


def _decimal_candidate(value: object) -> Decimal | None:
    if isinstance(value, (bool, LiteralText)):
        return None

    if isinstance(value, (int, Decimal)):
        return Decimal(value)

    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None

    if isinstance(value, str) and NUMBER.fullmatch(value):
        return _parse_number_text(value)

    return None


def _parse_number_text(value: str) -> Decimal | None:
    if _ambiguous_thousands_separator(value):
        return None

    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _ambiguous_thousands_separator(value: str) -> bool:
    """`1.234` puede ser un decimal o un separador de miles; no se adivina."""
    mantissa = re.split(r"[eE]", value)[0].lstrip("+-")

    if "." not in mantissa:
        return False

    integral, fractional = mantissa.split(".")

    return len(fractional) == 3 and 1 <= len(integral) <= 3 and integral != "0"


def _converted(values: list[object], convert) -> list[object]:
    return [None if is_empty(value) else convert(value) for value in values]


def infer_column(name: str, values: list[object]) -> tuple[str, list[object], str | None]:
    present = [value for value in values if not is_empty(value)]

    if not present:
        return "TEXT", [None] * len(values), None

    if IDENTIFIER_COLUMN.search(name):
        return "TEXT", [text_value(value) for value in values], None

    typed = (
        _boolean_column(values, present)
        or _native_temporal_column(values, present)
        or _iso_text_column(values, present)
        or _numeric_column(values, present)
    )

    if typed is not None:
        return typed

    return "TEXT", [text_value(value) for value in values], _mixed_text_warning(name, present)


def _boolean_column(values: list[object], present: list[object]) -> tuple | None:
    if all(isinstance(value, bool) for value in present):
        return "BOOLEAN", _converted(values, lambda value: value), None

    if all(isinstance(value, str) and value.lower() in {"true", "false"} for value in present):
        return "BOOLEAN", _converted(values, lambda value: value.lower() == "true"), None

    return None


def _native_temporal_column(values: list[object], present: list[object]) -> tuple | None:
    naive_timestamps = all(isinstance(value, datetime) for value in present) and all(
        value.tzinfo is None for value in present
    )

    if naive_timestamps:
        return "TIMESTAMP", _converted(values, lambda value: value), None

    if all(isinstance(value, date) and not isinstance(value, datetime) for value in present):
        return "DATE", _converted(values, lambda value: value), None

    return None


def _iso_text_column(values: list[object], present: list[object]) -> tuple | None:
    for pattern, parser, sql_type in (
        (ISO_DATE, date.fromisoformat, "DATE"),
        (ISO_TIMESTAMP, datetime.fromisoformat, "TIMESTAMP"),
    ):
        if not all(isinstance(value, str) and pattern.fullmatch(value) for value in present):
            continue

        try:
            return sql_type, _converted(values, parser), None
        except ValueError:
            return None

    return None


def _numeric_column(values: list[object], present: list[object]) -> tuple | None:
    numbers = [_number(value) for value in present]

    if not all(value is not None for value in numbers):
        return None

    if _fits_bigint(present, numbers):
        return "BIGINT", _converted(values, lambda value: int(_number(value))), None

    return "NUMERIC", _converted(values, _number), None


def _fits_bigint(present: list[object], numbers: list[Decimal]) -> bool:
    if not all(value == value.to_integral_value() for value in numbers):
        return False

    # Una fracción escrita conserva su escala incluso cuando representa un entero.
    if any(isinstance(value, str) and not INTEGER.fullmatch(value) for value in present):
        return False

    if any(isinstance(value, Decimal) and value.as_tuple().exponent < 0 for value in present):
        return False

    if any(isinstance(value, float) for value in present):
        return False

    return all(-(2**63) <= value < 2**63 for value in numbers)


def _mixed_text_warning(name: str, present: list[object]) -> str | None:
    uncertain = any(
        isinstance(value, (int, float, Decimal, date))
        or (isinstance(value, str) and re.match(r"^[+-]?\d", value))
        for value in present
    )

    if not uncertain:
        return None

    return f"La columna '{name}' permanece TEXT porque contiene valores mixtos o ambiguos."


def numeric_projection(name: str, values: list[object]) -> tuple[list[Decimal | None], int] | None:
    """Deriva números inequívocos sin reemplazar el texto ni interpretar marcadores. Un valor censurado («<0,001») es
    una medición, no texto libre: cuenta como tal pero queda sin dato, igual que un marcador de faltante."""
    if NON_MEASUREMENT_COLUMN.search(name):
        return None

    if not header_unit(name) and not MEASUREMENT_COLUMN.search(name):
        return None

    present = [value for value in values if not is_empty(value)]
    convert = _projection_parser(present)
    numbers = [convert(value) for value in present]
    convertible = sum(number is not None for number in numbers)
    censored = sum(_is_censored(value) for value, number in zip(present, numbers) if number is None)

    if not _mostly_numeric_with_gaps(convertible, censored, len(present)):
        return None

    markers = [value for value, number in zip(present, numbers) if number is None]

    if not all(_is_missing_marker(value) or _is_censored(value) for value in markers):
        return None

    return _converted(values, convert), len(present) - convertible


def _mostly_numeric_with_gaps(convertible: int, censored: int, present: int) -> bool:
    measured = convertible + censored
    return convertible >= 2 and convertible != present and measured * 4 >= present * 3


def _is_censored(value: object) -> bool:
    return isinstance(value, str) and CENSORED_VALUE.fullmatch(value.strip()) is not None


def _projection_parser(present: list[object]):
    """La columna demuestra cómo escribe sus decimales: con coma («12,5») o con punto aunque haya valores como «1.337»
    que, solos, podrían ser miles («0.837» y «9.12» prueban que son decimales). Sin esa prueba, no se adivina."""
    if _uses_decimal_comma(present):
        return _comma_number
    if _uses_decimal_point(present):
        return _point_number
    return _number


def _uses_decimal_point(present: list[object]) -> bool:
    texts = [value.strip() for value in present if isinstance(value, str)]
    decimals = [text for text in texts if NUMBER.fullmatch(text) and "." in text.split("e")[0].split("E")[0]]
    return any(not _ambiguous_thousands_separator(text) for text in decimals) and not any(
        DECIMAL_COMMA.fullmatch(text) for text in texts
    )


def _point_number(value: object) -> Decimal | None:
    if isinstance(value, str) and NUMBER.fullmatch(value.strip()):
        number = Decimal(value.strip())
        return number if number.is_finite() and abs(number.adjusted()) <= 1000 else None
    return _number(value)


def _uses_decimal_comma(present: list[object]) -> bool:
    """La columna escribe decimales con coma: ninguno con punto y alguno que no puede ser un separador de miles."""
    texts = [value.strip() for value in present if isinstance(value, str)]
    commas = [text for text in texts if DECIMAL_COMMA.fullmatch(text)]
    dotted = any(NUMBER.fullmatch(text) and "." in text for text in texts)
    unambiguous = any(
        len(text.split(",")[1]) != 3 or text.lstrip("+-").startswith("0,") for text in commas
    )
    return bool(commas) and not dotted and unambiguous


def _comma_number(value: object) -> Decimal | None:
    if isinstance(value, str) and DECIMAL_COMMA.fullmatch(value.strip()):
        return _number(value.strip().replace(",", "."))
    return _number(value)


def _is_missing_marker(value: object) -> bool:
    return isinstance(value, str) and MISSING_VALUE_MARKER.fullmatch(value.strip()) is not None
