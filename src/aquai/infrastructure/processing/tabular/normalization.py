"""Nombres SQL seguros y conversiones conservadoras sobre columnas completas."""

import math
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

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
        ascii_name = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
        base = re.sub(r"[^a-z0-9_]+", "_", ascii_name.lower()).rstrip("_") or f"column_{index}"
        if base[0].isdigit() or base.startswith("_aquai_"):
            base = "column_" + base.lstrip("_")
        base = base[:63]
        name, count = base, 2
        while name in used:
            suffix = f"_{count}"
            name = base[:63 - len(suffix)] + suffix
            count += 1
        used.add(name)
        names.append(name)
    return names


def header_unit(name: str) -> str | None:
    match = re.search(r"(?:\[([^\[\]]+)\]|\(([^()]+)\))\s*$", name)
    if not match:
        return None
    unit = (match.group(1) or match.group(2)).strip()
    if "\ufffd" in unit:
        return None
    simple = re.fullmatch(r"(?:%|[munpkMGTµμ]?g|[munpkµμ]?m|[munpµμ]?[Ll]|°[CF]|[munpµμ]?mol|s|min|h|ppm|ppb)", unit)
    inverse = re.fullmatch(r"[\wµμ°%^. ·-]+\s+[munpµμ]?[Llms](?:-\d+|\^-\d+)", unit)
    return unit if len(unit) <= 40 and (simple or inverse or re.fullmatch(r"[\wµμ°%^. ·-]+/[\wµμ°%^. ·-]+", unit)) else None


def text_value(value: object) -> str | None:
    if is_empty(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _number(value: object) -> Decimal | None:
    if isinstance(value, (bool, LiteralText)):
        return None
    if isinstance(value, (int, Decimal)):
        numeric = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            return None
        numeric = Decimal(str(value))
    elif isinstance(value, str) and NUMBER.fullmatch(value):
        mantissa = re.split(r"[eE]", value)[0].lstrip("+-")
        if "." in mantissa:
            integral, fractional = mantissa.split(".")
            if len(fractional) == 3 and 1 <= len(integral) <= 3 and integral != "0":
                return None
        try:
            numeric = Decimal(value)
        except InvalidOperation:
            return None
    else:
        return None
    return numeric if numeric.is_finite() and abs(numeric.adjusted()) <= 1000 else None


def infer_column(name: str, values: list[object]) -> tuple[str, list[object], str | None]:
    present = [value for value in values if not is_empty(value)]
    if not present:
        return "TEXT", [None] * len(values), None
    identifier = re.search(r"(?:^|[\s_-])(?:id|code|codigo|código|identificador)(?:$|[\s_-])", name, re.IGNORECASE)
    if identifier:
        return "TEXT", [text_value(value) for value in values], None
    if all(isinstance(value, bool) for value in present):
        return "BOOLEAN", [None if is_empty(value) else value for value in values], None
    if all(isinstance(value, str) and value.lower() in {"true", "false"} for value in present):
        return "BOOLEAN", [None if is_empty(value) else value.lower() == "true" for value in values], None
    if all(isinstance(value, datetime) for value in present) and all(value.tzinfo is None for value in present):
        return "TIMESTAMP", [None if is_empty(value) else value for value in values], None
    if all(isinstance(value, date) and not isinstance(value, datetime) for value in present):
        return "DATE", [None if is_empty(value) else value for value in values], None
    for pattern, parser, sql_type in ((ISO_DATE, date.fromisoformat, "DATE"),
                                      (ISO_TIMESTAMP, datetime.fromisoformat, "TIMESTAMP")):
        if all(isinstance(value, str) and pattern.fullmatch(value) for value in present):
            try:
                return sql_type, [None if is_empty(value) else parser(value) for value in values], None
            except ValueError:
                break
    numbers = [_number(value) for value in present]
    if all(value is not None for value in numbers):
        integers = all(value == value.to_integral_value() for value in numbers)
        # Una fracción escrita conserva su escala incluso cuando representa un entero.
        integers = integers and all(not isinstance(value, str) or INTEGER.fullmatch(value) for value in present)
        integers = integers and all(not isinstance(value, Decimal) or value.as_tuple().exponent >= 0 for value in present)
        integers = integers and all(not isinstance(value, float) for value in present)
        if integers and all(-(2 ** 63) <= value < 2 ** 63 for value in numbers):
            return "BIGINT", [None if is_empty(value) else int(_number(value)) for value in values], None
        return "NUMERIC", [None if is_empty(value) else _number(value) for value in values], None
    uncertain = any(isinstance(value, (int, float, Decimal, date))
                    or (isinstance(value, str) and re.match(r"^[+-]?\d", value)) for value in present)
    warning = f"La columna '{name}' permanece TEXT porque contiene valores mixtos o ambiguos." if uncertain else None
    return "TEXT", [text_value(value) for value in values], warning


def numeric_projection(name: str, values: list[object]) -> tuple[list[Decimal | None], int] | None:
    """Deriva números inequívocos sin reemplazar el texto ni interpretar marcadores."""
    if re.search(r"(?:id|code|c[oó]digo|identificador|fecha|date|sitio|site|muestra)", name, re.IGNORECASE):
        return None
    if not header_unit(name) and not re.search(r"\b(?:abundancia|cantidad|recuento|valor|medida|concentraci[oó]n|value|amount|count|measurement)\b", name, re.IGNORECASE):
        return None
    present = [value for value in values if not is_empty(value)]
    numeric = [_number(value) for value in present]
    count = sum(value is not None for value in numeric)
    if count < 2 or count == len(present) or count * 4 < len(present) * 3:
        return None
    markers = [value for value, number in zip(present, numeric) if number is None]
    if any(not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z .<>?-]{1,16}", value.strip())
           for value in markers):
        return None
    return [None if is_empty(value) else _number(value) for value in values], len(present) - count
