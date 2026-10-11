"""Parsers explícitos que conservan entradas no autorizadas o ambiguas."""

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from numbers import Number


@dataclass(frozen=True)
class ParsingResult:
    value: object
    changed: bool = False
    conflict: str | None = None


def trace_value(value: object) -> object:
    """Serializa evidencias sin recortar textos ni alterar sus espacios."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def parse_explicit_date(value: object, formats: tuple[str, ...]) -> ParsingResult:
    if value is None or isinstance(value, (date, datetime)):
        return ParsingResult(value)
    if isinstance(value, Number):
        return ParsingResult(value, conflict="El número de la columna de fecha no tiene evidencia suficiente para interpretar un serial Excel.")
    if not isinstance(value, str) or not value.strip():
        return ParsingResult(value)
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?", text):
        try:
            datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return ParsingResult(value, conflict="El texto ISO contiene una fecha inválida; se conserva el original.")
        return ParsingResult(text, changed=text != value)
    for explicit_format in formats:
        try:
            parsed = datetime.strptime(text, explicit_format)
        except ValueError:
            continue
        prepared = parsed.date().isoformat()
        return ParsingResult(prepared, changed=prepared != value)
    return ParsingResult(value, conflict="El texto de fecha no coincide con los formatos explícitos autorizados; se conserva el original.")


def parse_mixed_decimal(value: object, separators: tuple[str, ...] = (",", ".")) -> ParsingResult:
    if not isinstance(value, str) or not value.strip():
        return ParsingResult(value)
    text = value.strip()
    if not re.fullmatch(r"[+-]?[\d.,]+(?:[eE][+-]?\d+)?", text):
        return ParsingResult(value)
    mantissa, *exponent = re.split(r"[eE]", text)
    found = {separator for separator in (",", ".") if separator in mantissa}
    if not found.issubset(separators):
        return ParsingResult(value, conflict="El separador numérico no está autorizado en esta columna.")
    normalized = mantissa
    if len(found) == 2:
        decimal_separator = "," if mantissa.rfind(",") > mantissa.rfind(".") else "."
        grouping_separator = "." if decimal_separator == "," else ","
        integral, fractional = mantissa.rsplit(decimal_separator, 1)
        grouping = re.escape(grouping_separator)
        if (not fractional.isdigit() or decimal_separator in integral
                or not re.fullmatch(r"[+-]?\d{1,3}(?:" + grouping + r"\d{3})+", integral)):
            return ParsingResult(value, conflict="Los separadores decimales y de miles no forman una representación inequívoca.")
        normalized = integral.replace(grouping_separator, "") + "." + fractional
    elif found:
        separator = next(iter(found))
        if mantissa.count(separator) != 1:
            return ParsingResult(value, conflict="Los separadores repetidos requieren aclarar la representación de miles.")
        integral, fractional = mantissa.split(separator)
        unsigned_integral = integral.lstrip("+-")
        if (fractional.isdigit() and len(fractional) == 3 and unsigned_integral.isdigit()
                and 1 <= len(unsigned_integral) <= 3 and int(unsigned_integral) != 0):
            return ParsingResult(value, conflict="Un separador seguido por tres dígitos puede ser decimal o de miles; se conserva el original.")
        normalized = mantissa.replace(separator, ".")
    if exponent:
        normalized += "E" + exponent[0]
    try:
        numeric = Decimal(normalized)
        if not numeric.is_finite():
            raise InvalidOperation
        prepared = int(numeric) if numeric == numeric.to_integral_value() else float(numeric)
        if isinstance(prepared, float) and (not math.isfinite(prepared) or Decimal(str(prepared)) != numeric):
            return ParsingResult(value, conflict="La conversión perdería precisión o magnitud; se conserva el texto original.")
    except (InvalidOperation, ValueError, OverflowError):
        return ParsingResult(value, conflict="El texto numérico no puede convertirse conservando su magnitud.")
    return ParsingResult(prepared, changed=True)
