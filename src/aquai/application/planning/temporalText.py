"""Períodos explícitos guardados como TEXT; nunca completa años, días ni zonas horarias.

Formatos reconocidos (los mismos que aparecen literalmente en planillas):
AAAA-MM-DD, D/M/AAAA (orden día/mes demostrado por la propia columna), M/AAAA,
M-M/AAAA, M/AAAA-M/AAAA y AAAA. Cada valor se interpreta como el intervalo que nombra.
"""

import calendar
import re
from datetime import date

_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T]00:00(?::00)?)?$")
_DAY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_MONTH = re.compile(r"^(\d{1,2})/(\d{4})$")
_MONTHS = re.compile(r"^(\d{1,2})-(\d{1,2})/(\d{4})$")
_SPAN = re.compile(r"^(\d{1,2})/(\d{4})\s*-\s*(\d{1,2})/(\d{4})$")
_YEAR = re.compile(r"^(\d{4})$")


def _month(year, month, last=False):
    if not 1 <= month <= 12 or not 1000 <= year <= 9999:
        raise ValueError

    return date(year, month, calendar.monthrange(year, month)[1] if last else 1)


def _collapsed(value) -> str:
    return " ".join(str(value).split())


def _iso_day(parts, _day_first):
    day = date(*parts)

    return day, day, "day"


def _numeric_day(parts, day_first):
    if day_first is None:
        return None  # 03/04/2021 es ambiguo sin evidencia del orden en la columna.

    first, second, year = parts
    day, month = (first, second) if day_first else (second, first)
    value = date(year, month, day)

    return value, value, "day"


def _single_month(parts, _day_first):
    month, year = parts

    return _month(year, month), _month(year, month, True), "month"


def _month_range(parts, _day_first):
    start, end, year = parts
    if end < start:
        return None

    return _month(year, start), _month(year, end, True), "months"


def _month_span(parts, _day_first):
    start_month, start_year, end_month, end_year = parts
    start = _month(start_year, start_month)
    end = _month(end_year, end_month, True)

    return (start, end, "months") if start <= end else None


def _whole_year(parts, _day_first):
    (year,) = parts

    return date(year, 1, 1), date(year, 12, 31), "year"


_FORMATS = (
    (_ISO, _iso_day),
    (_DAY, _numeric_day),
    (_MONTH, _single_month),
    (_MONTHS, _month_range),
    (_SPAN, _month_span),
    (_YEAR, _whole_year),
)


def period_bounds(value: str, day_first: bool | None = None) -> tuple[date, date, str] | None:
    """Devuelve (inicio, fin, precisión) o None si el texto no nombra un período inequívoco."""
    text = _collapsed(value)

    for pattern, interpret in _FORMATS:
        match = pattern.match(text)
        if match is None:
            continue
        try:
            return interpret([int(part) for part in match.groups()], day_first)
        except ValueError:
            return None

    return None


def day_order(values) -> bool | None:
    """True si la columna demuestra día/mes, False si demuestra mes/día, None si no lo prueba."""
    pairs = _leading_pairs(values)
    day_first = any(first > 12 for first, _ in pairs)
    month_first = any(second > 12 for _, second in pairs)

    return None if day_first == month_first else day_first


def _leading_pairs(values) -> list[tuple[int, int]]:
    pairs = []

    for value in values:
        match = _DAY.match(_collapsed(value))
        if match:
            first, second, _year = match.groups()
            pairs.append((int(first), int(second)))

    return pairs


def profile_temporal_text(values, *, min_coverage=0.9):
    distinct = [
        value for value in dict.fromkeys(values) if isinstance(value, str) and value.strip()
    ]
    if len(distinct) < 2:
        return None

    order = day_order(distinct)
    parsed = {value: period_bounds(value, order) for value in distinct}
    interpreted = [value for value, bounds in parsed.items() if bounds is not None]
    if len(interpreted) / len(distinct) < min_coverage:
        return None

    return {
        "day_first": order,
        "interpreted_values": len(interpreted),
        "uninterpreted_values": [value for value, bounds in parsed.items() if bounds is None][:10],
        "precisions": sorted({parsed[value][2] for value in interpreted}),
        # Fechas AAAA-MM-DD junto a D/M/AAAA delatan días y meses invertidos al leer la planilla: sus valores no
        # acotan períodos pedidos.
        "values": [] if _mixed_day_notations(distinct) else interpreted,
    }


def _mixed_day_notations(values) -> bool:
    texts = [_collapsed(value) for value in values]
    return any(_ISO.match(text) for text in texts) and any(_DAY.match(text) for text in texts)


def within_period(value: str, day_first: bool | None, start, end) -> bool:
    """El período que nombra el valor cae entero dentro del pedido; un extremo None queda abierto: «12/2009-01/2010»
    no está dentro de 2009."""
    bounds = period_bounds(value, day_first)
    if bounds is None:
        return False

    first, last, _precision = bounds
    return (start is None or first >= start) and (end is None or last <= end)
