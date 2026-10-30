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


def period_bounds(value: str, day_first: bool | None = None) -> tuple[date, date, str] | None:
    """Devuelve (inicio, fin, precisión) o None si el texto no nombra un período inequívoco."""
    text = " ".join(str(value).split())
    try:
        if match := _ISO.match(text):
            day = date(*(int(part) for part in match.groups()))
            return day, day, "day"
        if match := _DAY.match(text):
            if day_first is None:
                return None  # 03/04/2021 es ambiguo sin evidencia del orden en la columna.
            first, second, year = (int(part) for part in match.groups())
            day, month = (first, second) if day_first else (second, first)
            value = date(year, month, day)
            return value, value, "day"
        if match := _MONTH.match(text):
            month, year = (int(part) for part in match.groups())
            return _month(year, month), _month(year, month, True), "month"
        if match := _MONTHS.match(text):
            start, end, year = (int(part) for part in match.groups())
            if end < start:
                return None
            return _month(year, start), _month(year, end, True), "months"
        if match := _SPAN.match(text):
            start_month, start_year, end_month, end_year = (int(part) for part in match.groups())
            start, end = _month(start_year, start_month), _month(end_year, end_month, True)
            return (start, end, "months") if start <= end else None
        if match := _YEAR.match(text):
            year = int(match.group(1))
            return date(year, 1, 1), date(year, 12, 31), "year"
    except ValueError:
        return None
    return None


def day_order(values) -> bool | None:
    """True si la columna demuestra día/mes, False si demuestra mes/día, None si no lo prueba."""
    pairs = [tuple(int(part) for part in match.groups()[:2])
             for value in values if (match := _DAY.match(" ".join(str(value).split())))]
    day_first = any(first > 12 for first, _ in pairs)
    month_first = any(second > 12 for _, second in pairs)
    return None if day_first == month_first else day_first


def profile_temporal_text(values, *, min_coverage=0.9):
    """Perfil de una columna TEXT si casi todos sus valores distintos nombran períodos."""
    distinct = [value for value in dict.fromkeys(values) if isinstance(value, str) and value.strip()]
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
    }
