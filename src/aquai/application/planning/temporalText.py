import calendar
import re
from datetime import date

_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T]00:00(?::00)?)?$")
_DAY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_MONTH = re.compile(r"^(\d{1,2})/(\d{4})$")
_MONTHS = re.compile(r"^(\d{1,2})-(\d{1,2})/(\d{4})$")
_SPAN = re.compile(r"^(\d{1,2})/(\d{4})\s*-\s*(\d{1,2})/(\d{4})$")
_YEAR = re.compile(r"^(\d{4})$")


#
# Devuelve el primer o último día de un mes, validando mes y año.
#
def _month(year, month, last=False):
    if not 1 <= month <= 12 or not 1000 <= year <= 9999:
        raise ValueError

    return date(year, month, calendar.monthrange(year, month)[1] if last else 1)


#
# Devuelve el texto con los espacios repetidos colapsados.
#
def _collapsed(value) -> str:
    return " ".join(str(value).split())


#
# Interpreta una fecha AAAA-MM-DD como un período de un día.
#
def _iso_day(parts, _day_first):
    day = date(*parts)

    return day, day, "day"


#
# Interpreta una fecha D/M/AAAA como un día, según el orden día/mes de la columna.
#
def _numeric_day(parts, day_first):
    if day_first is None:
        return None

    first, second, year = parts
    day, month = (first, second) if day_first else (second, first)
    value = date(year, month, day)

    return value, value, "day"


#
# Interpreta M/AAAA como el período de ese mes.
#
def _single_month(parts, _day_first):
    month, year = parts

    return _month(year, month), _month(year, month, True), "month"


#
# Interpreta M-M/AAAA como un rango de meses de un año.
#
def _month_range(parts, _day_first):
    start, end, year = parts
    if end < start:
        return None

    return _month(year, start), _month(year, end, True), "months"


#
# Interpreta M/AAAA-M/AAAA como un rango de meses entre dos años.
#
def _month_span(parts, _day_first):
    start_month, start_year, end_month, end_year = parts
    start = _month(start_year, start_month)
    end = _month(end_year, end_month, True)

    return (start, end, "months") if start <= end else None


#
# Interpreta un año como el período del año completo.
#
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


#
# Devuelve inicio, fin y precisión del período que nombra un texto, o None si es ambiguo.
#
def period_bounds(value: str, day_first: bool | None = None) -> tuple[date, date, str] | None:
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


#
# Indica si los valores demuestran el orden día/mes (True), mes/día (False) o ninguno (None).
#
def day_order(values) -> bool | None:
    pairs = _leading_pairs(values)
    day_first = any(first > 12 for first, _ in pairs)
    month_first = any(second > 12 for _, second in pairs)

    return None if day_first == month_first else day_first


#
# Devuelve los dos primeros números de cada fecha D/M/AAAA.
#
def _leading_pairs(values) -> list[tuple[int, int]]:
    pairs = []

    for value in values:
        match = _DAY.match(_collapsed(value))
        if match:
            first, second, _year = match.groups()
            pairs.append((int(first), int(second)))

    return pairs


#
# Perfila una columna TEXT de períodos si la mayoría de sus valores es interpretable.
#
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
        "values": [] if _mixed_day_notations(distinct) else interpreted,
    }


#
# Indica si los valores mezclan fechas AAAA-MM-DD con fechas D/M/AAAA.
#
def _mixed_day_notations(values) -> bool:
    texts = [_collapsed(value) for value in values]
    return any(_ISO.match(text) for text in texts) and any(_DAY.match(text) for text in texts)


#
# Indica si el período que nombra el valor cae entero dentro del rango pedido.
#
def within_period(value: str, day_first: bool | None, start, end) -> bool:
    bounds = period_bounds(value, day_first)
    if bounds is None:
        return False

    first, last, _precision = bounds
    return (start is None or first >= start) and (end is None or last <= end)
