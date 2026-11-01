import re
from decimal import ROUND_HALF_UP, Decimal, localcontext
from html import escape

from aquai.application.units import repaired_symbols

_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
# Sólo exponentes negativos tras un símbolo (mL-1): PO4 o N-NO3 son fórmulas, no potencias.
_NEGATIVE_EXPONENT = re.compile(r"(?<=[A-Za-zµμ])-(\d{1,2})(?![\d.,])")


def format_unit(unit: str) -> str:
    return _NEGATIVE_EXPONENT.sub(_superscript_exponent, repaired_symbols(unit))


def _superscript_exponent(match):
    return "⁻" + match.group(1).translate(_SUPERSCRIPTS)


def format_number(value: Decimal | int | float, language="es") -> str:
    # No interpreta separadores regionales ni elimina puntos como si fueran miles.
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    if not number.is_finite():
        raise ValueError("La presentación requiere un número finito.")
    if number.is_zero():
        return "0"

    mantissa, exponent = _rounded(number)
    integer, _, fraction = mantissa.partition(".")
    decimal, thousands = (",", ".") if language == "es" else (".", ",")
    # Como en la norma tipográfica, los enteros de cuatro cifras (p. ej., años) no se agrupan.
    if not exponent and len(integer.lstrip("-")) >= 5:
        integer = f"{int(integer):,}".replace(",", thousands)

    mantissa = integer + (decimal + fraction if fraction else "")

    return mantissa + (f"e{int(exponent[0])}" if exponent else "")


def _rounded(number):
    magnitude = number.adjusted()
    # Por debajo de 0,01 se conservan dos cifras significativas; nunca se
    # transforma un valor pequeño distinto de cero en un cero de presentación.
    places = max(2, 1 - magnitude) if magnitude < -2 else 2
    with localcontext() as context:
        context.prec = max(28, len(number.as_tuple().digits), magnitude + places + 2)
        rounded = number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        text = format(rounded, "e" if magnitude < -6 else "f")

    mantissa, *exponent = text.split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")

    return mantissa, exponent


def chart_hover_templates(series, *, histogram=False):
    if histogram:
        return [_bin_template(item) for item in series.bins]

    return [
        f"{_hover_value(point.x)}<br>{_hover_value(point.y)}<extra></extra>"
        for point in series.points
    ]


def _bin_template(item):
    lower = format_number(item.lower)
    upper = format_number(item.upper)
    if lower == upper and item.lower != item.upper:
        # No colapsar dos límites diferentes al abreviar un intervalo.
        lower, upper = (str(value).replace(".", ",") for value in (item.lower, item.upper))
    closing = "]" if item.upper_inclusive else ")"

    return f"[{lower}; {upper}{closing}<br>Frecuencia: {item.count}<extra></extra>"


def _hover_value(value):
    if value is None:
        return "Sin dato"
    if type(value) in {int, float}:
        return format_number(value)

    text = str(value).lower() if type(value) is bool else str(value)

    return escape(text).replace("%", "&#37;")


def chart_axis_formats(chart):
    x_values, y_values = _axis_values(chart)
    axes = (("xaxis", chart.x_axis.field, x_values), ("yaxis", chart.y_axis.field, y_values))
    result = {}
    for key, field, values in axes:
        tick_format = _tick_format(field.sql_type, values)
        if tick_format is not None:
            result[key] = tick_format

    return result


def _axis_values(chart):
    x_values = []
    y_values = []
    for series in chart.series:
        if chart.chart_type == "histogram":
            for item in series.bins:
                x_values.extend((item.lower, item.upper))
                y_values.append(item.count)
        else:
            for point in series.points:
                x_values.append(point.x)
                y_values.append(point.y)

    return x_values, y_values


def _tick_format(sql_type, values):
    if sql_type == "BIGINT":
        return "d"
    if sql_type != "NUMERIC":
        return None

    numbers = [
        abs(Decimal(str(value))) for value in values if type(value) in {int, float} and value != 0
    ]
    magnitude = max(numbers).adjusted() if numbers else 0
    if magnitude < -6:
        return ".1~e"
    if magnitude < -2:
        return f".{max(2, 1 - magnitude)}~f"

    return ".2~f"
