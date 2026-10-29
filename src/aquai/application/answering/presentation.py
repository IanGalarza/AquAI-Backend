"""Formato de números tipados para lectura, sin alterar la evidencia exacta."""

from decimal import Decimal, ROUND_HALF_UP, localcontext
from html import escape


def format_number(value: Decimal | int | float, language="es") -> str:
    # No interpreta separadores regionales ni elimina puntos como si fueran miles.
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    if not number.is_finite():
        raise ValueError("La presentación requiere un número finito.")
    if number.is_zero():
        return "0"
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
    if language == "es":
        mantissa = mantissa.replace(".", ",")
    return mantissa + (f"e{int(exponent[0])}" if exponent else "")


def chart_hover_templates(series, *, histogram=False):
    if histogram:
        templates = []
        for item in series.bins:
            lower, upper = format_number(item.lower), format_number(item.upper)
            if lower == upper and item.lower != item.upper:
                # No colapsar dos límites diferentes al abreviar un intervalo.
                lower, upper = (str(value).replace(".", ",") for value in (item.lower, item.upper))
            templates.append(f"[{lower}; {upper}{']' if item.upper_inclusive else ')'}"
                             f"<br>Frecuencia: {item.count}<extra></extra>")
        return templates

    def value_text(value):
        if value is None:
            return "Sin dato"
        if type(value) in {int, float}:
            return format_number(value)
        return escape(str(value).lower() if type(value) is bool else str(value)).replace("%", "&#37;")

    return [f"{value_text(point.x)}<br>{value_text(point.y)}<extra></extra>" for point in series.points]


def chart_axis_formats(chart):
    if chart.chart_type == "histogram":
        x = [value for series in chart.series for item in series.bins for value in (item.lower, item.upper)]
        y = [item.count for series in chart.series for item in series.bins]
    else:
        x = [point.x for series in chart.series for point in series.points]
        y = [point.y for series in chart.series for point in series.points]
    result = {}
    for key, field, values in (("xaxis", chart.x_axis.field, x), ("yaxis", chart.y_axis.field, y)):
        if field.sql_type == "BIGINT":
            result[key] = "d"
        elif field.sql_type == "NUMERIC":
            numbers = [abs(Decimal(str(value))) for value in values if type(value) in {int, float} and value != 0]
            magnitude = max(numbers).adjusted() if numbers else 0
            result[key] = ".1~e" if magnitude < -6 else f".{max(2, 1 - magnitude)}~f" if magnitude < -2 else ".2~f"
    return result
