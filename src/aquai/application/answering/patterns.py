from dataclasses import dataclass
from statistics import StatisticsError, correlation

_MIN_CORRELATION_POINTS = 3


@dataclass(frozen=True)
class PlottedPoint:
    """Un punto con valor del gráfico, con la serie a la que pertenece."""

    x: str | int | float | bool
    y: float
    series: str | None


#
# Devuelve los puntos del gráfico que tienen valor, en el orden en que se dibujan.
#
def plotted_points(chart) -> list[PlottedPoint]:
    return [
        PlottedPoint(point.x, point.y, series.name)
        for series in chart.series
        for point in series.points
        if point.y is not None
    ]


#
# Devuelve el punto más alto y el más bajo, o None si todos los valores son iguales.
#
def extremes(points) -> tuple[PlottedPoint, PlottedPoint] | None:
    highest = max(points, key=lambda item: item.y)
    lowest = min(points, key=lambda item: item.y)

    return (highest, lowest) if highest.y != lowest.y else None


#
# Devuelve el coeficiente de correlación de Pearson entre X e Y y la cantidad de puntos con que se calculó,
# o None si hay pocos puntos o alguna de las variables no varía.
#
def pearson(points) -> tuple[float, int] | None:
    pairs = [(float(item.x), item.y) for item in points if type(item.x) in {int, float}]
    if len(pairs) < _MIN_CORRELATION_POINTS:
        return None

    try:
        coefficient = correlation(*zip(*pairs, strict=True))
    except StatisticsError:
        return None

    return coefficient, len(pairs)


#
# Devuelve el intervalo del histograma con más valores, o None si varios empatan.
#
def modal_bin(chart):
    bins = [item for series in chart.series for item in series.bins]
    top = max(bins, key=lambda item: item.count)
    tied = sum(item.count == top.count for item in bins) > 1

    return None if tied else top
