from bisect import bisect_right
from datetime import date, datetime
from decimal import Decimal, localcontext
from math import isfinite

from aquai.application.contracts.execution import (
    ChartAxis,
    ChartPoint,
    ChartSeries,
    DatasetVersionReference,
    HistogramBin,
    VisualizationResult,
)
from aquai.application.contracts.planning import OutputField
from aquai.application.errors import AnalysisBlocked
from aquai.application.units import presentation_unit_status
from aquai.infrastructure.analysis.plotlyFigures import build_plotly_figure


def _display_number(value):
    number = float(value)

    if not isfinite(number) or (number == 0 and value != 0):
        raise AnalysisBlocked(
            "chart_numeric_range",
            "Un valor no puede representarse como float64 de presentación sin desbordar o perder su magnitud.",
        )

    return number


def _display_x(value):
    if isinstance(value, (Decimal, int)) and not isinstance(value, bool):
        return _display_number(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return value


def create_visualization(
    step, rows, parent_fields, datasets, settings, budget, *, contextual_warnings=None
):
    request = step.visualization
    fields = {field.name: field for field in parent_fields}
    x = fields[request.x.field]
    y = fields[request.y.field] if request.y else _count_field(x)
    series_field = fields[request.series.field] if request.series else None
    warnings = _initial_warnings(x, y, datasets, contextual_warnings)

    if request.chart_type == "histogram":
        series, excluded = _histogram_series(rows, x, request, settings, budget)
        warnings.append(
            "Intervalos de igual anchura: [inferior, superior), con el extremo máximo incluido en el último; variable constante usa un único intervalo puntual."
        )
    else:
        series, excluded = _cartesian_series(rows, x, y, series_field, request, settings, budget)

    _check_point_limit(series, settings)

    if excluded:
        warnings.append(
            f"Se excluyeron {excluded} filas sin coordenadas válidas; los valores faltantes no se sustituyeron por cero."
        )

    result = VisualizationResult(
        operation_id=step.id,
        chart_type=request.chart_type,
        title=request.title,
        x_axis=ChartAxis(field=x, label=request.x_label),
        y_axis=ChartAxis(field=y, label=request.y_label),
        series_field=series_field,
        series=series,
        datasets=_dataset_references(datasets),
        input_operation_ids=list(step.depends_on),
        input_rows=len(rows),
        excluded_rows=excluded,
        warnings=list(dict.fromkeys(warnings)),
    )

    budget.check()
    result.figure = build_plotly_figure(result)
    budget.check()

    return result


def _count_field(x):
    return OutputField(
        name="count",
        sql_type="BIGINT",
        numeric_eligible=True,
        unit=None,
        unit_status="dimensionless",
        sources=x.sources,
    )


def _initial_warnings(x, y, datasets, contextual_warnings):
    warnings = list(contextual_warnings or [])

    for item in datasets:
        warnings.extend(item.warnings)

    warnings.append(
        "Los números del gráfico se convierten explícitamente a float64 sólo para presentación; los resultados analíticos conservan Decimal."
    )

    for field in (x, y):
        if presentation_unit_status(field) == "unknown" and field.sql_type in {"BIGINT", "NUMERIC"}:
            warnings.append(
                f"La unidad de {field.name} no está documentada; se conserva como desconocida."
            )

    return warnings


def _histogram_series(rows, x, request, settings, budget):
    values = []
    excluded = 0

    for row in rows:
        budget.check()
        value = row[x.name]

        if value is None:
            excluded += 1
        else:
            values.append(Decimal(value))

    if not values:
        raise AnalysisBlocked(
            "no_chart_data",
            "No hay valores numéricos válidos para representar una distribución.",
        )

    with localcontext() as context:
        context.prec = settings.decimal_precision
        lower, upper = min(values), max(values)
        count = request.bins if lower != upper else 1
        edges = _bin_edges(lower, upper, count)
        frequencies = _bin_frequencies(values, edges, count, budget)
        series = [ChartSeries(bins=_histogram_bins(edges, frequencies, lower, upper))]

    return series, excluded


def _bin_edges(lower, upper, count):
    inner = [lower + (upper - lower) * Decimal(index) / Decimal(count) for index in range(count)]
    return inner + [upper]


def _bin_frequencies(values, edges, count, budget):
    frequencies = [0] * count

    for value in values:
        budget.check()
        index = min(count - 1, max(0, bisect_right(edges, value) - 1))
        frequencies[index] += 1

    return frequencies


def _histogram_bins(edges, frequencies, lower, upper):
    bins = []
    count = len(frequencies)

    for index in range(count):
        start = _display_number(edges[index])
        end = _display_number(edges[index + 1])

        if lower != upper and start >= end:
            raise AnalysisBlocked(
                "chart_numeric_range",
                "Los intervalos distintos no se distinguen en float64; no se publica un histograma engañoso.",
            )

        bins.append(
            HistogramBin(
                lower=start,
                upper=end,
                count=frequencies[index],
                upper_inclusive=index == count - 1,
            )
        )

    return bins


def _cartesian_series(rows, x, y, series_field, request, settings, budget):
    grouped, excluded = _group_coordinates(rows, x, y, series_field, request, budget)

    if not grouped:
        raise AnalysisBlocked("no_chart_data", "No hay coordenadas válidas para el gráfico.")

    if len(grouped) > settings.max_chart_series:
        raise AnalysisBlocked(
            "chart_series_limit",
            "La cantidad de series supera el límite; agregá o acotá explícitamente el plan.",
        )

    series = []

    for key, values in grouped.items():
        budget.check()
        series.append(_chart_series(key, values, request))

    return series, excluded


def _group_coordinates(rows, x, y, series_field, request, budget):
    grouped = {}
    excluded = 0

    for row in rows:
        budget.check()
        x_value, y_value = row[x.name], row[y.name]

        if x_value is None or (y_value is None and request.null_policy == "exclude"):
            excluded += 1
            continue

        key = row[series_field.name] if series_field else None
        grouped.setdefault(key, []).append((x_value, y_value))

    return grouped, excluded


def _chart_series(key, values, request):
    one_point_per_x = request.chart_type in {"bar", "line"}

    if one_point_per_x and len({value[0] for value in values}) != len(values):
        raise AnalysisBlocked(
            "ambiguous_chart_granularity",
            "Hay varias observaciones para una categoría/instante de la serie; el plan debe agregar o desambiguar antes de graficar.",
        )

    if request.chart_type == "line":
        values.sort(key=lambda pair: pair[0])

    points = [
        ChartPoint(x=_display_x(x_value), y=None if y_value is None else _display_number(y_value))
        for x_value, y_value in values
    ]

    if one_point_per_x and len({point.x for point in points}) != len(points):
        raise AnalysisBlocked(
            "chart_numeric_range",
            "Coordenadas distintas colapsan al convertirlas para presentación; el gráfico necesita otro eje.",
        )

    name = key.isoformat() if isinstance(key, (date, datetime)) else str(key)

    return ChartSeries(name=None if key is None else name, points=points)


def _check_point_limit(series, settings):
    size = sum(len(item.points) + len(item.bins) for item in series)

    if size > settings.max_chart_points:
        raise AnalysisBlocked(
            "chart_points_limit",
            "El gráfico completo excede el límite de puntos; no se recorta ni se muestrea implícitamente.",
        )


def _dataset_references(datasets):
    return [
        DatasetVersionReference(
            dataset_id=item.dataset_id, source_id=item.source_id, version=item.version
        )
        for item in datasets
    ]
