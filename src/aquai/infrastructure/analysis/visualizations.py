"""Gráficos completos de Plotly derivados de filas reales y verificadas."""

from bisect import bisect_right
from datetime import date, datetime
from decimal import Decimal, localcontext
from math import isfinite

from aquai.application.contracts.execution import (
    ChartAxis, ChartPoint, ChartSeries, DatasetVersionReference, HistogramBin, VisualizationResult,
)
from aquai.application.contracts.planning import OutputField
from aquai.infrastructure.analysis.operations import AnalysisBlocked
from aquai.infrastructure.analysis.plotlyFigures import build_plotly_figure


def _display_number(value):
    number = float(value)
    if not isfinite(number) or (number == 0 and value != 0):
        raise AnalysisBlocked("chart_numeric_range", "Un valor no puede representarse como float64 de presentación sin desbordar o perder su magnitud.")
    return number


def _display_x(value):
    if isinstance(value, (Decimal, int)) and not isinstance(value, bool):
        return _display_number(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def create_visualization(step, rows, parent_fields, datasets, settings, budget, *, contextual_warnings=None):
    request = step.visualization
    fields = {field.name: field for field in parent_fields}
    x = fields[request.x.field]
    y = fields[request.y.field] if request.y else OutputField(
        name="count", sql_type="BIGINT", numeric_eligible=True, unit=None,
        unit_status="dimensionless", sources=x.sources)
    series_field = fields[request.series.field] if request.series else None
    warnings = [*(contextual_warnings or []), *(warning for item in datasets for warning in item.warnings),
                "Los números del gráfico se convierten explícitamente a float64 sólo para presentación; los resultados analíticos conservan Decimal."]
    warnings.extend(f"La unidad de {field.name} no está documentada; se conserva como desconocida."
                    for field in (x, y) if field.unit_status == "unknown" and field.sql_type in {"BIGINT", "NUMERIC"})
    series = []
    excluded = 0
    if request.chart_type == "histogram":
        values = []
        for row in rows:
            budget.check()
            value = row[x.name]
            if value is None:
                excluded += 1
            else:
                values.append(Decimal(value))
        if not values:
            raise AnalysisBlocked("no_chart_data", "No hay valores numéricos válidos para representar una distribución.")
        with localcontext() as context:
            context.prec = settings.decimal_precision
            lower, upper = min(values), max(values)
            count = request.bins if lower != upper else 1
            edges = [lower + (upper - lower) * Decimal(index) / Decimal(count) for index in range(count)] + [upper]
            frequencies = [0] * count
            for value in values:
                budget.check()
                index = min(count - 1, max(0, bisect_right(edges, value) - 1))
                frequencies[index] += 1
            bins = []
            for index in range(count):
                start, end = _display_number(edges[index]), _display_number(edges[index + 1])
                if lower != upper and start >= end:
                    raise AnalysisBlocked("chart_numeric_range", "Los intervalos distintos no se distinguen en float64; no se publica un histograma engañoso.")
                bins.append(HistogramBin(lower=start, upper=end, count=frequencies[index], upper_inclusive=index == count - 1))
            series = [ChartSeries(bins=bins)]
        warnings.append("Intervalos de igual anchura: [inferior, superior), con el extremo máximo incluido en el último; variable constante usa un único intervalo puntual.")
    else:
        grouped = {}
        for row in rows:
            budget.check()
            xv, yv = row[x.name], row[y.name]
            if xv is None or (yv is None and request.null_policy == "exclude"):
                excluded += 1
                continue
            key = row[series_field.name] if series_field else None
            grouped.setdefault(key, []).append((xv, yv))
        if not grouped:
            raise AnalysisBlocked("no_chart_data", "No hay coordenadas válidas para el gráfico.")
        if len(grouped) > settings.max_chart_series:
            raise AnalysisBlocked("chart_series_limit", "La cantidad de series supera el límite; agregá o acotá explícitamente el plan.")
        for key, values in grouped.items():
            budget.check()
            if request.chart_type in {"bar", "line"}:
                if len({value[0] for value in values}) != len(values):
                    raise AnalysisBlocked("ambiguous_chart_granularity", "Hay varias observaciones para una categoría/instante de la serie; el plan debe agregar o desambiguar antes de graficar.")
            if request.chart_type == "line":
                values.sort(key=lambda pair: pair[0])
            points = [ChartPoint(x=_display_x(xv), y=None if yv is None else _display_number(yv)) for xv, yv in values]
            if request.chart_type in {"bar", "line"} and len({point.x for point in points}) != len(points):
                raise AnalysisBlocked("chart_numeric_range", "Coordenadas distintas colapsan al convertirlas para presentación; el gráfico necesita otro eje.")
            name = key.isoformat() if isinstance(key, (date, datetime)) else str(key)
            series.append(ChartSeries(name=None if key is None else name, points=points))
    size = sum(len(item.points) + len(item.bins) for item in series)
    if size > settings.max_chart_points:
        raise AnalysisBlocked("chart_points_limit", "El gráfico completo excede el límite de puntos; no se recorta ni se muestrea implícitamente.")
    if excluded:
        warnings.append(f"Se excluyeron {excluded} filas sin coordenadas válidas; los valores faltantes no se sustituyeron por cero.")
    result = VisualizationResult(
        operation_id=step.id, chart_type=request.chart_type, title=request.title,
        x_axis=ChartAxis(field=x, label=request.x_label), y_axis=ChartAxis(field=y, label=request.y_label),
        series_field=series_field, series=series,
        datasets=[DatasetVersionReference(dataset_id=item.dataset_id, source_id=item.source_id, version=item.version) for item in datasets],
        input_operation_ids=list(step.depends_on), input_rows=len(rows), excluded_rows=excluded,
        warnings=list(dict.fromkeys(warnings)),
    )
    budget.check()
    result.figure = build_plotly_figure(result)
    budget.check()
    return result
