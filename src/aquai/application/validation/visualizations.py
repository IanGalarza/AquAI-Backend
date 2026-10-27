"""Coherencia de figuras existentes; no construye gráficos ni recalcula estadísticas."""

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from html import escape
from math import isfinite

from aquai.application.contracts.planning import OutputField
from aquai.application.errors import PlannerValidationError


def _check(condition, message):
    if not condition:
        raise PlannerValidationError(message)


def _display(value):
    if isinstance(value, (int, Decimal)) and type(value) is not bool:
        number = float(value)
        _check(isfinite(number) and (number != 0 or value == 0), "Coordenada fuera del rango de presentación.")
        return number
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def validate_visualization(chart, step, parent, rows, sources):
    request = step.visualization
    fields = {field.name: field for field in parent.columns}
    x = fields[request.x.field]
    y = fields[request.y.field] if request.y is not None else OutputField(
        name="count", sql_type="BIGINT", numeric_eligible=True, unit=None,
        unit_status="dimensionless", sources=x.sources)
    series_field = fields[request.series.field] if request.series is not None else None
    _check(chart.operation_id == step.id and chart.chart_type == request.chart_type
           and chart.input_operation_ids == step.depends_on and chart.title == request.title,
           "El gráfico no corresponde a la operación planificada.")
    _check(chart.x_axis.field == x and chart.y_axis.field == y and chart.series_field == series_field
           and chart.x_axis.label == request.x_label and chart.y_axis.label == request.y_label,
           "Las variables o unidades del gráfico no corresponden al resultado autorizado.")
    expected_sources = {(item.dataset_id, item.source_id, item.version) for item in sources}
    actual_sources = {(item.dataset_id, item.source_id, item.version) for item in chart.datasets}
    _check(actual_sources == expected_sources and len(actual_sources) == len(chart.datasets),
           "La procedencia del gráfico no coincide con sus entradas.")
    _check(chart.input_rows == len(rows) and 0 <= chart.excluded_rows <= len(rows), "El gráfico informa cantidades incoherentes.")
    if request.chart_type == "histogram":
        _check(len(chart.series) == 1 and bool(chart.series[0].bins) and not chart.series[0].points,
               "El histograma no conserva sus intervalos.")
        bins = chart.series[0].bins
        valid = [row[x.name] for row in rows if row[x.name] is not None]
        _check(chart.excluded_rows == len(rows) - len(valid) and sum(item.count for item in bins) == len(valid),
               "Las frecuencias y exclusiones del histograma no conservan las observaciones.")
        _check(len(bins) == request.bins or (len(bins) == 1 and bins[0].lower == bins[0].upper),
               "La cantidad de intervalos no corresponde al plan.")
        for index, item in enumerate(bins):
            _check(item.lower <= item.upper and item.upper_inclusive == (index == len(bins) - 1),
                   "Los extremos del histograma son incoherentes.")
            if index:
                _check(bins[index - 1].upper == item.lower, "Los intervalos del histograma no son contiguos.")
        _check(all(bins[0].lower <= float(value) <= bins[-1].upper for value in valid),
               "El histograma no cubre las observaciones de entrada.")
    else:
        expected = Counter()
        for row in rows:
            xv, yv = row[x.name], row[y.name]
            if xv is None or (yv is None and request.null_policy == "exclude"):
                continue
            key = row[series_field.name] if series_field is not None else None
            name = key.isoformat() if isinstance(key, (date, datetime)) else str(key) if key is not None else None
            expected[(name, _display(xv), None if yv is None else float(yv))] += 1
        actual = Counter((series.name, point.x, point.y) for series in chart.series for point in series.points)
        _check(expected == actual and chart.excluded_rows == len(rows) - sum(actual.values()),
               "Las coordenadas del gráfico no conservan los resultados reales.")
        _check(len({series.name for series in chart.series}) == len(chart.series)
               and all(not series.bins for series in chart.series), "El gráfico repite series o contiene datos ajenos.")
        if request.chart_type in {"bar", "line"}:
            _check(all(len({point.x for point in series.points}) == len(series.points) for series in chart.series),
                   "El gráfico tiene una granularidad ambigua.")
        if request.chart_type == "line":
            _check(all([point.x for point in series.points] == sorted(point.x for point in series.points)
                       for series in chart.series), "La serie de líneas no está ordenada.")

    figure = chart.figure
    _check(figure is not None and len(figure.data) == len(chart.series), "Falta la figura Plotly completa.")
    for trace, series in zip(figure.data, chart.series, strict=True):
        allowed = {"type", "x", "y", "name", "marker", "line", "mode", "connectgaps", "width", "customdata", "hovertemplate"}
        _check(set(trace).issubset(allowed), "La figura contiene transformaciones o trazas no autorizadas.")
        _check(trace.get("type") == ("bar" if request.chart_type in {"bar", "histogram"} else "scatter"),
               "La figura cambia el tipo de representación.")
        _check(isinstance(trace.get("marker", {}).get("color"), str), "La figura agrega una codificación de color no documentada.")
        if request.chart_type == "histogram":
            nonconstant = any(item.lower != item.upper for item in series.bins)
            xv = [item.lower / 2 + item.upper / 2 for item in series.bins] if nonconstant else [format(item.lower, ".17g") for item in series.bins]
            _check(trace.get("x") == xv and trace.get("y") == [item.count for item in series.bins],
                   "Las barras Plotly no corresponden al histograma calculado.")
            if nonconstant:
                _check(trace.get("width") == [item.upper - item.lower for item in series.bins], "Plotly cambia la anchura de los intervalos.")
            _check(trace.get("customdata") == [[item.lower, item.upper, "]" if item.upper_inclusive else ")"] for item in series.bins],
                   "Plotly cambia los extremos de los intervalos.")
        else:
            xv = [str(point.x).lower() if type(point.x) is bool else point.x for point in series.points]
            _check(trace.get("x") == xv and trace.get("y") == [point.y for point in series.points], "La figura Plotly modifica los valores.")
            if request.chart_type in {"line", "scatter"}:
                _check(trace.get("mode") == ("lines+markers" if request.chart_type == "line" else "markers")
                       and trace.get("connectgaps") is False, "La figura introduce otra transformación visual.")
                _check(not isinstance(trace.get("marker", {}).get("size"), list), "La figura agrega una codificación de tamaño no documentada.")
            if request.chart_type == "line":
                _check(trace.get("line", {}).get("shape") == "linear", "La figura introduce una interpolación no autorizada.")
    layout, meta = figure.layout, figure.layout.get("meta", {})
    _check(layout.get("title", {}).get("text") == escape(chart.title) and not layout.get("shapes"),
           "La figura contiene títulos u objetos adicionales no verificados.")
    _check(layout.get("yaxis", {}).get("type") == "linear", "La figura cambia la escala planificada.")
    x_type = ("date" if x.sql_type in {"DATE", "TIMESTAMP"} else "category" if x.sql_type in {"TEXT", "BOOLEAN"} else "linear")
    if request.chart_type == "histogram" and all(item.lower == item.upper for series in chart.series for item in series.bins):
        x_type = "category"
    _check(layout.get("xaxis", {}).get("type") == x_type, "La figura cambia la escala horizontal.")
    for key, axis in (("xaxis", chart.x_axis), ("yaxis", chart.y_axis)):
        label = escape(axis.label)
        if axis.field.unit_status == "documented" and axis.field.unit is not None:
            label += f" [{escape(axis.field.unit)}]"
        elif axis.field.unit_status == "unknown" and axis.field.sql_type in {"BIGINT", "NUMERIC"}:
            label += " (unidad no documentada)"
        _check(layout.get(key, {}).get("title", {}).get("text") == label, "El eje de la figura modifica las unidades documentadas.")
    for key, value in {
        "operation_id": chart.operation_id, "chart_type": chart.chart_type,
        "x_axis": chart.x_axis.model_dump(mode="json"), "y_axis": chart.y_axis.model_dump(mode="json"),
        "datasets": [item.model_dump(mode="json") for item in chart.datasets],
        "input_operation_ids": chart.input_operation_ids, "input_rows": chart.input_rows,
        "excluded_rows": chart.excluded_rows, "numeric_encoding": chart.numeric_encoding,
    }.items():
        _check(meta.get(key) == value, "La figura perdió metadatos de procedencia.")
    _check(set(chart.warnings).issubset(meta.get("warnings", [])), "La figura perdió advertencias relevantes.")
