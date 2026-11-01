from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from html import escape
from math import isfinite

from aquai.application.answering.presentation import (
    chart_axis_formats,
    chart_hover_templates,
)
from aquai.application.contracts.planning import OutputField
from aquai.application.errors import PlannerValidationError
from aquai.application.units import presentation_unit_status

_TRACE_KEYS = {
    "type",
    "x",
    "y",
    "name",
    "marker",
    "line",
    "mode",
    "connectgaps",
    "width",
    "customdata",
    "hovertemplate",
}
_X_AXIS_TYPES = {
    "DATE": "date",
    "TIMESTAMP": "date",
    "TEXT": "category",
    "BOOLEAN": "category",
}


#
# Lanza un error de validación con el mensaje si la condición no se cumple.
#
def _check(condition, message):
    if not condition:
        raise PlannerValidationError(message)


#
# Convierte un valor de coordenada a su forma de presentación, rechazando rangos inválidos.
#
def _display(value):
    if isinstance(value, (int, Decimal)) and type(value) is not bool:
        number = float(value)
        _check(
            isfinite(number) and (number != 0 or value == 0),
            "Coordenada fuera del rango de presentación.",
        )
        return number

    return value.isoformat() if isinstance(value, (date, datetime)) else value


#
# Valida que un gráfico conserve el plan, la procedencia, los datos y la figura.
#
def validate_visualization(chart, step, parent, rows, sources):
    request = step.visualization
    x, y, series_field = _chart_fields(request, parent)

    _check_plan_correspondence(chart, step, request, x, y, series_field)
    _check_provenance(chart, sources)
    _check(
        chart.input_rows == len(rows) and 0 <= chart.excluded_rows <= len(rows),
        "El gráfico informa cantidades incoherentes.",
    )

    if request.chart_type == "histogram":
        _check_histogram(chart, request, x, rows)
    else:
        _check_points(chart, request, x, y, series_field, rows)

    _check_figure(chart, request, x)


#
# Obtiene los campos X, Y y de serie del gráfico a partir del resultado padre.
#
def _chart_fields(request, parent):
    fields = {field.name: field for field in parent.columns}
    x = fields[request.x.field]
    y = fields[request.y.field] if request.y is not None else _count_field(x)
    series_field = fields[request.series.field] if request.series is not None else None

    return x, y, series_field


#
# Construye el campo de conteo que se usa cuando el gráfico no tiene eje Y.
#
def _count_field(x) -> OutputField:
    return OutputField(
        name="count",
        sql_type="BIGINT",
        numeric_eligible=True,
        unit=None,
        unit_status="dimensionless",
        sources=x.sources,
    )


#
# Comprueba que el gráfico corresponda a la operación, los campos y las etiquetas del plan.
#
def _check_plan_correspondence(chart, step, request, x, y, series_field):
    _check(
        chart.operation_id == step.id
        and chart.chart_type == request.chart_type
        and chart.input_operation_ids == step.depends_on
        and chart.title == request.title,
        "El gráfico no corresponde a la operación planificada.",
    )
    _check(
        chart.x_axis.field == x
        and chart.y_axis.field == y
        and chart.series_field == series_field
        and chart.x_axis.label == request.x_label
        and chart.y_axis.label == request.y_label,
        "Las variables o unidades del gráfico no corresponden al resultado autorizado.",
    )


#
# Comprueba que los datasets del gráfico coincidan con sus fuentes de entrada.
#
def _check_provenance(chart, sources):
    expected_sources = {(item.dataset_id, item.source_id, item.version) for item in sources}
    actual_sources = {(item.dataset_id, item.source_id, item.version) for item in chart.datasets}

    _check(
        actual_sources == expected_sources and len(actual_sources) == len(chart.datasets),
        "La procedencia del gráfico no coincide con sus entradas.",
    )


#
# Valida que el histograma conserve intervalos, frecuencias y observaciones de entrada.
#
def _check_histogram(chart, request, x, rows):
    _check(
        len(chart.series) == 1 and bool(chart.series[0].bins) and not chart.series[0].points,
        "El histograma no conserva sus intervalos.",
    )

    bins = chart.series[0].bins
    valid = [row[x.name] for row in rows if row[x.name] is not None]
    _check(
        chart.excluded_rows == len(rows) - len(valid)
        and sum(item.count for item in bins) == len(valid),
        "Las frecuencias y exclusiones del histograma no conservan las observaciones.",
    )
    _check(
        len(bins) == request.bins or (len(bins) == 1 and bins[0].lower == bins[0].upper),
        "La cantidad de intervalos no corresponde al plan.",
    )

    _check_bin_edges(bins)
    _check(
        all(bins[0].lower <= float(value) <= bins[-1].upper for value in valid),
        "El histograma no cubre las observaciones de entrada.",
    )


#
# Valida que los intervalos del histograma sean ordenados y contiguos.
#
def _check_bin_edges(bins):
    for index, item in enumerate(bins):
        is_last = index == len(bins) - 1
        _check(
            item.lower <= item.upper and item.upper_inclusive == is_last,
            "Los extremos del histograma son incoherentes.",
        )
        if index:
            _check(
                bins[index - 1].upper == item.lower,
                "Los intervalos del histograma no son contiguos.",
            )


#
# Valida que los puntos del gráfico coincidan con las filas de entrada.
#
def _check_points(chart, request, x, y, series_field, rows):
    expected = _expected_points(request, x, y, series_field, rows)
    actual = Counter(
        (series.name, point.x, point.y) for series in chart.series for point in series.points
    )

    _check(
        expected == actual and chart.excluded_rows == len(rows) - sum(actual.values()),
        "Las coordenadas del gráfico no conservan los resultados reales.",
    )
    _check(
        len({series.name for series in chart.series}) == len(chart.series)
        and all(not series.bins for series in chart.series),
        "El gráfico repite series o contiene datos ajenos.",
    )

    if request.chart_type in {"bar", "line"}:
        _check(
            all(_has_unique_x(series) for series in chart.series),
            "El gráfico tiene una granularidad ambigua.",
        )

    if request.chart_type == "line":
        _check(
            all(_is_sorted_by_x(series) for series in chart.series),
            "La serie de líneas no está ordenada.",
        )


#
# Calcula los puntos esperados del gráfico a partir de las filas de entrada.
#
def _expected_points(request, x, y, series_field, rows) -> Counter:
    expected = Counter()

    for row in rows:
        x_value, y_value = row[x.name], row[y.name]
        if x_value is None or (y_value is None and request.null_policy == "exclude"):
            continue

        key = row[series_field.name] if series_field is not None else None
        point = (_series_name(key), _display(x_value), None if y_value is None else float(y_value))
        expected[point] += 1

    return expected


#
# Devuelve el nombre de serie como texto, o None si no hay clave.
#
def _series_name(key):
    if isinstance(key, (date, datetime)):
        return key.isoformat()

    return str(key) if key is not None else None


#
# Indica si cada valor de X aparece una sola vez en la serie.
#
def _has_unique_x(series) -> bool:
    return len({point.x for point in series.points}) == len(series.points)


#
# Indica si los puntos de la serie están ordenados por X.
#
def _is_sorted_by_x(series) -> bool:
    return [point.x for point in series.points] == sorted(point.x for point in series.points)


#
# Valida la figura Plotly: sus trazas coinciden con las series y su diseño con el gráfico.
#
def _check_figure(chart, request, x):
    figure = chart.figure
    _check(
        figure is not None and len(figure.data) == len(chart.series),
        "Falta la figura Plotly completa.",
    )

    for trace, series in zip(figure.data, chart.series, strict=True):
        _check_trace(trace, series, request)

    _check_layout(chart, request, x, figure.layout)


#
# Valida que una traza Plotly sea del tipo autorizado y conserve las etiquetas.
#
def _check_trace(trace, series, request):
    histogram = request.chart_type == "histogram"
    expected_type = "bar" if request.chart_type in {"bar", "histogram"} else "scatter"

    _check(
        set(trace).issubset(_TRACE_KEYS),
        "La figura contiene transformaciones o trazas no autorizadas.",
    )
    _check(trace.get("type") == expected_type, "La figura cambia el tipo de representación.")
    _check(
        isinstance(trace.get("marker", {}).get("color"), str),
        "La figura agrega una codificación de color no documentada.",
    )
    _check(
        trace.get("hovertemplate") == chart_hover_templates(series, histogram=histogram),
        "Las etiquetas del gráfico modifican los valores de presentación autorizados.",
    )

    if histogram:
        _check_histogram_trace(trace, series)
    else:
        _check_point_trace(trace, series, request.chart_type)


#
# Valida que las barras Plotly correspondan a los intervalos del histograma.
#
def _check_histogram_trace(trace, series):
    nonconstant = any(item.lower != item.upper for item in series.bins)
    if nonconstant:
        x_values = [item.lower / 2 + item.upper / 2 for item in series.bins]
    else:
        x_values = [format(item.lower, ".17g") for item in series.bins]

    _check(
        trace.get("x") == x_values and trace.get("y") == [item.count for item in series.bins],
        "Las barras Plotly no corresponden al histograma calculado.",
    )
    if nonconstant:
        _check(
            trace.get("width") == [item.upper - item.lower for item in series.bins],
            "Plotly cambia la anchura de los intervalos.",
        )
    _check(
        trace.get("customdata") == [_interval_edges(item) for item in series.bins],
        "Plotly cambia los extremos de los intervalos.",
    )


#
# Devuelve los extremos de un intervalo y el tipo de cierre superior.
#
def _interval_edges(item) -> list:
    return [item.lower, item.upper, "]" if item.upper_inclusive else ")"]


#
# Valida que una traza de puntos conserve valores, modo y forma de línea.
#
def _check_point_trace(trace, series, chart_type):
    x_values = [_plotly_x(point.x) for point in series.points]
    _check(
        trace.get("x") == x_values and trace.get("y") == [point.y for point in series.points],
        "La figura Plotly modifica los valores.",
    )

    if chart_type in {"line", "scatter"}:
        expected_mode = "lines+markers" if chart_type == "line" else "markers"
        _check(
            trace.get("mode") == expected_mode and trace.get("connectgaps") is False,
            "La figura introduce otra transformación visual.",
        )
        _check(
            not isinstance(trace.get("marker", {}).get("size"), list),
            "La figura agrega una codificación de tamaño no documentada.",
        )

    if chart_type == "line":
        _check(
            trace.get("line", {}).get("shape") == "linear",
            "La figura introduce una interpolación no autorizada.",
        )


#
# Devuelve el valor de X listo para Plotly, con los booleanos como texto.
#
def _plotly_x(value):
    return str(value).lower() if type(value) is bool else value


#
# Valida el diseño de la figura: título, ejes, formatos, procedencia y advertencias.
#
def _check_layout(chart, request, x, layout):
    meta = layout.get("meta", {})
    _check(
        layout.get("title", {}).get("text") == escape(chart.title) and not layout.get("shapes"),
        "La figura contiene títulos u objetos adicionales no verificados.",
    )
    _check(
        layout.get("yaxis", {}).get("type") == "linear", "La figura cambia la escala planificada."
    )

    x_type = _x_axis_type(chart, request, x)
    _check(layout.get("xaxis", {}).get("type") == x_type, "La figura cambia la escala horizontal.")
    _check(
        layout.get("separators") == ",.", "La figura modifica el separador decimal de presentación."
    )

    for key, value in chart_axis_formats(chart).items():
        if key == "xaxis" and x_type != "linear":
            continue
        _check(
            layout.get(key, {}).get("tickformat") == value,
            "El eje modifica la precisión de presentación autorizada.",
        )

    for key, axis in (("xaxis", chart.x_axis), ("yaxis", chart.y_axis)):
        _check(
            layout.get(key, {}).get("title", {}).get("text") == _axis_title(axis),
            "El eje de la figura modifica las unidades documentadas.",
        )

    for key, value in _provenance_metadata(chart).items():
        _check(meta.get(key) == value, "La figura perdió metadatos de procedencia.")
    _check(
        set(chart.warnings).issubset(meta.get("warnings", [])),
        "La figura perdió advertencias relevantes.",
    )


#
# Devuelve el tipo de eje X esperado según el gráfico y el tipo SQL del campo.
#
def _x_axis_type(chart, request, x) -> str:
    if request.chart_type == "histogram" and _has_only_point_bins(chart):
        return "category"

    return _X_AXIS_TYPES.get(x.sql_type, "linear")


#
# Indica si todos los intervalos del histograma son puntuales.
#
def _has_only_point_bins(chart) -> bool:
    for series in chart.series:
        for item in series.bins:
            if item.lower != item.upper:
                return False

    return True


#
# Devuelve el título del eje con su unidad o la advertencia de unidad no documentada.
#
def _axis_title(axis) -> str:
    label = escape(axis.label)
    status = presentation_unit_status(axis.field)

    if status == "documented" and axis.field.unit is not None:
        return label + f" [{escape(axis.field.unit)}]"

    if status == "unknown" and axis.field.sql_type in {"BIGINT", "NUMERIC"}:
        return label + " (unidad no documentada)"

    return label


#
# Devuelve los metadatos de procedencia que debe conservar la figura.
#
def _provenance_metadata(chart) -> dict:
    return {
        "operation_id": chart.operation_id,
        "chart_type": chart.chart_type,
        "x_axis": chart.x_axis.model_dump(mode="json"),
        "y_axis": chart.y_axis.model_dump(mode="json"),
        "datasets": [item.model_dump(mode="json") for item in chart.datasets],
        "input_operation_ids": chart.input_operation_ids,
        "input_rows": chart.input_rows,
        "excluded_rows": chart.excluded_rows,
        "numeric_encoding": chart.numeric_encoding,
    }
