import json
from html import escape
from math import isfinite

import plotly
import plotly.graph_objects as go
from plotly.io import to_json
from plotly.offline import get_plotlyjs_version

from aquai.application.answering.presentation import (
    chart_axis_formats,
    chart_hover_templates,
)
from aquai.application.contracts.execution import (
    ChartAxis,
    PlotlyFigure,
    VisualizationResult,
)
from aquai.application.errors import AnalysisBlocked
from aquai.application.units import presentation_unit_status

_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442")
_MARKER_SYMBOLS = ("circle", "square", "diamond", "cross")
_LINE_DASHES = ("solid", "dash", "dot", "dashdot")
_X_AXIS_TYPES = {"DATE": "date", "TIMESTAMP": "date", "TEXT": "category", "BOOLEAN": "category"}


#
# Devuelve el título del eje con su unidad o la advertencia de unidad no documentada.
#
def _axis_title(axis: ChartAxis) -> str:
    label = escape(axis.label)
    field = axis.field
    status = presentation_unit_status(field)

    if status == "documented" and field.unit is not None:
        return f"{label} [{escape(field.unit)}]"

    if status == "unknown" and field.sql_type in {"BIGINT", "NUMERIC"}:
        return f"{label} (unidad no documentada)"

    return label


#
# Devuelve los valores de X de una serie, con los booleanos como texto.
#
def _x_values(series):
    return [str(point.x).lower() if type(point.x) is bool else point.x for point in series.points]


#
# Construye la figura Plotly de un gráfico validado, lista para serializar.
#
def build_plotly_figure(chart: VisualizationResult) -> PlotlyFigure:
    figure = go.Figure()
    figure.update_layout(template="none")

    for index, series in enumerate(chart.series):
        figure.add_trace(_trace(chart, series, index))

    constant_histogram = _is_constant_histogram(chart)
    x_type = _x_axis_type(chart, constant_histogram)
    x_axis = _x_axis(chart, figure, x_type)
    y_axis = _y_axis(chart)
    _apply_tick_formats(chart, x_type, x_axis, y_axis)

    figure.update_layout(
        title={"text": escape(chart.title), "x": 0.5},
        xaxis=x_axis,
        yaxis=y_axis,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
        margin={"l": 85, "r": 35, "t": 75, "b": 100},
        showlegend=len(chart.series) > 1 or chart.series_field is not None,
        legend={"title": {"text": escape(chart.series_field.name) if chart.series_field else ""}},
        barmode="group",
        bargap=0 if chart.chart_type == "histogram" and not constant_histogram else 0.2,
        hovermode="closest",
        autosize=True,
        separators=",.",
        annotations=[_note_annotation(chart)],
        meta=_figure_meta(chart),
    )

    payload = json.loads(
        to_json(figure, validate=True, pretty=False, remove_uids=True, engine="json")
    )

    return PlotlyFigure(
        renderer_version=plotly.__version__,
        plotly_js_version=get_plotlyjs_version(),
        data=payload["data"],
        layout=payload["layout"],
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": False,
            "toImageButtonOptions": {"format": "svg", "filename": chart.operation_id},
        },
    )


#
# Crea la traza Plotly de una serie según el tipo de gráfico.
#
def _trace(chart, series, index):
    color = _COLORS[index % len(_COLORS)]
    name = escape(series.name) if series.name is not None else escape(chart.y_axis.label)

    if chart.chart_type == "histogram":
        trace = _histogram_trace(series)
    elif chart.chart_type == "bar":
        trace = go.Bar(x=_x_values(series), y=[point.y for point in series.points])
    else:
        trace = _scatter_trace(chart, series, index)

    trace.update(
        name=name,
        marker_color=color,
        hovertemplate=chart_hover_templates(series, histogram=chart.chart_type == "histogram"),
    )

    return trace


#
# Crea la traza de un histograma con los intervalos ya calculados.
#
def _histogram_trace(series):
    nonconstant = [item for item in series.bins if item.lower != item.upper]

    if nonconstant:
        trace = _interval_bars(series)
    else:
        trace = go.Bar(
            x=[format(item.lower, ".17g") for item in series.bins],
            y=[item.count for item in series.bins],
        )

    trace.update(
        customdata=[
            [item.lower, item.upper, "]" if item.upper_inclusive else ")"] for item in series.bins
        ]
    )

    return trace


#
# Crea barras con centro y anchura de cada intervalo ya calculado del histograma.
#
def _interval_bars(series):
    centers, widths = [], []

    for item in series.bins:
        width = item.upper - item.lower
        center = item.lower / 2 + item.upper / 2
        representable = (
            isfinite(width) and isfinite(center) and width > 0 and item.lower < center < item.upper
        )

        if not representable:
            raise AnalysisBlocked(
                "chart_numeric_range",
                "Los intervalos no pueden representarse fielmente en Plotly.",
            )

        centers.append(center)
        widths.append(width)

    return go.Bar(x=centers, y=[item.count for item in series.bins], width=widths)


#
# Crea la traza de líneas o puntos de una serie.
#
def _scatter_trace(chart, series, index):
    trace = go.Scatter(
        x=_x_values(series),
        y=[point.y for point in series.points],
        mode="lines+markers" if chart.chart_type == "line" else "markers",
        connectgaps=False,
    )
    trace.update(marker={"size": 7, "symbol": _MARKER_SYMBOLS[index % 4]})

    if chart.chart_type == "line":
        trace.update(line={"shape": "linear", "width": 2, "dash": _LINE_DASHES[index % 4]})

    return trace


#
# Indica si el histograma es de una variable constante.
#
def _is_constant_histogram(chart):
    return chart.chart_type == "histogram" and all(
        item.lower == item.upper for series in chart.series for item in series.bins
    )


#
# Devuelve el tipo de eje X según el tipo SQL del campo.
#
def _x_axis_type(chart, constant_histogram):
    if constant_histogram:
        return "category"

    return _X_AXIS_TYPES.get(chart.x_axis.field.sql_type, "linear")


#
# Construye la configuración del eje X, con categorías explícitas si corresponde.
#
def _x_axis(chart, figure, x_type):
    axis = {
        "title": {"text": _axis_title(chart.x_axis)},
        "type": x_type,
        "automargin": True,
        "showline": True,
        "linecolor": "#333333",
        "showgrid": x_type != "category",
        "gridcolor": "#e5e7eb",
        "zeroline": False,
    }

    if x_type != "category":
        return axis

    categories = list(dict.fromkeys(value for trace in figure.data for value in trace.x))
    axis.update(
        categoryorder="array",
        categoryarray=categories,
        tickmode="array",
        tickvals=categories,
        ticktext=[escape(str(value)) for value in categories],
    )

    return axis


#
# Construye la configuración del eje Y según el tipo de gráfico.
#
def _y_axis(chart):
    axis = {
        "title": {"text": _axis_title(chart.y_axis)},
        "type": "linear",
        "automargin": True,
        "showline": True,
        "linecolor": "#333333",
        "showgrid": True,
        "gridcolor": "#e5e7eb",
        "zeroline": False,
    }

    if chart.chart_type in {"bar", "histogram"}:
        axis["rangemode"] = "tozero"

    if chart.chart_type != "histogram":
        return axis

    maximum_count = max((item.count for series in chart.series for item in series.bins), default=0)

    if maximum_count < 10:
        axis.update(tickmode="linear", dtick=1)
    else:
        axis.update(tickformat="d")

    return axis


#
# Aplica a los ejes los formatos de marcas de presentación del gráfico.
#
def _apply_tick_formats(chart, x_type, x_axis, y_axis):
    formats = chart_axis_formats(chart)

    if x_type == "linear" and "xaxis" in formats:
        x_axis["tickformat"] = formats["xaxis"]

    if "yaxis" in formats:
        y_axis["tickformat"] = formats["yaxis"]


#
# Crea la nota al pie con las filas de entrada y excluidas y las unidades no documentadas.
#
def _note_annotation(chart):
    note = f"{chart.input_rows} filas de entrada; {chart.excluded_rows} excluidas."
    undocumented = any(
        presentation_unit_status(axis.field) == "unknown"
        and axis.field.sql_type in {"BIGINT", "NUMERIC"}
        for axis in (chart.x_axis, chart.y_axis)
    )

    if undocumented:
        note += " Hay unidades no documentadas."

    return {
        "text": note,
        "xref": "paper",
        "yref": "paper",
        "x": 0,
        "y": -0.24,
        "showarrow": False,
        "xanchor": "left",
        "font": {"size": 11, "color": "#555555"},
    }


#
# Devuelve los metadatos de procedencia que se incluyen en la figura.
#
def _figure_meta(chart):
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
        "warnings": chart.warnings,
    }
