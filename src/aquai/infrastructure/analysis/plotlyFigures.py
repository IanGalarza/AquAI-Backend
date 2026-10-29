"""Construcción de figuras Plotly desde datos ya verificados, sin decisiones LLM."""

import json
from html import escape
from math import isfinite

import plotly
import plotly.graph_objects as go
from plotly.io import to_json
from plotly.offline import get_plotlyjs_version

from aquai.application.contracts.execution import ChartAxis, PlotlyFigure, VisualizationResult
from aquai.application.answering.presentation import chart_axis_formats, chart_hover_templates
from aquai.application.units import presentation_unit_status
from aquai.infrastructure.analysis.operations import AnalysisBlocked


_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442")


def _axis_title(axis: ChartAxis) -> str:
    label = escape(axis.label)
    field = axis.field
    status = presentation_unit_status(field)
    if status == "documented" and field.unit is not None:
        return f"{label} [{escape(field.unit)}]"
    if status == "unknown" and field.sql_type in {"BIGINT", "NUMERIC"}:
        return f"{label} (unidad no documentada)"
    return label


def _x_values(chart: VisualizationResult, series):
    # Plotly representa categorías booleanas con etiquetas explícitas.
    return [str(point.x).lower() if type(point.x) is bool else point.x for point in series.points]


def build_plotly_figure(chart: VisualizationResult) -> PlotlyFigure:
    """Construye trazas, ejes, leyendas y contexto; el navegador sólo renderiza."""
    figure = go.Figure()
    figure.update_layout(template="none")
    for index, series in enumerate(chart.series):
        color = _COLORS[index % len(_COLORS)]
        name = escape(series.name) if series.name is not None else escape(chart.y_axis.label)
        if chart.chart_type == "histogram":
            # go.Histogram calcula frecuencias en JavaScript. Usamos los bins
            # ya calculados con Decimal, representados mediante barras explícitas.
            nonconstant = [item for item in series.bins if item.lower != item.upper]
            if nonconstant:
                centers, widths = [], []
                for item in series.bins:
                    width = item.upper - item.lower
                    center = item.lower / 2 + item.upper / 2
                    if not isfinite(width) or not isfinite(center) or width <= 0 or not item.lower < center < item.upper:
                        raise AnalysisBlocked("chart_numeric_range", "Los intervalos no pueden representarse fielmente en Plotly.")
                    centers.append(center)
                    widths.append(width)
                trace = go.Bar(x=centers, y=[item.count for item in series.bins], width=widths)
            else:
                # No se inventa anchura numérica para una distribución constante.
                trace = go.Bar(x=[format(item.lower, ".17g") for item in series.bins],
                               y=[item.count for item in series.bins])
            trace.update(customdata=[[
                item.lower, item.upper, "]" if item.upper_inclusive else ")",
            ] for item in series.bins])
        elif chart.chart_type == "bar":
            trace = go.Bar(x=_x_values(chart, series), y=[point.y for point in series.points])
        else:
            trace = go.Scatter(x=_x_values(chart, series), y=[point.y for point in series.points],
                mode="lines+markers" if chart.chart_type == "line" else "markers",
                connectgaps=False)
            trace.update(marker={"size": 7, "symbol": ("circle", "square", "diamond", "cross")[index % 4]})
            if chart.chart_type == "line":
                trace.update(line={"shape": "linear", "width": 2,
                    "dash": ("solid", "dash", "dot", "dashdot")[index % 4]})
        trace.update(name=name, marker_color=color,
                     hovertemplate=chart_hover_templates(series, histogram=chart.chart_type == "histogram"))
        figure.add_trace(trace)

    x_type = ("date" if chart.x_axis.field.sql_type in {"DATE", "TIMESTAMP"} else
              "category" if chart.x_axis.field.sql_type in {"TEXT", "BOOLEAN"} else "linear")
    constant_histogram = chart.chart_type == "histogram" and all(
        item.lower == item.upper for series in chart.series for item in series.bins)
    if constant_histogram:
        x_type = "category"
    x_axis = {"title": {"text": _axis_title(chart.x_axis)}, "type": x_type,
              "automargin": True, "showline": True, "linecolor": "#333333",
              "showgrid": x_type != "category", "gridcolor": "#e5e7eb", "zeroline": False}
    if x_type == "category":
        categories = list(dict.fromkeys(value for trace in figure.data for value in trace.x))
        x_axis.update(categoryorder="array", categoryarray=categories, tickmode="array",
                      tickvals=categories, ticktext=[escape(str(value)) for value in categories])
    y_axis = {"title": {"text": _axis_title(chart.y_axis)}, "type": "linear", "automargin": True,
              "showline": True, "linecolor": "#333333", "showgrid": True, "gridcolor": "#e5e7eb", "zeroline": False}
    if chart.chart_type in {"bar", "histogram"}:
        y_axis["rangemode"] = "tozero"
    if chart.chart_type == "histogram":
        maximum_count = max((item.count for series in chart.series for item in series.bins), default=0)
        if maximum_count < 10:
            y_axis.update(tickmode="linear", dtick=1)
        else:
            y_axis.update(tickformat="d")

    formats = chart_axis_formats(chart)
    if x_type == "linear" and "xaxis" in formats:
        x_axis["tickformat"] = formats["xaxis"]
    if "yaxis" in formats:
        y_axis["tickformat"] = formats["yaxis"]
    note = f"{chart.input_rows} filas de entrada; {chart.excluded_rows} excluidas."
    if any(presentation_unit_status(axis.field) == "unknown" and axis.field.sql_type in {"BIGINT", "NUMERIC"}
           for axis in (chart.x_axis, chart.y_axis)):
        note += " Hay unidades no documentadas."
    figure.update_layout(title={"text": escape(chart.title), "x": 0.5},
        xaxis=x_axis, yaxis=y_axis, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "size": 13, "color": "#222222"},
        margin={"l": 85, "r": 35, "t": 75, "b": 100},
        showlegend=len(chart.series) > 1 or chart.series_field is not None,
        legend={"title": {"text": escape(chart.series_field.name) if chart.series_field else ""}},
        barmode="group", bargap=0 if chart.chart_type == "histogram" and not constant_histogram else 0.2,
        hovermode="closest", autosize=True, separators=",.",
        annotations=[{"text": note, "xref": "paper", "yref": "paper", "x": 0, "y": -0.24,
                      "showarrow": False, "xanchor": "left", "font": {"size": 11, "color": "#555555"}}],
        meta={"operation_id": chart.operation_id, "chart_type": chart.chart_type,
              "x_axis": chart.x_axis.model_dump(mode="json"), "y_axis": chart.y_axis.model_dump(mode="json"),
              "datasets": [item.model_dump(mode="json") for item in chart.datasets],
              "input_operation_ids": chart.input_operation_ids, "input_rows": chart.input_rows,
              "excluded_rows": chart.excluded_rows, "numeric_encoding": chart.numeric_encoding,
              "warnings": chart.warnings})
    # La librería valida la figura. Sólo JSON estándar cruza a Application/LangGraph.
    payload = json.loads(to_json(figure, validate=True, pretty=False, remove_uids=True, engine="json"))
    return PlotlyFigure(renderer_version=plotly.__version__, plotly_js_version=get_plotlyjs_version(),
        data=payload["data"], layout=payload["layout"],
        config={"responsive": True, "displaylogo": False, "scrollZoom": False,
                "toImageButtonOptions": {"format": "svg", "filename": chart.operation_id}})
