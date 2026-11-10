from collections import Counter
from datetime import date
from decimal import Decimal

from aquai.application.answering.presentation import format_number
from aquai.application.answering.wording import date_text, file_stem, join_words
from aquai.application.execution import execution_records

_CHARTS = {
    "es": {
        "bar": "el gráfico de barras",
        "line": "el gráfico de líneas",
        "scatter": "el gráfico de dispersión",
        "histogram": "el histograma",
    },
    "en": {
        "bar": "the bar chart",
        "line": "the line chart",
        "scatter": "the scatter plot",
        "histogram": "the histogram",
    },
}
_GROUPS = {
    "input_limit": "limit",
    "result_limit": "limit",
    "memory_limit": "limit",
    "incomplete_input": "limit",
    "execution_limit": "limit",
    "python_timeout": "timeout",
    "query_timeout": "timeout",
    "incompatible_units": "incompatible_units",
}
_CHART_REASONS = {
    "es": {
        "ambiguous_chart_granularity": "hay más de un valor para algunas categorías{examples} y el gráfico necesita uno solo "
        "por categoría para no mezclar mediciones. Si querés, pedime el promedio, el máximo "
        "o el total de cada categoría y lo grafico.",
        "chart_points_limit": "tendría demasiados puntos para mostrarlos todos, y no lo recorto para no ocultar datos. "
        "Podés acotar el período o agrupar los valores, por ejemplo por mes.",
        "chart_series_limit": "tendría demasiadas series para distinguirlas. Podés elegir menos categorías.",
        "no_chart_data": "no hay valores válidos para representar.",
        "chart_numeric_range": "no se puede dibujar sin deformar los valores.",
    },
    "en": {
        "ambiguous_chart_granularity": "some categories have more than one value{examples}, and the chart needs a single "
        "value per category so measurements are not mixed. You can ask me for the mean, "
        "maximum or total of each category and I will chart it.",
        "chart_points_limit": "it would have too many points to show them all, and I do not trim it to avoid hiding data. "
        "You can narrow the period or group the values, for example by month.",
        "chart_series_limit": "it would have too many series to tell apart. You can choose fewer categories.",
        "no_chart_data": "there are no valid values to plot.",
        "chart_numeric_range": "it cannot be drawn without distorting the values.",
    },
}
_REASONS = {
    "es": {
        "limit": "Una parte del análisis necesitaba procesar más datos de los permitidos, así que no la calculé para no "
        "dar un resultado incompleto. Podés acotar la consulta a un sitio, un período o una variable.",
        "incompatible_units": "No combiné variables con unidades distintas o no documentadas, porque el resultado no "
        "sería válido.",
        "timeout": "Una parte del análisis tardó más de lo permitido y no se completó. Podés probar con una "
        "consulta más acotada.",
        "other": "Otra parte de la consulta no se pudo calcular de forma verificable, así que no la incluyo.",
    },
    "en": {
        "limit": "Part of the analysis needed more data than allowed, so I did not calculate it rather than give an "
        "incomplete result. You can narrow the question to a site, a period or a variable.",
        "incompatible_units": "I did not combine variables with different or undocumented units, because the result "
        "would not be valid.",
        "timeout": "Part of the analysis took longer than allowed and did not finish. You can try a narrower question.",
        "other": "Another part of the question could not be calculated in a verifiable way, so I left it out.",
    },
}


_FAILED_STATUSES = {"blocked", "error", "unsupported"}
_CATEGORICAL_CHARTS = {"bar", "line"}


#
# Devuelve los avisos que explican por qué no se pudo completar cada operación fallida.
#
def blocked_explanations(state, language) -> list[str]:
    steps = {step.id: step for step in state.planning.operations} if state.planning else {}
    records = execution_records(state)
    texts = []
    unexplained = False
    for record in records.values():
        step = steps.get(record.operation_id)
        if record.status not in _FAILED_STATUSES or step is None:
            continue

        text = _explanation(step, record, records, language, _chart_file(step, steps, state))
        if text is None:
            unexplained = True
        else:
            texts.append(text)

    if unexplained and texts:
        texts.append(_REASONS[language]["other"])

    return list(dict.fromkeys(texts))


#
# Elige la explicación de una operación fallida según su código de error.
#
def _explanation(step, record, records, language, file=None):
    code = record.error.code if record.error else None
    if step.operation == "visualize" and code in _CHART_REASONS[language]:
        return _chart_explanation(step, code, records, language, file)
    if code in _GROUPS:
        return _REASONS[language][_GROUPS[code]]

    return None


#
# Devuelve el archivo del que lee un gráfico, sólo si el plan lee varios archivos y hace falta distinguirlo.
#
def _chart_file(step, steps, state):
    files = {
        item.dataset_id: file_stem(item.provenance.original_filename)
        for item in (state.retrieval.candidates if state.retrieval else [])
    }
    read_files = {
        files.get(ref.dataset_id)
        for item in steps.values()
        if item.operation == "read"
        for ref in item.columns[:1]
    }
    if step.operation != "visualize" or len(read_files) < 2:
        return None

    current = step
    while current is not None and current.operation != "read" and current.depends_on:
        current = steps.get(current.depends_on[0])

    return files.get(current.columns[0].dataset_id) if current is not None and current.columns else None


#
# Explica por qué no se pudo armar un gráfico, con ejemplos de categorías repetidas si aplica.
#
def _chart_explanation(step, code, records, language, file=None):
    examples = _repeated(step, records, language) if code == "ambiguous_chart_granularity" else ""
    lead = (
        "No pude armar {chart} porque "
        if language == "es"
        else "I could not build {chart} because "
    )
    chart = _CHARTS[language][step.visualization.chart_type]
    if file:
        chart += f" del archivo {file}" if language == "es" else f" for the file {file}"

    return lead.format(chart=chart) + _CHART_REASONS[language][code].format(examples=examples)


#
# Indica ejemplos de categorías repetidas que impiden graficar un valor por categoría.
#
def _repeated(step, records, language):
    request = step.visualization
    parent = records.get(step.depends_on[0]) if step.depends_on else None
    if parent is None or not parent.rows or request.chart_type not in _CATEGORICAL_CHARTS:
        return ""

    counts = _category_counts(parent.rows, request)
    categories = [category for (_, category), times in counts.items() if times > 1]
    repeated = list(dict.fromkeys(_shown(category, language) for category in categories))[:2]
    if not repeated:
        return ""

    names = join_words(repeated, language)
    plural = len(repeated) > 1
    if language == "es":
        return f" (por ejemplo, {names} aparece{'n' if plural else ''} más de una vez)"

    return f" (for example, {names} appear{'' if plural else 's'} more than once)"


#
# Cuenta cuántas veces aparece cada categoría graficable, por serie.
#
def _category_counts(rows, request):
    series = request.series.field if request.series is not None else None
    counts = Counter()
    for row in rows:
        if _plottable(row, request):
            counts[(row.get(series) if series else None, row[request.x.field])] += 1

    return counts


#
# Indica si una fila tiene los valores necesarios para dibujarse en el gráfico.
#
def _plottable(row, request):
    if row.get(request.x.field) is None:
        return False

    return row.get(request.y.field) is not None or request.null_policy == "preserve"


#
# Devuelve un valor como texto legible, con fecha o número formateados según el idioma.
#
def _shown(value, language):
    if isinstance(value, date):
        return date_text(value, language)
    if isinstance(value, (int, float, Decimal)) and type(value) is not bool:
        return format_number(value, language)

    return str(value)
