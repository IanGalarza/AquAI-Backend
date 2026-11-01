import logging
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from aquai.application.answering.naming import FieldNames
from aquai.application.answering.presentation import format_unit
from aquai.application.answering.segments import segments_from_text
from aquai.application.answering.wording import (
    CELL_RANGE,
    NUMERIC_TYPES,
    SPREADSHEET_TYPES,
    column_label,
    dataset_label,
    display_unit,
    sentence,
)
from aquai.application.contracts.answering import AnswerSegment
from aquai.application.contracts.conversations import (
    ChartView,
    CitationVariable,
    CitationView,
    ColumnView,
    MessagePresentation,
    SegmentView,
    TableView,
)

_logger = logging.getLogger(__name__)
_FINISHED = {"completed", "failed", "cancelled"}


class _Layout(NamedTuple):
    by_columns: bool
    first: int
    last: int


#
# Devuelve el número de columna correspondiente a una letra de planilla.
#
def _column_index(letters):
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index


#
# Devuelve la letra de planilla correspondiente a un número de columna.
#
def _column_letter(index):
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


#
# Devuelve las posiciones enteras de una lista, o una lista vacía.
#
def _positions(value):
    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, int)]


#
# Devuelve la disposición de los datos en la hoja, o None si no es una planilla.
#
def _layout(provenance):
    locator = provenance.source_locator
    match = CELL_RANGE.match(str(locator.get("range", "")))
    if provenance.file_type not in SPREADSHEET_TYPES or match is None:
        return None

    by_columns = locator.get("orientation") == "columns"
    if by_columns:
        headers = _positions(locator.get("header_columns"))
        start = _column_index(match[1])
        end = _column_index(match[3])
    else:
        headers = _positions(locator.get("header_rows"))
        start = int(match[2])
        end = int(match[4])
    first = max(headers) + 1 if headers else start

    return _Layout(by_columns, first, end)


class _Stored(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Ref(_Stored):
    kind: str = "column"
    dataset_id: UUID | None = None
    column: str = ""
    step_id: str = ""
    field: str = ""


class _Field(_Stored):
    name: str
    sql_type: str
    unit: str | None = None
    unit_status: str = "unknown"
    sources: list[_Ref] = Field(default_factory=list)


class _Table(_Stored):
    evidence_id: str
    operation_id: str
    columns: list[_Field]
    total_rows: int = 0


class _Axis(_Stored):
    field: _Field
    label: str


class _DatasetVersion(_Stored):
    dataset_id: UUID


class _Chart(_Stored):
    operation_id: str
    x_axis: _Axis | None = None
    y_axis: _Axis | None = None
    series_field: _Field | None = None
    input_operation_ids: list[str] = Field(default_factory=list)
    datasets: list[_DatasetVersion] = Field(default_factory=list)


class _Calculation(_Stored):
    function: str
    inputs: list[_Ref] = Field(default_factory=list)
    output_name: str
    quantile: float | None = None
    ddof: int = 1


class _Grouping(_Stored):
    field: _Ref | None = None
    output_name: str
    time_grain: str = "raw"


class _Step(_Stored):
    id: str
    operation: str = ""
    depends_on: list[str] = Field(default_factory=list)
    columns: list[_Ref] = Field(default_factory=list)
    calculations: list[_Calculation] = Field(default_factory=list)
    group_by: list[_Grouping] = Field(default_factory=list)
    output_fields: list[_Field] = Field(default_factory=list)


class _Provenance(_Stored):
    original_filename: str
    file_type: str = ""
    source_locator: dict[str, JsonValue] = Field(default_factory=dict)


class _Source(_Stored):
    source_id: UUID
    dataset_id: UUID
    dataset_name: str
    version: int
    provenance: _Provenance


class _Evidence(_Stored):
    evidence_id: str
    operation_id: str
    dataset_ids: list[UUID] = Field(default_factory=list)


class _Response(_Stored):
    answer: str
    language: str = "es"
    segments: list[AnswerSegment] = Field(default_factory=list)
    sources: list[_Source] = Field(default_factory=list)
    operation_references: list[_Evidence] = Field(default_factory=list)
    results: list[_Table] = Field(default_factory=list)
    visualizations: list[_Chart] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class _Snapshot(_Stored):
    response: _Response
    operations: list[_Step] = Field(default_factory=list)
    storage_warnings: list[str] = Field(default_factory=list)


#
# Devuelve la presentación estructurada de un mensaje del asistente, o None.
#
def present_message(view) -> MessagePresentation | None:
    if view["role"] != "assistant" or view["status"]["name"] not in _FINISHED:
        return None

    data = view["content_data"]
    duration = _duration_ms(view, data)
    plain = MessagePresentation(segments=[SegmentView(text=view["content"])], duration_ms=duration)
    if data.get("kind") != "assistant_response":
        return plain

    try:
        return _Presenter(_Snapshot.model_validate(data), view["sources"]).present(duration)
    except (ValidationError, LookupError, TypeError, ValueError):
        _logger.warning(
            "Mensaje %s sin presentación estructurada; se muestra su texto.", view.get("id")
        )
        return plain


#
# Devuelve la duración del turno en milisegundos, o None si no es confiable.
#
def _duration_ms(view, data):
    error = data.get("error")
    interrupted = isinstance(error, dict) and error.get("code") == "interrupted_generation"
    if view["updated_at"] is None or interrupted:
        return None

    elapsed = view["updated_at"] - view["created_at"]

    return max(0, round(elapsed.total_seconds() * 1000))


class _Presenter:
    #
    # Guarda la fotografía de la respuesta, el idioma y las citas del mensaje.
    #
    def __init__(self, snapshot, sources):
        self._snapshot = snapshot
        self._response = snapshot.response
        self._language = "en" if self._response.language == "en" else "es"
        self._steps = {step.id: step for step in snapshot.operations}
        self._columns = _stored_columns(sources)
        self._names = FieldNames(self._steps, self._column, self._language)
        self._citations = self._build_citations()
        self._datasets = {item.dataset_id: item.id for item in self._citations}
        self._evidence = {
            ref.evidence_id: self._cite(ref.dataset_ids)
            for ref in self._response.operation_references
        }
        self._operations = {
            ref.evidence_id: ref.operation_id for ref in self._response.operation_references
        }

    #
    # Construye la presentación del mensaje con segmentos, citas, tablas y gráficos.
    #
    def present(self, duration):
        response = self._response
        tables = {table.evidence_id: table for table in response.results}
        charts = {chart.operation_id: chart for chart in response.visualizations}
        segments = response.segments or self._legacy_segments()
        views = self._segments(segments, tables, charts)
        trailing_tables = self._trailing_tables(views)
        trailing_charts = self._trailing_charts(views)

        return MessagePresentation(
            segments=views,
            citations=self._citations,
            tables=[self._table(table) for table in response.results],
            charts=[self._chart(chart) for chart in response.visualizations],
            trailing_tables=trailing_tables,
            trailing_charts=trailing_charts,
            duration_ms=duration,
        )

    #
    # Devuelve las tablas que ningún segmento presenta, en respuestas sin segmentos.
    #
    def _trailing_tables(self, views):
        if self._response.segments:
            return []

        placed = {view.table for view in views if view.table}

        return [
            table.evidence_id
            for table in self._response.results
            if table.total_rows > 1 and table.evidence_id not in placed
        ]

    #
    # Devuelve los gráficos que ningún segmento presenta.
    #
    def _trailing_charts(self, views):
        placed = {view.chart for view in views if view.chart}

        return [
            chart.operation_id
            for chart in self._response.visualizations
            if chart.operation_id not in placed
        ]

    #
    # Devuelve la etiqueta de una columna de un dataset, o None.
    #
    def _column(self, dataset_id, sql_name):
        column = self._columns.get((dataset_id, sql_name))

        return column_label(column["original_name"], column.get("unit")) if column else None

    #
    # Construye los segmentos de una respuesta anterior al formato estructurado.
    #
    def _legacy_segments(self):
        hidden = {*self._response.limitations, " ".join(self._snapshot.storage_warnings)}

        return segments_from_text(self._response.answer, self._evidence, hidden)

    #
    # Devuelve las citas de los datasets indicados, sin repetidos.
    #
    def _cite(self, dataset_ids):
        citations = [self._datasets[key] for key in dataset_ids if key in self._datasets]

        return list(dict.fromkeys(citations))

    #
    # Convierte los segmentos de la respuesta en vistas de segmento.
    #
    def _segments(self, segments, tables, charts):
        views = []
        for index, segment in enumerate(segments):
            following = segments[index + 1] if index + 1 < len(segments) else None
            views.append(self._segment(segment, following, tables, charts))

        return views

    #
    # Convierte un segmento en su vista, con citas y tabla o gráfico adjuntos.
    #
    def _segment(self, segment, following, tables, charts):
        cited = (
            following is None
            or following.evidence_ids != segment.evidence_ids
            or segment.attachment
        )
        operation = self._operations.get(segment.attachment)

        return SegmentView(
            kind=segment.kind,
            text=segment.text,
            items=segment.items,
            citations=self._segment_citations(segment) if cited else [],
            table=segment.attachment if segment.attachment in tables else None,
            chart=operation if operation in charts else None,
        )

    #
    # Devuelve las citas de la evidencia de un segmento.
    #
    def _segment_citations(self, segment):
        citations = []
        for key in segment.evidence_ids:
            citations.extend(self._evidence.get(key, []))

        return list(dict.fromkeys(citations))

    #
    # Construye las citas de las fuentes de la respuesta.
    #
    def _build_citations(self):
        variables = self._variables()
        citations = []
        seen = set()
        for source in self._response.sources:
            if source.dataset_id in seen:
                continue
            seen.add(source.dataset_id)
            refs = variables.get(source.dataset_id, [])
            citations.append(self._citation(source, f"c{len(citations) + 1}", refs))

        return citations

    #
    # Construye la cita de una fuente con su archivo, hoja, filas y variables.
    #
    def _citation(self, source, citation_id, refs):
        provenance = source.provenance
        spreadsheet = provenance.file_type in SPREADSHEET_TYPES
        sheet = provenance.source_locator.get("sheet") if spreadsheet else None
        layout = _layout(provenance)
        rows = layout if layout is not None and not layout.by_columns else None

        return CitationView(
            id=citation_id,
            source_id=source.source_id,
            dataset_id=source.dataset_id,
            version=source.version,
            file_name=provenance.original_filename,
            file_type=provenance.file_type,
            sheet=sheet if isinstance(sheet, str) else None,
            dataset=dataset_label(source.dataset_name) if spreadsheet else "",
            first_row=rows.first if rows else None,
            last_row=rows.last if rows else None,
            variables=[self._located(ref, layout, spreadsheet) for ref in refs],
        )

    #
    # Devuelve las columnas leídas agrupadas por dataset.
    #
    def _variables(self):
        refs = self._read_columns() or self._result_columns()
        variables = {}
        for ref in refs:
            if ref.dataset_id is not None:
                variables.setdefault(ref.dataset_id, {}).setdefault(ref.column, ref)

        return {key: list(values.values()) for key, values in variables.items()}

    #
    # Devuelve las columnas que leen los pasos del plan.
    #
    def _read_columns(self):
        refs = []
        for step in self._steps.values():
            refs.extend(step.columns)

        return refs

    #
    # Devuelve las columnas de origen de los resultados.
    #
    def _result_columns(self):
        refs = []
        for table in self._response.results:
            for field in table.columns:
                refs.extend(field.sources)

        return refs

    #
    # Devuelve la variable con su ubicación en la planilla original.
    #
    def _located(self, ref, layout, spreadsheet):
        name = self._names.column(ref.dataset_id, ref.column)
        column = self._columns.get((ref.dataset_id, ref.column)) or {}
        source_row = column.get("source_row")
        source_column = column.get("source_column")

        if layout is None:
            return _unranged_variable(name, source_column, spreadsheet)
        if layout.by_columns and isinstance(source_row, int):
            return _row_variable(name, source_row, layout)
        if not layout.by_columns and isinstance(source_column, int):
            return _column_variable(name, source_column, layout)

        return CitationVariable(name=name)

    #
    # Construye la vista de una tabla de resultados con sus columnas y citas.
    #
    def _table(self, table):
        origin = self._names.origin(self._steps.get(table.operation_id))
        grains = _time_grains(origin) if origin else {}
        columns = [self._column_view(table, field, grains) for field in table.columns]

        return TableView(
            evidence_id=table.evidence_id,
            columns=columns,
            citations=self._evidence.get(table.evidence_id, []),
        )

    #
    # Construye la vista de una columna con su etiqueta, unidad y agrupación temporal.
    #
    def _column_view(self, table, field, grains):
        unit, documented = display_unit(field, self._language)

        return ColumnView(
            name=field.name,
            label=sentence(self._names.result(table.operation_id, field.name, field)),
            unit=unit,
            unit_documented=documented,
            numeric=field.sql_type in NUMERIC_TYPES,
            time_grain=grains.get(field.name),
        )

    #
    # Construye la vista de un gráfico con sus títulos y citas.
    #
    def _chart(self, chart):
        source = chart.input_operation_ids[0] if chart.input_operation_ids else ""
        legend = None
        if chart.series_field:
            series = chart.series_field
            legend = sentence(self._names.result(source, series.name, series))

        return ChartView(
            operation_id=chart.operation_id,
            citations=self._cite([item.dataset_id for item in chart.datasets]),
            x_title=self._axis_title(chart.x_axis),
            y_title=self._axis_title(chart.y_axis),
            legend_title=legend,
        )

    #
    # Devuelve el título de un eje con su unidad formateada.
    #
    def _axis_title(self, axis):
        if axis is None:
            return None

        unit, documented = display_unit(axis.field, self._language)
        label = axis.label
        raw = axis.field.unit
        if documented and raw and raw in label:
            return label.replace(raw, format_unit(raw))

        return f"{label} ({unit})" if unit else label


#
# Devuelve las columnas guardadas de las fuentes por dataset y nombre.
#
def _stored_columns(sources):
    columns = {}
    for source in sources:
        for column in source["details"].get("columns") or []:
            if isinstance(column, dict):
                columns[(source["dataset_id"], column.get("sql_name"))] = column

    return columns


#
# Devuelve la agrupación temporal de cada campo de un paso.
#
def _time_grains(step):
    return {item.output_name: item.time_grain for item in step.group_by if item.time_grain != "raw"}


#
# Devuelve la variable cuando no se puede verificar un rango de celdas.
#
def _unranged_variable(name, source_column, spreadsheet):
    if not isinstance(source_column, int):
        return CitationVariable(name=name)

    column = _column_letter(source_column) if spreadsheet else str(source_column)

    return CitationVariable(name=name, column=column)


#
# Devuelve la variable de una fila con el rango de celdas que ocupa.
#
def _row_variable(name, row, layout):
    cells = f"{_column_letter(layout.first)}{row}:{_column_letter(layout.last)}{row}"

    return CitationVariable(name=name, row=row, cells=cells)


#
# Devuelve la variable de una columna con el rango de celdas que ocupa.
#
def _column_variable(name, position, layout):
    letter = _column_letter(position)

    return CitationVariable(
        name=name, column=letter, cells=f"{letter}{layout.first}:{letter}{layout.last}"
    )
