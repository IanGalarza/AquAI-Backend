"""Lectura de un mensaje del asistente: párrafos con citas, tablas rotuladas y duración real del turno."""

import logging
from typing import NamedTuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from aquai.application.answering.naming import FieldNames
from aquai.application.answering.presentation import format_unit
from aquai.application.answering.segments import segments_from_text
from aquai.application.answering.wording import (
    CELL_RANGE, NUMERIC_TYPES, SPREADSHEET_TYPES, column_label, dataset_label, display_unit, sentence,
)
from aquai.application.contracts.answering import AnswerSegment
from aquai.application.contracts.conversations import (
    ChartView, CitationVariable, CitationView, ColumnView, MessagePresentation, SegmentView, TableView,
)


_logger = logging.getLogger(__name__)
_FINISHED = {"completed", "failed", "cancelled"}


class _Layout(NamedTuple):
    """Datos de la tabla en la hoja: filas (o columnas, si los registros van por columna) sin encabezados."""

    by_columns: bool
    first: int
    last: int


def _column_index(letters):
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - 64
    return index


def _column_letter(index):
    letters = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _positions(value):
    return [item for item in value if isinstance(item, int)] if isinstance(value, list) else []


def _layout(provenance):
    """Sólo las planillas conservan celdas reales; en un CSV el rango cuenta registros, no líneas del archivo."""
    locator = provenance.source_locator
    match = CELL_RANGE.match(str(locator.get("range", "")))
    if provenance.file_type not in SPREADSHEET_TYPES or match is None:
        return None
    left, top, right, bottom = _column_index(match[1]), int(match[2]), _column_index(match[3]), int(match[4])
    if locator.get("orientation") == "columns":
        headers = _positions(locator.get("header_columns"))
        return _Layout(True, max(headers) + 1 if headers else left, right)
    headers = _positions(locator.get("header_rows"))
    return _Layout(False, max(headers) + 1 if headers else top, bottom)


class _Stored(BaseModel):
    # Lectura tolerante: las fotografías anteriores conservan sus campos, aunque falten los nuevos.
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


def present_message(view) -> MessagePresentation | None:
    """Presentación de un mensaje ya proyectado; los turnos en curso y del usuario no la tienen."""
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
        # Una versión desconocida no debe impedir leer el historial: se conserva su texto original.
        _logger.warning("Mensaje %s sin presentación estructurada; se muestra su texto.", view.get("id"))
        return plain


def _duration_ms(view, data):
    """Duración del turno en el servidor: desde su reserva hasta que se guardó el resultado."""
    error = data.get("error")
    if view["updated_at"] is None or (isinstance(error, dict) and error.get("code") == "interrupted_generation"):
        return None
    return max(0, round((view["updated_at"] - view["created_at"]).total_seconds() * 1000))


class _Presenter:
    def __init__(self, snapshot, sources):
        self._snapshot, self._response = snapshot, snapshot.response
        self._language = "en" if self._response.language == "en" else "es"
        self._steps = {step.id: step for step in snapshot.operations}
        self._columns = {(source["dataset_id"], column.get("sql_name")): column for source in sources
                         for column in source["details"].get("columns") or [] if isinstance(column, dict)}
        self._names = FieldNames(self._steps, self._column, self._language)
        self._citations = self._build_citations()
        self._datasets = {item.dataset_id: item.id for item in self._citations}
        self._evidence = {ref.evidence_id: self._cite(ref.dataset_ids) for ref in self._response.operation_references}
        self._operations = {ref.evidence_id: ref.operation_id for ref in self._response.operation_references}

    def present(self, duration):
        response = self._response
        tables = {table.evidence_id: table for table in response.results}
        charts = {chart.operation_id: chart for chart in response.visualizations}
        segments = response.segments or self._legacy_segments()
        views = self._segments(segments, tables, charts)
        placed_tables = {view.table for view in views if view.table}
        placed_charts = {view.chart for view in views if view.chart}
        # En el formato estructurado, una tabla sin párrafo que la presente ya está descrita en el texto.
        trailing_tables = [] if response.segments else [
            table.evidence_id for table in response.results if table.total_rows > 1 and table.evidence_id not in placed_tables]
        return MessagePresentation(
            segments=views, citations=self._citations,
            tables=[self._table(table) for table in response.results],
            charts=[self._chart(chart) for chart in response.visualizations],
            trailing_tables=trailing_tables,
            trailing_charts=[chart.operation_id for chart in response.visualizations if chart.operation_id not in placed_charts],
            duration_ms=duration,
        )

    def _column(self, dataset_id, sql_name):
        """Nombre original guardado en MessageSource; sin él, FieldNames usa el identificador legible."""
        column = self._columns.get((dataset_id, sql_name))
        return column_label(column["original_name"], column.get("unit")) if column else None

    def _legacy_segments(self):
        """Formato anterior: los marcadores [eN] pasan a citas y los avisos técnicos quedan fuera del texto."""
        hidden = {*self._response.limitations, " ".join(self._snapshot.storage_warnings)}
        return segments_from_text(self._response.answer, self._evidence, hidden)

    def _cite(self, dataset_ids):
        return list(dict.fromkeys(self._datasets[key] for key in dataset_ids if key in self._datasets))

    def _segments(self, segments, tables, charts):
        views = []
        for index, segment in enumerate(segments):
            following = segments[index + 1] if index + 1 < len(segments) else None
            # Párrafos seguidos de una misma evidencia comparten la cita, al final del último o antes de su tabla.
            cited = following is None or following.evidence_ids != segment.evidence_ids or segment.attachment
            operation = self._operations.get(segment.attachment)
            views.append(SegmentView(
                kind=segment.kind, text=segment.text, items=segment.items,
                citations=list(dict.fromkeys(item for key in segment.evidence_ids for item in self._evidence.get(key, []))) if cited else [],
                table=segment.attachment if segment.attachment in tables else None,
                chart=operation if operation in charts else None,
            ))
        return views

    def _build_citations(self):
        variables = self._variables()
        citations, seen = [], set()
        for source in self._response.sources:
            if source.dataset_id in seen:
                continue
            seen.add(source.dataset_id)
            provenance = source.provenance
            spreadsheet = provenance.file_type in SPREADSHEET_TYPES
            sheet = provenance.source_locator.get("sheet") if spreadsheet else None
            layout = _layout(provenance)
            rows = layout if layout is not None and not layout.by_columns else None
            citations.append(CitationView(
                id=f"c{len(citations) + 1}", source_id=source.source_id, dataset_id=source.dataset_id,
                version=source.version, file_name=provenance.original_filename, file_type=provenance.file_type,
                sheet=sheet if isinstance(sheet, str) else None, dataset=dataset_label(source.dataset_name) if spreadsheet else "",
                first_row=rows.first if rows else None, last_row=rows.last if rows else None,
                variables=[self._located(ref, layout, spreadsheet) for ref in variables.get(source.dataset_id, [])],
            ))
        return citations

    def _variables(self):
        """Columnas leídas de cada dataset; sin plan conservado, las de sus resultados."""
        refs = [ref for step in self._steps.values() for ref in step.columns] or [
            ref for table in self._response.results for field in table.columns for ref in field.sources]
        variables = {}
        for ref in refs:
            if ref.dataset_id is not None:
                variables.setdefault(ref.dataset_id, {}).setdefault(ref.column, ref)
        return {key: list(values.values()) for key, values in variables.items()}

    def _located(self, ref, layout, spreadsheet):
        """Columna (o fila) de la variable y el rango de celdas con sus datos, según la ingesta."""
        name = self._names.column(ref.dataset_id, ref.column)
        column = self._columns.get((ref.dataset_id, ref.column)) or {}
        if layout is not None and layout.by_columns and isinstance(column.get("source_row"), int):
            row = column["source_row"]
            return CitationVariable(name=name, row=row,
                                    cells=f"{_column_letter(layout.first)}{row}:{_column_letter(layout.last)}{row}")
        if layout is not None and not layout.by_columns and isinstance(column.get("source_column"), int):
            letter = _column_letter(column["source_column"])
            return CitationVariable(name=name, column=letter, cells=f"{letter}{layout.first}:{letter}{layout.last}")
        if layout is None and isinstance(column.get("source_column"), int):
            # Sin rango verificable sólo se informa la columna; en un CSV, por su posición.
            position = column["source_column"]
            return CitationVariable(name=name, column=_column_letter(position) if spreadsheet else str(position))
        return CitationVariable(name=name)

    def _table(self, table):
        origin = self._names.origin(self._steps.get(table.operation_id))
        grains = {item.output_name: item.time_grain for item in origin.group_by if item.time_grain != "raw"} if origin else {}
        columns = []
        for field in table.columns:
            unit, documented = display_unit(field, self._language)
            columns.append(ColumnView(
                name=field.name, label=sentence(self._names.result(table.operation_id, field.name, field)),
                unit=unit, unit_documented=documented, numeric=field.sql_type in NUMERIC_TYPES,
                time_grain=grains.get(field.name),
            ))
        return TableView(evidence_id=table.evidence_id, columns=columns, citations=self._evidence.get(table.evidence_id, []))

    def _chart(self, chart):
        source = chart.input_operation_ids[0] if chart.input_operation_ids else ""
        legend = sentence(self._names.result(source, chart.series_field.name, chart.series_field)) if chart.series_field else None
        return ChartView(
            operation_id=chart.operation_id, citations=self._cite([item.dataset_id for item in chart.datasets]),
            x_title=self._axis_title(chart.x_axis), y_title=self._axis_title(chart.y_axis), legend_title=legend,
        )

    def _axis_title(self, axis):
        """Rótulo del eje con su unidad verificada una sola vez, en notación tipográfica."""
        if axis is None:
            return None
        unit, documented = display_unit(axis.field, self._language)
        label, raw = axis.label, axis.field.unit
        if documented and raw and raw in label:
            return label.replace(raw, format_unit(raw))
        return f"{label} ({unit})" if unit else label
