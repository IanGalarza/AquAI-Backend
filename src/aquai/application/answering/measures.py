from dataclasses import dataclass
from decimal import Decimal

from aquai.application.answering.naming import FieldNames
from aquai.application.answering.presentation import format_number
from aquai.application.answering.wording import NUMERIC_TYPES, column_label
from aquai.application.execution import decode_rows, execution_records
from aquai.application.matching import contains_phrase, variable_name, variable_variants
from aquai.application.units import concentration_basis, presentation_unit_status, unitless_name

_COMPARABLE_FUNCTIONS = {None, "mean", "median", "min", "max", "quantile"}
_MIN_HEAD_LENGTH = 4


@dataclass(frozen=True)
class MeasuredResult:
    """Valor medido en un resultado validado, con lo necesario para contrastarlo con la literatura."""

    evidence_id: str
    operation_id: str
    field: str
    label: str
    variable: str
    column: str
    unit: str | None
    unit_status: str
    basis: str | None
    function: str | None
    values: tuple[Decimal, ...]
    sources: tuple[tuple[str, str], ...]

    #
    # Devuelve los nombres con que los documentos pueden referirse a la variable.
    #
    @property
    def names(self) -> list[str]:
        column = unitless_name(self.column).strip()
        words = self.variable.split()
        head = words[0] if words and len(words[0]) >= _MIN_HEAD_LENGTH else None
        names = [*variable_variants(self.variable), *variable_variants(column)]
        if head is not None:
            names += variable_variants(head)

        return list(dict.fromkeys(names))

    #
    # Indica si el resultado es un valor sin unidad confirmada.
    #
    @property
    def lacks_unit(self) -> bool:
        return self.unit_status == "unknown"


#
# Devuelve los valores medidos de las salidas validadas, sin conteos ni dispersiones.
#
def measured_results(state, language="es") -> list[MeasuredResult]:
    validation = state.validation
    if validation is None or state.planning is None:
        return []

    context = _Context(state, language)
    tables = [item for item in validation.usable_evidence if item.kind == "table"]
    requested = [item for item in tables if item.is_requested_output]
    measured = [item for evidence in requested or tables for item in context.measures(evidence)]

    return measured or context.chart_measures(state, validation)


#
# Devuelve el valor formateado para el texto.
#
def shown_value(value, language="es") -> str:
    return format_number(value, language)


class _Context:
    #
    # Guarda los pasos, los resultados y las columnas del turno.
    #
    def __init__(self, state, language):
        plan = state.planning
        self._steps = {step.id: step for step in plan.operations}
        self._records = execution_records(state)
        self._columns = {
            (item.dataset_id, column.sql_name): column
            for item in plan.selected_datasets
            for column in item.columns_metadata
        }
        self._names = FieldNames(self._steps, self._label, language)
        self._mentions = [
            item.text
            for item in (state.routing.data_mentions if state.routing else [])
            if not isinstance(item, str) and item.kind == "variable"
        ]

    #
    # Devuelve la etiqueta de una columna, o None si no se conoce.
    #
    def _label(self, dataset_id, sql_name):
        column = self._named(dataset_id, sql_name)

        return column_label(column.original_name, column.unit) if column else None

    #
    # Devuelve la columna que da nombre a una medición: la de origen si es una proyección numérica.
    #
    def _named(self, dataset_id, sql_name):
        column = self._columns.get((dataset_id, sql_name))

        return self._columns.get((dataset_id, column.derived_from), column) if column else None

    #
    # Devuelve los valores medidos de una evidencia de tabla.
    #
    def measures(self, evidence):
        step = self._steps[evidence.operation_id]
        record = self._records[evidence.operation_id]
        origin = self._names.origin(step)
        calculations = {item.output_name: item for item in origin.calculations}
        grouped = {item.output_name for item in origin.group_by}
        rows = decode_rows(record, record.columns)

        return [
            self._measure(evidence, field, calculations.get(field.name), rows)
            for field in record.columns
            if self._measurable(field, calculations.get(field.name), grouped)
        ]

    #
    # Indica si un campo es una magnitud medida de una sola columna.
    #
    def _measurable(self, field, calculation, grouped) -> bool:
        function = calculation.function if calculation else None

        return (
            field.numeric_eligible
            and field.sql_type in NUMERIC_TYPES
            and field.name not in grouped
            and function in _COMPARABLE_FUNCTIONS
            and len(field.sources) == 1
            and (field.sources[0].dataset_id, field.sources[0].column) in self._columns
        )

    #
    # Construye el valor medido de un campo.
    #
    def _measure(self, evidence, field, calculation, rows):
        ref = field.sources[0]
        column = self._named(ref.dataset_id, ref.column)
        values = [row[field.name] for row in rows if row[field.name] is not None]

        return MeasuredResult(
            evidence_id=evidence.evidence_id,
            operation_id=evidence.operation_id,
            field=field.name,
            label=self._names.result(evidence.operation_id, field.name, field),
            variable=self._variable(column.original_name),
            column=column.original_name,
            unit=field.unit,
            unit_status=presentation_unit_status(field),
            basis=concentration_basis(column.original_name, field.unit),
            function=calculation.function if calculation else None,
            values=tuple(Decimal(str(value)) for value in values),
            sources=((str(ref.dataset_id), ref.column),),
        )

    #
    # Devuelve los valores medidos de un gráfico que no tiene una tabla autorizada.
    #
    def chart_measures(self, state, validation):
        execution = state.python_execution
        charts = {item.operation_id: item for item in execution.visualizations} if execution else {}
        found = []
        for evidence in validation.usable_evidence:
            chart = charts.get(evidence.operation_id)
            if chart is None:
                continue

            histogram = chart.chart_type == "histogram"
            field = chart.x_axis.field if histogram else chart.y_axis.field
            if self._measurable(field, None, set()):
                points = [] if histogram else [point for item in chart.series for point in item.points]
                rows = [{field.name: point.y} for point in points]
                found.append(self._measure(evidence, field, None, rows))

        return found

    #
    # Devuelve la variable como la nombró el usuario si es la de la columna; si no, su encabezado.
    #
    def _variable(self, original_name):
        column = unitless_name(original_name).strip()
        for mention in (variable_name(item) for item in self._mentions):
            names = variable_variants(mention)
            if any(contains_phrase(column, name, plural_insensitive=True) for name in names):
                return mention

        return column
