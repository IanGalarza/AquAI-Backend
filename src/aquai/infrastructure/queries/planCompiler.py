from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Numeric,
    and_,
    case,
    cast,
    func,
    literal_column,
    or_,
    select,
)
from sqlalchemy.sql import Select

from aquai.application.contracts.planning import ColumnRef, OutputField, ValidatedStep

_pg = func.pg_catalog

_FILTER_OPERATORS = {
    "eq": lambda column, values: column == values[0],
    "ne": lambda column, values: column != values[0],
    "ieq": lambda column, values: _pg.lower(column) == values[0].lower(),
    "lt": lambda column, values: column < values[0],
    "le": lambda column, values: column <= values[0],
    "gt": lambda column, values: column > values[0],
    "ge": lambda column, values: column >= values[0],
    "in": lambda column, values: column.in_(values),
    "not_in": lambda column, values: column.not_in(values),
    "is_null": lambda column, values: column.is_(None),
    "is_not_null": lambda column, values: column.is_not(None),
}

_AGGREGATES = {"sum": _pg.sum, "mean": _pg.avg, "min": _pg.min, "max": _pg.max}


class UnsupportedSqlOperation(ValueError):
    pass


@dataclass
class CompiledStep:
    statement: Select
    ancestors: set[str]
    dataset_ids: set[UUID]
    columns: list[OutputField]
    warnings: list[str] = field(default_factory=list)


#
# Compila un paso validado del plan a una sentencia SQL, rechazando operaciones no habilitadas.
#
def compile_step(
    step: ValidatedStep, compiled: dict[str, CompiledStep], tables: dict
) -> CompiledStep:
    if step.tool != "sql" or step.operation in {"join", "append"}:
        raise UnsupportedSqlOperation("La operación no está habilitada para ejecución SQL.")

    return _StepCompiler(step, compiled, tables).compile()


class _StepCompiler:
    #
    # Prepara el origen, los ancestros y los datasets del paso según sea de lectura o dependa de otro.
    #
    def __init__(self, step: ValidatedStep, compiled: dict[str, CompiledStep], tables: dict):
        self._step = step
        self._parent: CompiledStep | None = None
        self._predicates = []
        self._grouping = []
        self._warnings = []

        if step.operation == "read":
            dataset_id = step.columns[0].dataset_id
            self._source = tables[dataset_id]
            self._ancestors = {step.id}
            self._dataset_ids = {dataset_id}
        else:
            self._parent = compiled[step.depends_on[0]]
            self._source = self._parent.statement.subquery()
            self._ancestors = self._parent.ancestors | {step.id}
            self._dataset_ids = self._parent.dataset_ids

    #
    # Compila el paso en una sentencia SQL con filtros, proyecciones, agrupación, orden y límite.
    #
    def compile(self) -> CompiledStep:
        step = self._step
        self._predicates = [self._filter_predicate(predicate) for predicate in step.filters]

        if step.operation == "calculate" and step.calculations[0].function != "correlation":
            _ensure_single_row_cohort(step)

        if step.operation == "latest":
            return self._compile_latest()

        projections = self._projections()
        statement = select(*projections).select_from(self._source).where(*self._predicates)
        if self._grouping:
            statement = statement.group_by(*self._grouping)
        statement = self._ordered(statement)
        statement = self._limited(statement)

        return CompiledStep(
            statement, self._ancestors, self._dataset_ids, step.output_fields, self._warnings
        )

    #
    # Resuelve una referencia de columna o de campo a la columna autorizada del origen.
    #
    def _resolve(self, ref):
        step = self._step
        if isinstance(ref, ColumnRef):
            if step.operation != "read" or ref.dataset_id not in self._dataset_ids:
                raise UnsupportedSqlOperation("Referencia de columna fuera del dataset autorizado.")
            return self._source.c[ref.column]

        if step.operation == "read" or ref.step_id != step.depends_on[0]:
            raise UnsupportedSqlOperation("Referencia fuera de la dependencia autorizada.")
        return self._source.c[ref.field]

    #
    # Convierte un filtro del plan en un predicado SQL.
    #
    def _filter_predicate(self, predicate):
        column = self._resolve(predicate.field)
        values = [_filter_value(column, value) for value in predicate.values]
        return _FILTER_OPERATORS[predicate.operator](column, values)

    #
    # Compila el paso que devuelve las filas con la fecha máxima del alcance filtrado.
    #
    def _compile_latest(self) -> CompiledStep:
        step = self._step
        cohort = (
            select(*(self._source.c[column.name] for column in step.output_fields))
            .where(*self._predicates)
            .subquery()
        )
        name = step.order_by[0].field.field
        maximum = select(_pg.max(cohort.c[name])).correlate(None).scalar_subquery()
        statement = (
            select(*(cohort.c[column.name] for column in step.output_fields))
            .where(cohort.c[name] == maximum)
            .order_by(cohort.c[name].desc())
        )
        return CompiledStep(statement, self._ancestors, self._dataset_ids, step.output_fields)

    #
    # Devuelve las columnas que proyecta el paso según su operación.
    #
    def _projections(self) -> list:
        step = self._step
        if step.operation == "read":
            return [self._resolve(ref).label(ref.column) for ref in step.columns]
        if step.operation == "sort":
            return [self._source.c[column.name] for column in step.output_fields]

        groups = [self._group_projection(group) for group in step.group_by]
        calculations = [self._calculation_projection(item) for item in step.calculations]
        return groups + calculations

    #
    # Construye la proyección de un grupo, con su intervalo de tiempo, y lo registra para agrupar.
    #
    def _group_projection(self, group):
        expression = _time_bucket(self._resolve(group.field), group.time_grain)
        self._grouping.append(expression)
        return expression.label(group.output_name)

    #
    # Construye la proyección de un cálculo, aplicando su política de nulos y su tipo de salida.
    #
    def _calculation_projection(self, calculation):
        operands = [self._resolve(ref) for ref in calculation.inputs]
        expression = self._calculation_expression(calculation, operands)

        scalar_aggregate = (
            self._step.operation == "aggregate" or calculation.function == "correlation"
        )
        if scalar_aggregate and calculation.null_policy == "preserve" and operands:
            expression = _null_if_any_input_missing(expression, operands)

        output = next(
            column for column in self._step.output_fields if column.name == calculation.output_name
        )
        if output.sql_type == "NUMERIC":
            expression = cast(expression, Numeric)
        return expression.label(calculation.output_name)

    #
    # Construye la expresión SQL de una función de cálculo como conteo, agregado o correlación.
    #
    def _calculation_expression(self, calculation, operands):
        function = calculation.function
        if function == "count":
            return _pg.count(operands[0]) if operands else _pg.count(literal_column("*"))
        if function == "count_distinct":
            return _pg.count(operands[0].distinct())
        if function in _AGGREGATES:
            return _AGGREGATES[function](operands[0])
        if function == "correlation":
            expression = cast(_pg.corr(*operands), Numeric)
            self._warnings.append(
                "La correlación de PostgreSQL utiliza doble precisión; no es un cálculo decimal exacto."
            )
            return expression
        return self._row_arithmetic(calculation, operands)

    #
    # Construye la aritmética entre dos columnas por fila, verificando unidades y política de nulos.
    #
    def _row_arithmetic(self, calculation, operands):
        self._ensure_compatible_units(calculation)
        left, right = (cast(operand, Numeric) for operand in operands)
        expression = self._arithmetic(calculation.function, left, right)

        if calculation.null_policy == "exclude":
            self._predicates.append(and_(*(operand.is_not(None) for operand in operands)))
        return expression

    #
    # Construye la diferencia o el cociente de dos operandos, evitando la división por cero.
    #
    def _arithmetic(self, function, left, right):
        if function == "difference":
            return left - right
        if function == "ratio":
            expression = left / func.nullif(right, 0)
            self._warnings.append(
                "La división por cero produce NULL; no se sustituye por cero ni se devuelve infinito."
            )
            return expression
        raise UnsupportedSqlOperation("Función no soportada por el ejecutor SQL.")

    #
    # Verifica que los dos operandos de un cálculo tengan la misma unidad verificada.
    #
    def _ensure_compatible_units(self, calculation) -> None:
        fields = [self._parent_field(ref.field) for ref in calculation.inputs]
        left_unit = (fields[0].unit, fields[0].unit_status)
        right_unit = (fields[1].unit, fields[1].unit_status)
        if left_unit != right_unit:
            raise UnsupportedSqlOperation(
                "La operación necesita compatibilidad de unidades verificada; no se ejecuta una transformación implícita."
            )

    #
    # Devuelve el campo de salida del paso padre con ese nombre.
    #
    def _parent_field(self, name: str) -> OutputField:
        return next(column for column in self._parent.columns if column.name == name)

    #
    # Aplica al SQL el orden pedido por el paso, si lo hay.
    #
    def _ordered(self, statement: Select) -> Select:
        if not self._step.order_by:
            return statement
        return statement.order_by(*(self._ordering(order) for order in self._step.order_by))

    #
    # Construye la expresión de orden de un campo, dejando los nulos al final.
    #
    def _ordering(self, order):
        expression = self._resolve(order.field)
        directed = expression.asc() if order.direction == "asc" else expression.desc()
        return directed.nulls_last()

    #
    # Aplica al SQL el límite del paso, conservando los empates si hay orden.
    #
    def _limited(self, statement: Select) -> Select:
        step = self._step
        if step.limit is None:
            return statement
        if step.order_by:
            return statement.fetch(step.limit, with_ties=True)
        return statement.limit(step.limit)


#
# Convierte el valor de un filtro al tipo de la columna, rechazando fechas con zona horaria.
#
def _filter_value(column, value):
    if isinstance(column.type, Date):
        return date.fromisoformat(value)
    if isinstance(column.type, DateTime):
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is not None:
            raise UnsupportedSqlOperation(
                "El filtro incluye zona horaria para una columna sin zona documentada."
            )
        return moment
    if isinstance(column.type, Numeric) and type(value) is float:
        return Decimal(str(value))
    return value


#
# Verifica que los cálculos por fila de un paso compartan operandos y política de nulos.
#
def _ensure_single_row_cohort(step: ValidatedStep) -> None:
    cohorts = {_row_cohort(calculation) for calculation in step.calculations}
    if len(cohorts) > 1:
        raise UnsupportedSqlOperation(
            "Los cálculos por fila con operandos o políticas de nulos diferentes deben planificarse por separado para no alterar sus cohortes."
        )


#
# Devuelve la política de nulos y los operandos que definen la cohorte de un cálculo.
#
def _row_cohort(calculation):
    operands = frozenset((ref.step_id, ref.field) for ref in calculation.inputs)
    return calculation.null_policy, operands


#
# Agrupa una expresión de fecha por año, mes o día según el intervalo pedido.
#
def _time_bucket(expression, time_grain: str):
    if time_grain == "year":
        return cast(_pg.date_part("year", expression), BigInteger)
    if time_grain in {"day", "month"}:
        return cast(_pg.date_trunc(time_grain, expression), Date)
    return expression


#
# Anula el resultado agregado si alguna fila tiene un operando faltante.
#
def _null_if_any_input_missing(expression, operands):
    any_missing = or_(*(operand.is_(None) for operand in operands))
    return case((_pg.count(literal_column("*")).filter(any_missing) > 0, None), else_=expression)
