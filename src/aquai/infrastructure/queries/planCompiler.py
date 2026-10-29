"""Compila exclusivamente operadores autorizados como SELECT de SQLAlchemy."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import BigInteger, Date, DateTime, Numeric, and_, case, cast, func, literal, literal_column, or_, select
from sqlalchemy.sql import Select

from aquai.application.contracts.planning import ColumnRef, OutputField, ValidatedStep


class UnsupportedSqlOperation(ValueError):
    pass


@dataclass
class CompiledStep:
    statement: Select
    ancestors: set[str]
    dataset_ids: set[UUID]
    columns: list[OutputField]
    warnings: list[str] = field(default_factory=list)


def compile_step(step: ValidatedStep, compiled: dict[str, CompiledStep], tables: dict) -> CompiledStep:
    pg = func.pg_catalog
    if step.tool != "sql" or step.operation in {"join", "append"}:
        raise UnsupportedSqlOperation("La operación no está habilitada para ejecución SQL.")
    if step.operation == "read":
        dataset_id = step.columns[0].dataset_id
        source = tables[dataset_id]
        ancestors, dataset_ids = {step.id}, {dataset_id}
    else:
        parent = compiled[step.depends_on[0]]
        source = parent.statement.subquery()
        ancestors = parent.ancestors | {step.id}
        dataset_ids = parent.dataset_ids

    def resolve(ref):
        if isinstance(ref, ColumnRef):
            if step.operation != "read" or ref.dataset_id not in dataset_ids:
                raise UnsupportedSqlOperation("Referencia de columna fuera del dataset autorizado.")
            return source.c[ref.column]
        if step.operation == "read" or ref.step_id != step.depends_on[0]:
            raise UnsupportedSqlOperation("Referencia fuera de la dependencia autorizada.")
        return source.c[ref.field]

    predicates = []
    for predicate in step.filters:
        column = resolve(predicate.field)
        values = []
        for value in predicate.values:
            if isinstance(column.type, Date):
                value = date.fromisoformat(value)
            elif isinstance(column.type, DateTime):
                value = datetime.fromisoformat(value)
                if value.tzinfo is not None:
                    raise UnsupportedSqlOperation("El filtro incluye zona horaria para una columna sin zona documentada.")
            elif isinstance(column.type, Numeric) and type(value) is float:
                value = Decimal(str(value))
            values.append(value)
        operators = {
            "eq": lambda: column == values[0], "ne": lambda: column != values[0],
            "ieq": lambda: pg.lower(column) == values[0].lower(),
            "lt": lambda: column < values[0], "le": lambda: column <= values[0],
            "gt": lambda: column > values[0], "ge": lambda: column >= values[0],
            "in": lambda: column.in_(values), "not_in": lambda: column.not_in(values),
            "is_null": lambda: column.is_(None), "is_not_null": lambda: column.is_not(None),
        }
        predicates.append(operators[predicate.operator]())

    warnings = []
    grouping = []
    if step.operation == "calculate" and step.calculations[0].function != "correlation":
        cohorts = {(calculation.null_policy, frozenset((ref.step_id, ref.field) for ref in calculation.inputs))
                   for calculation in step.calculations}
        if len(cohorts) > 1:
            raise UnsupportedSqlOperation("Los cálculos por fila con operandos o políticas de nulos diferentes deben planificarse por separado para no alterar sus cohortes.")
    if step.operation == "latest":
        # El máximo se calcula sobre el mismo alcance filtrado y se conservan
        # todas sus filas, incluidos duplicados y mediciones con igual fecha.
        cohort = select(*(source.c[column.name] for column in step.output_fields)).where(*predicates).subquery()
        name = step.order_by[0].field.field
        maximum = select(pg.max(cohort.c[name])).correlate(None).scalar_subquery()
        statement = select(*(cohort.c[column.name] for column in step.output_fields)).where(
            cohort.c[name] == maximum).order_by(cohort.c[name].desc())
        return CompiledStep(statement, ancestors, dataset_ids, step.output_fields)
    if step.operation == "read":
        projections = [resolve(ref).label(ref.column) for ref in step.columns]
    elif step.operation == "sort":
        projections = [source.c[column.name] for column in step.output_fields]
    else:
        projections = []
        for group in step.group_by:
            expression = resolve(group.field)
            if group.time_grain == "year":
                expression = cast(pg.date_part("year", expression), BigInteger)
            elif group.time_grain in {"day", "month"}:
                expression = cast(pg.date_trunc(group.time_grain, expression), Date)
            grouping.append(expression)
            projections.append(expression.label(group.output_name))
        for calculation in step.calculations:
            operands = [resolve(ref) for ref in calculation.inputs]
            nulls = or_(*(operand.is_(None) for operand in operands)) if operands else literal(False)
            function = calculation.function
            if function == "count":
                expression = pg.count(operands[0]) if operands else pg.count(literal_column("*"))
            elif function == "count_distinct":
                expression = pg.count(operands[0].distinct())
            elif function in {"sum", "mean", "min", "max"}:
                expression = {"sum": pg.sum, "mean": pg.avg, "min": pg.min, "max": pg.max}[function](operands[0])
            elif function == "correlation":
                expression = cast(pg.corr(*operands), Numeric)
                warnings.append("La correlación de PostgreSQL utiliza doble precisión; no es un cálculo decimal exacto.")
            else:
                fields = [next(field for field in parent.columns if field.name == ref.field) for ref in calculation.inputs]
                if (fields[0].unit, fields[0].unit_status) != (fields[1].unit, fields[1].unit_status):
                    raise UnsupportedSqlOperation("La operación necesita compatibilidad de unidades verificada; no se ejecuta una transformación implícita.")
                left, right = (cast(operand, Numeric) for operand in operands)
                if function == "difference":
                    expression = left - right
                elif function == "ratio":
                    expression = left / func.nullif(right, 0)
                    warnings.append("La división por cero produce NULL; no se sustituye por cero ni se devuelve infinito.")
                else:
                    raise UnsupportedSqlOperation("Función no soportada por el ejecutor SQL.")
                if calculation.null_policy == "exclude":
                    predicates.append(and_(*(operand.is_not(None) for operand in operands)))
            scalar_aggregate = step.operation == "aggregate" or function == "correlation"
            if scalar_aggregate and calculation.null_policy == "preserve" and operands:
                expression = case((pg.count(literal_column("*")).filter(nulls) > 0, None), else_=expression)
            output = next(column for column in step.output_fields if column.name == calculation.output_name)
            if output.sql_type == "NUMERIC":
                expression = cast(expression, Numeric)
            projections.append(expression.label(calculation.output_name))

    statement = select(*projections).select_from(source).where(*predicates)
    if grouping:
        statement = statement.group_by(*grouping)
    if step.order_by:
        statement = statement.order_by(*(
            resolve(order.field).asc() if order.direction == "asc" else resolve(order.field).desc()
            for order in step.order_by
        ))
    if step.limit is not None:
        statement = statement.limit(step.limit)
    return CompiledStep(statement, ancestors, dataset_ids, step.output_fields, warnings)
