"""Operadores tabulares cerrados. Pandas organiza filas; Decimal calcula números."""

from datetime import date, datetime
from decimal import Decimal, localcontext
import re

import pandas as pd

from aquai.application.contracts.execution import ValueExclusion


class AnalysisBlocked(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def decode_rows(record, fields, budget):
    """Sólo rehidrata el tipo del esquema verificado, nunca interpreta un TEXT."""
    rows = []
    names = {field.name for field in fields}
    for row in record.rows:
        budget.check()
        if set(row) != names:
            raise AnalysisBlocked("invalid_schema", "Una fila no coincide con el esquema de su resultado autorizado.")
        decoded = {}
        for field in fields:
            value = row[field.name]
            if value is None:
                decoded[field.name] = None
                continue
            if field.sql_type == "NUMERIC":
                if type(value) is str:
                    if len(value) > 1100 or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
                        raise AnalysisBlocked("invalid_numeric", "La representación NUMERIC no es decimal canónica.")
                    value = Decimal(value)
                elif type(value) is int:
                    value = Decimal(value)
                if not isinstance(value, Decimal) or not value.is_finite():
                    raise AnalysisBlocked("invalid_numeric", "NUMERIC exige decimales finitos; no se convierte float silenciosamente.")
                if len(value.as_tuple().digits) > 1000 or abs(value.as_tuple().exponent) > 1000:
                    raise AnalysisBlocked("numeric_limit", "El número excede la precisión/tamaño de entrada permitidos.")
            elif field.sql_type == "BIGINT":
                if type(value) is not int or not -(2**63) <= value < 2**63:
                    raise AnalysisBlocked("invalid_integer", "BIGINT exige enteros dentro de su rango original.")
            elif field.sql_type == "DATE":
                value = date.fromisoformat(value) if type(value) is str else value
                if type(value) is not date:
                    raise AnalysisBlocked("invalid_date", "La fecha no coincide con DATE.")
            elif field.sql_type == "TIMESTAMP":
                value = datetime.fromisoformat(value) if type(value) is str else value
                if not isinstance(value, datetime) or value.tzinfo is not None:
                    raise AnalysisBlocked("invalid_date", "TIMESTAMP requiere su fecha completa sin zona horaria implícita.")
            elif field.sql_type == "BOOLEAN":
                if type(value) is not bool:
                    raise AnalysisBlocked("invalid_boolean", "La entrada no coincide con BOOLEAN.")
            elif field.sql_type == "TEXT":
                if type(value) is not str:
                    raise AnalysisBlocked("invalid_text", "La entrada no coincide con TEXT.")
            else:
                raise AnalysisBlocked("unsupported_type", "El tipo de entrada no está habilitado para Python.")
            decoded[field.name] = value
        rows.append(decoded)
    return rows


def filter_rows(rows, filters, fields, budget):
    types = {field.name: field.sql_type for field in fields}
    result = rows
    for predicate in filters:
        name = predicate.field.field
        values = list(predicate.values)
        kind = types[name]
        if kind == "NUMERIC":
            values = [Decimal(str(value)) for value in values]
        elif kind == "DATE":
            values = [date.fromisoformat(value) for value in values]
        elif kind == "TIMESTAMP":
            values = [datetime.fromisoformat(value) for value in values]
            if any(value.tzinfo is not None for value in values):
                raise AnalysisBlocked("ambiguous_timezone", "El filtro no puede introducir una zona horaria no documentada.")

        def matches(row):
            budget.check()
            value = row[name]
            op = predicate.operator
            if op == "is_null":
                return value is None
            if op == "is_not_null":
                return value is not None
            if value is None:  # Semántica SQL de NULL en comparaciones.
                return False
            return {
                "eq": lambda: value == values[0], "ne": lambda: value != values[0],
                "lt": lambda: value < values[0], "le": lambda: value <= values[0],
                "gt": lambda: value > values[0], "ge": lambda: value >= values[0],
                "in": lambda: value in values, "not_in": lambda: value not in values,
            }[op]()
        result = [row for row in result if matches(row)]
    return result


def _quantile(values, probability):
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistic(rows, calculation, precision, budget):
    names = [ref.field for ref in calculation.inputs]
    pairs = []
    nulls = 0
    for row in rows:
        budget.check()
        values = [row[name] for name in names]
        if any(value is None for value in values):
            nulls += 1
        else:
            pairs.append(values)
    exclusion = ValueExclusion(output_name=calculation.output_name, input_rows=len(rows), valid_rows=len(pairs),
                               null_rows=nulls, policy=calculation.null_policy)
    warnings = []
    if nulls and calculation.null_policy == "preserve":
        return None, exclusion, warnings
    function = calculation.function
    if function == "count":
        return len(pairs), exclusion, warnings
    if function == "count_distinct":
        return len({values[0] for values in pairs}), exclusion, warnings
    if not pairs:
        return None, exclusion, ["No hay observaciones válidas para el cálculo; se devuelve null."]
    if function in {"min", "max"}:
        return (min if function == "min" else max)(values[0] for values in pairs), exclusion, warnings
    with localcontext() as context:
        context.prec = precision
        numeric = [[Decimal(value) for value in values] for values in pairs]
        values = [pair[0] for pair in numeric]
        size = Decimal(len(values))
        if function == "sum":
            value = sum(values, Decimal(0))
        elif function == "mean":
            value = sum(values, Decimal(0)) / size
        elif function in {"median", "quantile"}:
            value = _quantile(values, Decimal("0.5") if function == "median" else Decimal(str(calculation.quantile)))
        elif function == "stddev":
            if len(values) <= calculation.ddof:
                return None, exclusion, ["El número de observaciones válidas no supera ddof; el desvío es indefinido."]
            mean = sum(values, Decimal(0)) / size
            value = (sum(((item - mean) ** 2 for item in values), Decimal(0)) / (size - calculation.ddof)).sqrt()
        elif function == "correlation":
            if len(values) < 2:
                return None, exclusion, ["La correlación necesita al menos dos pares completos."]
            right = [pair[1] for pair in numeric]
            mx, my = sum(values, Decimal(0)) / size, sum(right, Decimal(0)) / size
            dx, dy = [item - mx for item in values], [item - my for item in right]
            scale = (sum((item * item for item in dx), Decimal(0)) * sum((item * item for item in dy), Decimal(0))).sqrt()
            if not scale:
                return None, exclusion, ["La correlación es indefinida cuando una variable no tiene variación."]
            value = sum((x * y for x, y in zip(dx, dy, strict=True)), Decimal(0)) / scale
        else:
            raise AnalysisBlocked("unsupported_operation", "La función no está habilitada como estadística escalar.")
        budget.check()
        return value, exclusion, warnings


def tabular_operation(step, rows, parent_fields, settings, budget):
    filtered = filter_rows(rows, step.filters, parent_fields, budget)
    frame = pd.DataFrame(filtered, columns=[field.name for field in parent_fields], dtype=object)
    # Incluye margen para claves, ordenamientos y copias temporales acotadas.
    memory = int(frame.memory_usage(index=True, deep=True).sum())
    if memory * 4 > settings.max_memory_bytes:
        raise AnalysisBlocked("memory_limit", "La tabla y sus copias de trabajo exceden el presupuesto de memoria.")
    warnings, exclusions = [], []
    if step.operation == "sort":
        for ordering in reversed(step.order_by):
            budget.check()
            ascending = ordering.direction == "asc"
            frame = frame.sort_values(ordering.field.field, ascending=ascending, kind="stable",
                                      na_position="last" if ascending else "first")
        if step.limit is not None:
            frame = frame.head(step.limit)
        return frame.to_dict("records"), exclusions, warnings, len(rows) - len(filtered)
    if step.operation == "calculate" and step.calculations[0].function != "correlation":
        cohorts = {(calc.null_policy, frozenset(ref.field for ref in calc.inputs)) for calc in step.calculations}
        if len(cohorts) != 1:
            raise AnalysisBlocked("incompatible_cohorts", "Separar cálculos por fila con operandos o políticas de nulos distintos para no alterar las cohortes.")
        output = []
        for calculation in step.calculations:
            fields = [next(field for field in parent_fields if field.name == ref.field) for ref in calculation.inputs]
            if (fields[0].unit, fields[0].unit_status) != (fields[1].unit, fields[1].unit_status):
                raise AnalysisBlocked("incompatible_units", "La operación exige unidades compatibles verificadas; no se convierten implícitamente.")
            exclusions.append(ValueExclusion(output_name=calculation.output_name, input_rows=len(filtered),
                valid_rows=0, null_rows=0, policy=calculation.null_policy))
        with localcontext() as context:
            context.prec = settings.decimal_precision
            for row in filtered:
                budget.check()
                values, skip = {}, False
                for calculation, exclusion in zip(step.calculations, exclusions, strict=True):
                    left, right = (row[ref.field] for ref in calculation.inputs)
                    value = None
                    if left is None or right is None:
                        exclusion.null_rows += 1
                        skip = calculation.null_policy == "exclude"
                    else:
                        exclusion.valid_rows += 1
                        left, right = Decimal(left), Decimal(right)
                        if calculation.function == "ratio" and right == 0:
                            exclusion.zero_divisions += 1
                        else:
                            value = left - right if calculation.function == "difference" else left / right
                    values[calculation.output_name] = value
                if not skip:
                    output.append(values)
        if any(item.zero_divisions for item in exclusions):
            warnings.append("Las divisiones por cero producen null y quedan contadas; no se sustituyen por cero.")
        return output, exclusions, warnings, len(rows) - len(filtered)
    groups = {}
    for row in filtered:
        budget.check()
        key = []
        for group in step.group_by:
            value = row[group.field.field]
            if value is not None and group.time_grain != "raw":
                value = (value.year if group.time_grain == "year" else
                         date(value.year, value.month, 1) if group.time_grain == "month" else
                         date(value.year, value.month, value.day))
            key.append(value)
        groups.setdefault(tuple(key), []).append(row)
    if not step.group_by and not groups:
        groups[()] = []
    output = []
    for key, group_rows in groups.items():
        budget.check()
        row = {group.output_name: value for group, value in zip(step.group_by, key, strict=True)}
        for calculation in step.calculations:
            value, exclusion, messages = _statistic(group_rows, calculation, settings.decimal_precision, budget)
            exclusion.group = {group.output_name: item for group, item in zip(step.group_by, key, strict=True)}
            row[calculation.output_name] = value
            exclusions.append(exclusion)
            warnings.extend(messages)
        output.append(row)
    return output, exclusions, list(dict.fromkeys(warnings)), len(rows) - len(filtered)
