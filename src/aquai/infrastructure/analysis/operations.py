from datetime import date, datetime
from decimal import Decimal, localcontext

import pandas as pd

from aquai.application.contracts.execution import ValueExclusion
from aquai.application.errors import AnalysisBlocked

_FILTER_PARSERS = {
    "NUMERIC": lambda value: Decimal(str(value)),
    "DATE": date.fromisoformat,
    "TIMESTAMP": datetime.fromisoformat,
}

_COMPARISONS = {
    "eq": lambda value, values: value == values[0],
    "ne": lambda value, values: value != values[0],
    "ieq": lambda value, values: value.lower() == values[0].lower(),
    "lt": lambda value, values: value < values[0],
    "le": lambda value, values: value <= values[0],
    "gt": lambda value, values: value > values[0],
    "ge": lambda value, values: value >= values[0],
    "in": lambda value, values: value in values,
    "not_in": lambda value, values: value not in values,
}


#
# Filtra las filas aplicando en secuencia los predicados del paso.
#
def filter_rows(rows, filters, fields, budget):
    types = {field.name: field.sql_type for field in fields}
    result = rows

    for predicate in filters:
        name = predicate.field.field
        values = _typed_filter_values(predicate.values, types[name])
        result = [row for row in result if _matches(row, name, predicate.operator, values, budget)]

    return result


#
# Convierte los valores de un filtro al tipo de la columna.
#
def _typed_filter_values(raw_values, kind):
    parser = _FILTER_PARSERS.get(kind)

    if parser is None:
        return list(raw_values)

    values = [parser(value) for value in raw_values]

    if kind == "TIMESTAMP" and any(value.tzinfo is not None for value in values):
        raise AnalysisBlocked(
            "ambiguous_timezone",
            "El filtro no puede introducir una zona horaria no documentada.",
        )

    return values


#
# Indica si una fila cumple un predicado, con semántica SQL para los nulos.
#
def _matches(row, name, operator, values, budget):
    budget.check()
    value = row[name]

    if operator == "is_null":
        return value is None

    if operator == "is_not_null":
        return value is not None

    if value is None:
        return False

    return _COMPARISONS[operator](value, values)


#
# Calcula un cuantil con interpolación lineal entre valores.
#
def _quantile(values, probability):
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


#
# Separa las filas con todas las entradas presentes y cuenta las que tienen nulos.
#
def _complete_inputs(rows, names, budget):
    pairs = []
    nulls = 0

    for row in rows:
        budget.check()
        values = [row[name] for name in names]

        if any(value is None for value in values):
            nulls += 1
        else:
            pairs.append(values)

    return pairs, nulls


#
# Calcula una estadística escalar sobre las filas y devuelve su valor y exclusiones.
#
def _statistic(rows, calculation, precision, budget):
    pairs, nulls = _complete_inputs(rows, [ref.field for ref in calculation.inputs], budget)
    exclusion = ValueExclusion(
        output_name=calculation.output_name,
        input_rows=len(rows),
        valid_rows=len(pairs),
        null_rows=nulls,
        policy=calculation.null_policy,
    )
    warnings = []
    function = calculation.function

    if nulls and calculation.null_policy == "preserve":
        return None, exclusion, warnings

    if function == "count":
        return len(pairs), exclusion, warnings

    if function == "count_distinct":
        return len({values[0] for values in pairs}), exclusion, warnings

    if not pairs:
        return None, exclusion, ["No hay observaciones válidas para el cálculo; se devuelve null."]

    if function in {"min", "max"}:
        extreme = min if function == "min" else max
        return extreme(values[0] for values in pairs), exclusion, warnings

    with localcontext() as context:
        context.prec = precision
        numeric = [[Decimal(value) for value in values] for values in pairs]
        compute = _DECIMAL_STATISTICS.get(function)

        if compute is None:
            raise AnalysisBlocked(
                "unsupported_operation", "La función no está habilitada como estadística escalar."
            )

        value, undefined = compute(numeric, calculation)

        if undefined is not None:
            return None, exclusion, [undefined]

        budget.check()

        return value, exclusion, warnings


#
# Devuelve el primer valor de cada fila de entradas.
#
def _first_inputs(numeric):
    return [pair[0] for pair in numeric]


#
# Calcula la suma de la primera entrada.
#
def _sum(numeric, calculation):
    return sum(_first_inputs(numeric), Decimal(0)), None


#
# Calcula el promedio de la primera entrada.
#
def _mean(numeric, calculation):
    values = _first_inputs(numeric)
    return sum(values, Decimal(0)) / Decimal(len(values)), None


#
# Calcula la mediana de la primera entrada.
#
def _median(numeric, calculation):
    return _quantile(_first_inputs(numeric), Decimal("0.5")), None


#
# Calcula el cuantil indicado por el cálculo sobre la primera entrada.
#
def _quantile_statistic(numeric, calculation):
    return _quantile(_first_inputs(numeric), Decimal(str(calculation.quantile))), None


#
# Calcula el desvío estándar según ddof, o indica que es indefinido.
#
def _stddev(numeric, calculation):
    values = _first_inputs(numeric)
    size = Decimal(len(values))

    if len(values) <= calculation.ddof:
        return None, "El número de observaciones válidas no supera ddof; el desvío es indefinido."

    mean = sum(values, Decimal(0)) / size
    squared_deviations = sum(((item - mean) ** 2 for item in values), Decimal(0))

    return (squared_deviations / (size - calculation.ddof)).sqrt(), None


#
# Calcula el coeficiente de correlación entre dos entradas, o indica que es indefinido.
#
def _correlation(numeric, calculation):
    left = _first_inputs(numeric)
    size = Decimal(len(left))

    if len(left) < 2:
        return None, "La correlación necesita al menos dos pares completos."

    right = [pair[1] for pair in numeric]
    left_mean = sum(left, Decimal(0)) / size
    right_mean = sum(right, Decimal(0)) / size
    left_deviations = [item - left_mean for item in left]
    right_deviations = [item - right_mean for item in right]
    scale = (
        sum((item * item for item in left_deviations), Decimal(0))
        * sum((item * item for item in right_deviations), Decimal(0))
    ).sqrt()

    if not scale:
        return None, "La correlación es indefinida cuando una variable no tiene variación."

    products = (x * y for x, y in zip(left_deviations, right_deviations, strict=True))

    return sum(products, Decimal(0)) / scale, None


_DECIMAL_STATISTICS = {
    "sum": _sum,
    "mean": _mean,
    "median": _median,
    "quantile": _quantile_statistic,
    "stddev": _stddev,
    "correlation": _correlation,
}


#
# Aplica a las filas la operación tabular del paso: latest, ordenar, calcular o agregar.
#
def tabular_operation(step, rows, parent_fields, settings, budget, period_key=None):
    filtered = filter_rows(rows, step.filters, parent_fields, budget)
    frame = pd.DataFrame(filtered, columns=[field.name for field in parent_fields], dtype=object)
    _check_memory(frame, settings)

    if step.operation == "latest":
        return _latest(step, rows, filtered, period_key)

    if step.operation == "sort":
        return _sort(step, rows, filtered, frame, budget)

    if step.operation == "calculate" and step.calculations[0].function != "correlation":
        return _calculate(step, rows, filtered, parent_fields, settings, budget)

    return _aggregate(step, rows, filtered, settings, budget)


#
# Verifica que la tabla y sus copias de trabajo no excedan el presupuesto de memoria.
#
def _check_memory(frame, settings):
    memory = int(frame.memory_usage(index=True, deep=True).sum())

    if memory * 4 > settings.max_memory_bytes:
        raise AnalysisBlocked(
            "memory_limit", "La tabla y sus copias de trabajo exceden el presupuesto de memoria."
        )


#
# Devuelve las filas del período más reciente, advirtiendo las que no tienen fecha.
#
def _latest(step, rows, filtered, period_key):
    name = step.order_by[0].field.field
    warnings = []

    undated = sum(1 for row in filtered if row[name] is None)
    if undated:
        warnings.append(f"Se excluyeron {undated} registros sin fecha.")

    key = period_key or (lambda value: value)
    dated = [(row, key(row[name])) for row in filtered if row[name] is not None]

    if period_key is not None:
        ignored = sum(1 for _, value in dated if value is None)
        dated = [(row, value) for row, value in dated if value is not None]

        if ignored:
            warnings.append(
                f"Se excluyeron {ignored} registros cuya fecha no nombra un período interpretable."
            )

    maximum = max((value for _, value in dated), default=None)
    output = [row for row, value in dated if value == maximum]

    return output, [], warnings, len(rows) - len(filtered)


#
# Ordena las filas según los criterios del paso y aplica el límite.
#
def _sort(step, rows, filtered, frame, budget):
    for ordering in reversed(step.order_by):
        budget.check()
        ascending = ordering.direction == "asc"
        frame = frame.sort_values(
            ordering.field.field, ascending=ascending, kind="stable", na_position="last"
        )

    records = frame.to_dict("records")

    if step.limit is not None and len(records) > step.limit:
        records = _limit_with_ties(records, step)

    return records, [], [], len(rows) - len(filtered)


#
# Aplica el límite conservando las filas empatadas con la última incluida.
#
def _limit_with_ties(records, step):
    keys = [ordering.field.field for ordering in step.order_by]
    last = [records[step.limit - 1][key] for key in keys]
    tied = [row for row in records[step.limit :] if [row[key] for key in keys] == last]

    return records[: step.limit] + tied


#
# Calcula por fila las diferencias o razones del paso y contabiliza las exclusiones.
#
def _calculate(step, rows, filtered, parent_fields, settings, budget):
    cohorts = {
        (calculation.null_policy, frozenset(ref.field for ref in calculation.inputs))
        for calculation in step.calculations
    }

    if len(cohorts) != 1:
        raise AnalysisBlocked(
            "incompatible_cohorts",
            "Separar cálculos por fila con operandos o políticas de nulos distintos para no alterar las cohortes.",
        )

    exclusions = [
        _row_exclusion(calculation, parent_fields, len(filtered))
        for calculation in step.calculations
    ]
    output = []
    warnings = []

    with localcontext() as context:
        context.prec = settings.decimal_precision

        for row in filtered:
            budget.check()
            values, skip = _row_calculations(row, step.calculations, exclusions)

            if not skip:
                output.append(values)

    if any(item.zero_divisions for item in exclusions):
        warnings.append(
            "Las divisiones por cero producen null y quedan contadas; no se sustituyen por cero."
        )

    return output, exclusions, warnings, len(rows) - len(filtered)


#
# Crea el contador de exclusiones de un cálculo por fila, exigiendo unidades compatibles.
#
def _row_exclusion(calculation, parent_fields, input_rows):
    fields = [
        next(field for field in parent_fields if field.name == ref.field)
        for ref in calculation.inputs
    ]

    if (fields[0].unit, fields[0].unit_status) != (fields[1].unit, fields[1].unit_status):
        raise AnalysisBlocked(
            "incompatible_units",
            "La operación exige unidades compatibles verificadas; no se convierten implícitamente.",
        )

    return ValueExclusion(
        output_name=calculation.output_name,
        input_rows=input_rows,
        valid_rows=0,
        null_rows=0,
        policy=calculation.null_policy,
    )


#
# Calcula los valores de una fila para cada cálculo e indica si debe omitirse.
#
def _row_calculations(row, calculations, exclusions):
    values, skip = {}, False

    for calculation, exclusion in zip(calculations, exclusions, strict=True):
        left, right = (row[ref.field] for ref in calculation.inputs)
        value = None

        if left is None or right is None:
            exclusion.null_rows += 1
            skip = calculation.null_policy == "exclude"
        else:
            exclusion.valid_rows += 1
            value = _pair_result(calculation.function, Decimal(left), Decimal(right), exclusion)

        values[calculation.output_name] = value

    return values, skip


#
# Calcula la diferencia o razón de un par, contando las divisiones por cero.
#
def _pair_result(function, left, right, exclusion):
    if function == "ratio" and right == 0:
        exclusion.zero_divisions += 1
        return None

    return left - right if function == "difference" else left / right


#
# Agrupa las filas y calcula las estadísticas de cada grupo.
#
def _aggregate(step, rows, filtered, settings, budget):
    groups = _group_rows(step, filtered, budget)
    output, exclusions, warnings = [], [], []

    for key, group_rows in groups.items():
        budget.check()
        row = _group_labels(step, key)

        for calculation in step.calculations:
            value, exclusion, messages = _statistic(
                group_rows, calculation, settings.decimal_precision, budget
            )
            exclusion.group = _group_labels(step, key)
            row[calculation.output_name] = value
            exclusions.append(exclusion)
            warnings.extend(messages)

        output.append(row)

    return output, exclusions, list(dict.fromkeys(warnings)), len(rows) - len(filtered)


#
# Agrupa las filas según las columnas de agrupación del paso.
#
def _group_rows(step, filtered, budget):
    groups = {}

    for row in filtered:
        budget.check()
        key = [_grouping_value(row[group.field.field], group.time_grain) for group in step.group_by]
        groups.setdefault(tuple(key), []).append(row)

    if not step.group_by and not groups:
        groups[()] = []

    return groups


#
# Devuelve el valor de agrupación truncado al grano temporal indicado.
#
def _grouping_value(value, time_grain):
    if value is None or time_grain == "raw":
        return value

    if time_grain == "year":
        return value.year

    if time_grain == "month":
        return date(value.year, value.month, 1)

    return date(value.year, value.month, value.day)


#
# Devuelve las etiquetas de columnas de agrupación para una clave de grupo.
#
def _group_labels(step, key):
    return {group.output_name: value for group, value in zip(step.group_by, key, strict=True)}
