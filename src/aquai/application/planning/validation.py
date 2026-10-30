"""Verificación del plan contra metadatos; no ejecuta sus operaciones."""

from datetime import date, datetime
from uuid import UUID

from aquai.application.contracts.planning import (
    ColumnRef, OutputField, PlannerDecision, SelectedDataset,
    ValidatedStep, ValidationIssue,
)
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.errors import PlannerValidationError
from aquai.application.matching import contains_phrase
from aquai.application.units import presentation_unit_status


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlannerValidationError(message)


def validate_temporal_scope(operations, output_steps, *, require_latest=False, requested_values=()):
    """El alcance temporal se revalida también antes de ejecutar y publicar."""
    steps = {step.id: step for step in operations}
    relevant, visit = set(), list(output_steps)
    while visit:
        key = visit.pop()
        if key in relevant or key not in steps:
            continue
        relevant.add(key)
        visit.extend(steps[key].depends_on)
    targets = [step for step in operations if step.id in relevant and step.operation == "latest"]
    issues, scopes = [], []
    if require_latest and not targets:
        issues.append(ValidationIssue(code="missing_latest_selection", severity="clarification",
            message="La consulta pide recencia, pero el plan no selecciona la fecha máxima conservando sus mediciones empatadas."))
    for target in targets:
        values, visit = set(), [target]
        while visit:
            step = visit.pop()
            for predicate in step.filters:
                if predicate.operator in {"eq", "ieq", "in"}:
                    values.update(value for value in predicate.values if isinstance(value, str))
            visit.extend(steps[key] for key in step.depends_on)
        scopes.append(values)

    def kept(value, scope):
        # Una resolución a valores almacenados («5Punta piedras») conserva la mención pedida.
        return any(item.casefold() == value.casefold() or contains_phrase(item, value) for item in scope)

    for target, scope in zip(targets, scopes):
        if requested_values and not any(kept(value, scope) for value in requested_values):
            issues.append(ValidationIssue(code="missing_requested_filter", severity="clarification", step_id=target.id,
                message="La selección temporal omite los valores de filtro solicitados por el usuario."))
    for value in requested_values:
        if targets and not any(kept(value, scope) for scope in scopes):
            issues.append(ValidationIssue(code="missing_requested_filter", severity="clarification",
                message=f"La selección temporal no conserva el filtro solicitado para «{value}»."))
    if not requested_values or targets:
        return issues
    # Sin recencia: cada salida publicada debe acotarse a alguno de los valores pedidos y
    # cada valor pedido debe aparecer en alguna salida (p. ej., un sitio nuevo tras una aclaración).
    outputs = []
    for key in output_steps:
        if key not in steps:
            continue
        values, visit, seen = set(), [steps[key]], set()
        while visit:
            step = visit.pop()
            if step.id in seen:
                continue
            seen.add(step.id)
            for predicate in step.filters:
                if predicate.operator in {"eq", "ieq", "in"}:
                    values.update(value for value in predicate.values if isinstance(value, str))
            visit.extend(steps[parent] for parent in step.depends_on if parent in steps)
        outputs.append((key, values))
    for key, scope in outputs:
        if not any(kept(value, scope) for value in requested_values):
            issues.append(ValidationIssue(code="missing_requested_filter", severity="clarification", step_id=key,
                message=f"La salida {key} no se acota a ninguno de los valores pedidos: "
                        + ", ".join(f"«{value}»" for value in requested_values) + "."))
    for value in requested_values:
        if outputs and not any(kept(value, scope) for _, scope in outputs):
            issues.append(ValidationIssue(code="missing_requested_filter", severity="clarification",
                message=f"Ninguna salida filtra por «{value}», que la consulta pide."))
    return issues


def validate_selections(decision: PlannerDecision, catalog: list[VerifiedDataset]) -> list[SelectedDataset]:
    available = {dataset.dataset_id: dataset for dataset in catalog}
    result = []
    seen = set()
    for selection in decision.datasets:
        _require(selection.dataset_id not in seen, "El modelo repitió un dataset.")
        seen.add(selection.dataset_id)
        dataset = available.get(selection.dataset_id)
        _require(dataset is not None, "El modelo seleccionó un dataset fuera del contexto verificado.")
        _require(selection.version == dataset.verified_version, "La versión propuesta no coincide con PostgreSQL.")
        columns = {column.sql_name: column for column in dataset.columns_metadata}
        _require(len(set(selection.columns)) == len(selection.columns), "La selección repite columnas.")
        _require(set(selection.columns).issubset(columns), "El modelo seleccionó una columna inexistente.")
        names = list(selection.columns)
        for name in selection.columns:
            origin = columns[name].derived_from
            if origin:
                _require(origin in columns, "La proyección numérica no tiene columna de origen verificable.")
                if origin not in names:
                    names.append(origin)
        result.append(SelectedDataset(
            **dataset.model_dump(exclude={"columns_metadata"}),
            columns_metadata=[columns[name].model_copy(deep=True) for name in names],
            selection_reason=selection.reason,
        ))
    return result


def validate_operations(decision: PlannerDecision, selected: list[SelectedDataset],
                        temporal_text=()) -> tuple[list[ValidatedStep], list[ValidationIssue]]:
    datasets = {dataset.dataset_id: dataset for dataset in selected}
    periods = {(item.dataset_id, item.column) for item in temporal_text}
    requested = {item.dataset_id: set(item.columns) for item in decision.datasets}
    steps = {}
    outputs: dict[str, dict[str, OutputField]] = {}
    used_datasets: set[UUID] = set()
    limited_steps = set()
    issues = []

    def issue(code, message, step_id, severity="clarification"):
        issues.append(ValidationIssue(code=code, message=message, step_id=step_id, severity=severity))

    def raw_field(ref: ColumnRef) -> OutputField:
        _require(ref.dataset_id in requested and ref.column in requested[ref.dataset_id], "Una operación usa una columna no seleccionada.")
        column = next(column for column in datasets[ref.dataset_id].columns_metadata if column.sql_name == ref.column)
        return OutputField(
            name=column.sql_name, sql_type=column.sql_type, numeric_eligible=column.numeric_eligible is True,
            unit=column.unit, unit_status="documented" if column.unit is not None else "unknown", sources=[ref],
        )

    for step in decision.steps:
        _require(step.id not in steps, "Los identificadores de pasos deben ser únicos.")
        _require(len(set(step.depends_on)) == len(step.depends_on), "Un paso repite dependencias.")
        _require(all(dependency in steps for dependency in step.depends_on),
                 f"El plan contiene un ciclo o dependencia futura/inexistente: el paso {step.id} depende de {step.depends_on}.")

        def resolve(ref):
            if isinstance(ref, ColumnRef):
                _require(step.operation == "read" and ref in step.columns,
                 f"Sólo read puede usar referencias directas, declaradas en columns: el paso {step.id} usa la columna {ref.column}.")
                return raw_field(ref)
            _require(ref.step_id in step.depends_on,
                     f"La referencia debe provenir de una dependencia directa: el paso {step.id} usa "
                     f"{ref.step_id}.{ref.field}, pero depende de {step.depends_on}.")
            _require(ref.field in outputs[ref.step_id],
                     f"La operación refiere a un resultado inexistente: {ref.step_id} no expone «{ref.field}»; "
                     f"expone {sorted(outputs[ref.step_id])}.")
            return outputs[ref.step_id][ref.field]

        if step.operation in {"join", "append"}:
            _require(len(step.depends_on) >= 2, "Combinar requiere al menos dos resultados previos.")
            for key in step.join_on:
                resolve(key.left)
                resolve(key.right)
            issue("unverified_observation_compatibility", "La combinación requiere confirmar claves, granularidad, cardinalidad y solapamientos; el catálogo actual no los garantiza.", step.id)
            return [], issues

        _require(not step.join_on, "Las claves de unión sólo corresponden a join.")
        _require((step.operation == "visualize") == (step.visualization is not None), "visualize requiere su especificación exclusiva de gráfico.")
        if step.operation == "read":
            _require(step.tool == "sql" and not step.depends_on and bool(step.columns), "read requiere SQL, columnas y ninguna dependencia.")
            _require(not step.group_by and not step.calculations, "read no puede ocultar agrupaciones ni cálculos.")
            ids = {ref.dataset_id for ref in step.columns}
            _require(len(ids) == 1,
                     f"Cada read debe leer un único dataset: el paso {step.id} mezcla columnas de {len(ids)} datasets. "
                     "Usá un read por dataset y una rama independiente para cada uno (read -> sort, aggregate o latest "
                     "propios) y poné el último paso de cada rama en output_steps.")
            used_datasets.update(ids)
            fields = [raw_field(ref) for ref in step.columns]
        else:
            _require(len(step.depends_on) == 1 and not step.columns, "La operación requiere un resultado previo; no columnas directas.")
            source = step.depends_on[0]
            _require(steps[source].operation != "visualize", "Un gráfico no es una tabla de entrada para cálculos.")
            if step.tool == "sql" and steps[source].tool == "python":
                issue("materialization_required", "El paso SQL queda pendiente: consumir resultados Python requiere una estrategia de materialización todavía no implementada.", step.id, "warning")
            fields = list(outputs[source].values())
            if step.operation not in {"sort", "latest"}:
                _require(not step.order_by and step.limit is None, "Orden y límite requieren un paso sort explícito después del cálculo.")
            if step.operation == "visualize":
                _require(step.tool == "python" and not step.group_by and not step.calculations and not step.filters,
                         "visualize requiere Python y una entrada explícita sin cálculos ni filtros ocultos.")
                chart = step.visualization
                x = resolve(chart.x)
                y = resolve(chart.y) if chart.y is not None else None
                if chart.series is not None:
                    resolve(chart.series)
                numeric = lambda field: field.numeric_eligible and field.sql_type in {"BIGINT", "NUMERIC"}
                if chart.chart_type in {"scatter", "histogram"}:
                    _require(numeric(x), "El eje x del gráfico requiere una variable numérica elegible.")
                if chart.chart_type == "line":
                    _require(numeric(x) or x.sql_type in {"DATE", "TIMESTAMP"}, "Una línea requiere un eje numérico o temporal verificado; no se interpretan fechas TEXT.")
                if y is not None:
                    _require(numeric(y), "El eje y del gráfico requiere una variable numérica elegible.")
                if source in limited_steps:
                    issue("explicitly_limited_chart", "El gráfico representa el límite explícito del plan, no todo el dataset.", step.id, "warning")
                fields = []
            elif step.operation == "sort":
                _require(not step.group_by and not step.calculations and bool(step.order_by), "sort sólo debe ordenar/filtrar/limitar un resultado previo.")
            elif step.operation == "latest":
                _require(not step.group_by and not step.calculations and step.limit is None
                         and len(step.order_by) == 1 and step.order_by[0].direction == "desc",
                         "latest requiere una fecha descendente, sin agregaciones ni límites que corten empates.")
                temporal = resolve(step.order_by[0].field)
                # Un TEXT sólo ordena si su columna fue perfilada como períodos explícitos;
                # Python interpreta cada valor con las mismas reglas y descarta los ambiguos.
                period = (temporal.sql_type == "TEXT" and len(temporal.sources) == 1
                          and (temporal.sources[0].dataset_id, temporal.sources[0].column) in periods)
                _require(temporal.sql_type in {"DATE", "TIMESTAMP"} or (period and step.tool == "python"),
                         "latest requiere una fecha real DATE/TIMESTAMP o períodos TEXT verificados; no ordena textos libres ni números de fila.")
                if source in limited_steps:
                    issue("limited_latest_input", "La fecha más reciente requiere la entrada completa; no se calcula sobre una lectura limitada.", step.id)
            else:
                _require(bool(step.calculations), f"El paso {step.id} ({step.operation}) debe declarar cálculos.")
                if (step.operation == "aggregate" and steps[source].operation == "aggregate"
                        and not steps[source].group_by):
                    # Agregar una única fila ya agregada (contarla, promediarla) no responde nada.
                    issue("degenerate_aggregate", f"El paso {step.id} vuelve a agregar el resultado de {source}, que ya es "
                          f"una única fila sin agrupación; publicá {source} en output_steps en lugar de agregarlo otra vez.",
                          step.id)
                if source in limited_steps:
                    issue("limited_calculation_input", "El cálculo depende de una lectura limitada; hay que precisar si se pretende calcular sobre una muestra.", step.id)
                fields = []
                if step.operation == "calculate":
                    _require(not step.group_by, "calculate no admite agrupaciones implícitas.")
                    functions = {calculation.function for calculation in step.calculations}
                    _require(not ("correlation" in functions and len(functions) > 1), "No mezclar una correlación escalar con cálculos por registro.")
                for grouping in step.group_by:
                    _require(step.operation == "aggregate", "Las agrupaciones pertenecen a aggregate.")
                    field = resolve(grouping.field).model_copy(deep=True)
                    if grouping.time_grain != "raw":
                        if field.sql_type not in {"DATE", "TIMESTAMP"}:
                            issue("ambiguous_date", "No se puede agrupar por calendario una columna TEXT o de fecha parcial sin una interpretación confirmada.", step.id)
                        field.sql_type = "BIGINT" if grouping.time_grain == "year" else "DATE"
                        field.numeric_eligible = False
                        field.unit, field.unit_status = None, "dimensionless"
                    field.name = grouping.output_name
                    fields.append(field)
                for calculation in step.calculations:
                    function = calculation.function
                    aggregate = function in {"count", "count_distinct", "sum", "mean", "min", "max", "median", "stddev", "quantile"}
                    _require(aggregate == (step.operation == "aggregate"), "La función no corresponde al tipo de operación.")
                    if function in {"median", "stddev", "quantile"}:
                        _require(step.tool == "python", "La nueva función estadística está habilitada exclusivamente en Python.")
                    operands = [resolve(ref) for ref in calculation.inputs]
                    refs = [ref for operand in operands for ref in operand.sources]
                    unit, unit_status, sql_type = None, "dimensionless", "BIGINT"
                    if function not in {"count", "count_distinct"}:
                        for operand in operands:
                            temporal_extreme = function in {"min", "max"} and operand.sql_type in {"DATE", "TIMESTAMP"}
                            if not temporal_extreme and not (operand.numeric_eligible and operand.sql_type in {"BIGINT", "NUMERIC"}):
                                issue("ineligible_calculation", "El cálculo requiere una columna numérica elegible o una proyección numérica documentada; no se convierte TEXT automáticamente.", step.id)
                        sql_type = operands[0].sql_type if function in {"min", "max"} else "NUMERIC"
                        unit, unit_status = operands[0].unit, operands[0].unit_status
                        if function in {"difference", "ratio"}:
                            compatible = operands[0].unit is not None and operands[0].unit == operands[1].unit
                            same_variable = operands[0].sources == operands[1].sources
                            if not compatible and not same_variable:
                                issue("unverified_units", "La operación necesita unidades comparables documentadas; no se infieren equivalencias ni conversiones.", step.id)
                        if function in {"ratio", "correlation"}:
                            unit, unit_status = None, "dimensionless"
                        elif presentation_unit_status(operands[0]) == "unknown" and sql_type in {"BIGINT", "NUMERIC"}:
                            issue("undocumented_unit", "El resultado conserva una unidad desconocida; no se infiere a partir del nombre.", step.id, "warning")
                    fields.append(OutputField(
                        name=calculation.output_name, sql_type=sql_type,
                        numeric_eligible=sql_type in {"BIGINT", "NUMERIC"}, unit=unit,
                        unit_status=unit_status, sources=list({(ref.dataset_id, ref.column): ref for ref in refs}.values()),
                    ))

        equalities = {}
        for predicate in step.filters:
            if predicate.operator in {"eq", "ieq", "in"}:
                key = predicate.field.model_dump_json()
                values = {str(value).casefold() for value in predicate.values}
                if key in equalities and not equalities[key] & values:
                    issue("contradictory_filters", f"El paso {step.id} exige valores incompatibles en el mismo campo "
                          f"({sorted(equalities[key])} y {sorted(values)}); ninguna fila puede cumplir ambos. "
                          "Para varios valores usá in o ramas separadas.", step.id)
                equalities[key] = equalities.get(key, values) & values or values
        for predicate in step.filters:
            field = resolve(predicate.field)
            _require(predicate.operator != "ieq" or field.sql_type == "TEXT",
                     "ieq sólo admite igualdad literal de columnas TEXT sin distinguir mayúsculas.")
            if predicate.operator in {"lt", "le", "gt", "ge"} and field.sql_type == "TEXT":
                issue("text_range", "Un rango sobre TEXT necesita confirmar si el orden es textual, numérico o temporal.", step.id)
            for value in predicate.values:
                if field.sql_type in {"BIGINT", "NUMERIC"}:
                    _require(type(value) in {int, float}, "Un filtro numérico requiere constantes numéricas.")
                elif field.sql_type == "BOOLEAN":
                    _require(type(value) is bool, "Un filtro booleano requiere valores booleanos.")
                else:
                    _require(isinstance(value, str), "Un filtro de texto/fecha requiere constantes textuales.")
                    if field.sql_type in {"DATE", "TIMESTAMP"}:
                        try:
                            (date.fromisoformat if field.sql_type == "DATE" else datetime.fromisoformat)(value)
                        except ValueError as error:
                            raise PlannerValidationError("El filtro temporal no usa una fecha ISO completa.") from error
        for ordering in step.order_by:
            field = resolve(ordering.field)
            _require(not (step.limit == 1 and field.sql_type in {"DATE", "TIMESTAMP"}),
                     "Un límite de una fila sobre fechas puede elegir arbitrariamente un empate; usá latest para conservar todas las mediciones.")
        if step.limit is not None:
            ancestors = list(step.depends_on)
            while ancestors:
                ancestor = steps[ancestors.pop()]
                _require(ancestor.operation != "latest", "No se permite recortar con un límite las mediciones empatadas de latest.")
                ancestors.extend(ancestor.depends_on)
        _require(len({field.name for field in fields}) == len(fields), "Las columnas de salida de un paso deben tener nombres únicos.")
        validated = ValidatedStep(**step.model_dump(), output_fields=fields)
        steps[step.id] = validated
        outputs[step.id] = {field.name: field for field in fields}
        if step.limit is not None or any(dependency in limited_steps for dependency in step.depends_on):
            limited_steps.add(step.id)

    if decision.status == "plannable":
        _require(used_datasets == set(datasets), "Todo dataset seleccionado debe participar en el plan.")
        _require(len(set(decision.output_steps)) == len(decision.output_steps), "Las salidas finales no deben repetirse.")
        _require(all(step_id in steps for step_id in decision.output_steps), "Una salida final refiere a un paso inexistente.")
    return list(steps.values()), issues
