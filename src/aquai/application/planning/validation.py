from datetime import date, datetime
from uuid import UUID

from aquai.application.contracts.planning import (
    ColumnRef,
    DatasetSelection,
    OutputField,
    PlannableDecision,
    PlannerDecision,
    PlanningResult,
    SelectedDataset,
    ValidatedStep,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.errors import PlannerValidationError
from aquai.application.matching import contains_phrase
from aquai.application.units import presentation_unit_status

_EQUALITY_OPERATORS = {"eq", "ieq", "in"}
_RANGE_OPERATORS = {"lt", "le", "gt", "ge"}
_NUMERIC_TYPES = {"BIGINT", "NUMERIC"}
_TEMPORAL_TYPES = {"DATE", "TIMESTAMP"}
_ISO_PARSERS = {"DATE": date.fromisoformat, "TIMESTAMP": datetime.fromisoformat}
_COMBINE_OPERATIONS = {"join", "append"}
_PASSTHROUGH_OPERATIONS = {"sort", "latest"}
_AGGREGATE_FUNCTIONS = {
    "count",
    "count_distinct",
    "sum",
    "mean",
    "min",
    "max",
    "median",
    "stddev",
    "quantile",
}
_PYTHON_ONLY_FUNCTIONS = {"median", "stddev", "quantile"}
_COUNT_FUNCTIONS = {"count", "count_distinct"}
_EXTREME_FUNCTIONS = {"min", "max"}
_AUTHORIZED_METADATA = (
    "source_id",
    "semantic_point_id",
    "physical_table_name",
    "row_count",
    "columns_metadata",
    "provenance",
    "warnings",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlannerValidationError(message)


def _is_numeric(field: OutputField) -> bool:
    return field.numeric_eligible and field.sql_type in _NUMERIC_TYPES


def _equality_text_values(step) -> set[str]:
    values = set()

    for predicate in step.filters:
        if predicate.operator in _EQUALITY_OPERATORS:
            values.update(value for value in predicate.values if isinstance(value, str))

    return values


def _keeps_value(scope, value) -> bool:
    # Una resolución a valores almacenados («5Punta piedras») conserva la mención pedida.
    return any(
        item.casefold() == value.casefold() or contains_phrase(item, value) for item in scope
    )


def validate_temporal_scope(operations, output_steps, *, require_latest=False, requested_values=()):
    """El alcance temporal se revalida también antes de ejecutar y publicar."""
    steps = {step.id: step for step in operations}
    relevant = _reachable_step_ids(steps, output_steps)
    targets = [step for step in operations if step.id in relevant and step.operation == "latest"]
    issues = []

    if require_latest and not targets:
        issues.append(
            ValidationIssue(
                code="missing_latest_selection",
                severity="clarification",
                message="La consulta pide recencia, pero el plan no selecciona la fecha máxima conservando sus mediciones empatadas.",
            )
        )

    scopes = [_latest_scope(target, steps) for target in targets]
    issues.extend(_unscoped_latest_issues(targets, scopes, requested_values))

    if not requested_values or targets:
        return issues

    # Sin recencia: cada salida publicada debe acotarse a alguno de los valores pedidos y
    # cada valor pedido debe aparecer en alguna salida (p. ej., un sitio nuevo tras una aclaración).
    outputs = [(key, _output_scope(steps[key], steps)) for key in output_steps if key in steps]
    issues.extend(_unscoped_output_issues(outputs, requested_values))

    return issues


def _reachable_step_ids(steps, output_steps) -> set[str]:
    relevant, visit = set(), list(output_steps)

    while visit:
        key = visit.pop()
        if key in relevant or key not in steps:
            continue
        relevant.add(key)
        visit.extend(steps[key].depends_on)

    return relevant


def _latest_scope(target, steps) -> set[str]:
    values, visit = set(), [target]

    while visit:
        step = visit.pop()
        values.update(_equality_text_values(step))
        visit.extend(steps[key] for key in step.depends_on)

    return values


def _output_scope(output, steps) -> set[str]:
    values, visit, seen = set(), [output], set()

    while visit:
        step = visit.pop()
        if step.id in seen:
            continue
        seen.add(step.id)
        values.update(_equality_text_values(step))
        visit.extend(steps[parent] for parent in step.depends_on if parent in steps)

    return values


def _unscoped_latest_issues(targets, scopes, requested_values) -> list[ValidationIssue]:
    issues = []

    for target, scope in zip(targets, scopes):
        if requested_values and not any(_keeps_value(scope, value) for value in requested_values):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    step_id=target.id,
                    message="La selección temporal omite los valores de filtro solicitados por el usuario.",
                )
            )

    for value in requested_values:
        if targets and not any(_keeps_value(scope, value) for scope in scopes):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    message=f"La selección temporal no conserva el filtro solicitado para «{value}».",
                )
            )

    return issues


def _unscoped_output_issues(outputs, requested_values) -> list[ValidationIssue]:
    issues = []

    for key, scope in outputs:
        if not any(_keeps_value(scope, value) for value in requested_values):
            listed = ", ".join(f"«{value}»" for value in requested_values)
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    step_id=key,
                    message=f"La salida {key} no se acota a ninguno de los valores pedidos: {listed}.",
                )
            )

    for value in requested_values:
        if outputs and not any(_keeps_value(scope, value) for _, scope in outputs):
            issues.append(
                ValidationIssue(
                    code="missing_requested_filter",
                    severity="clarification",
                    message=f"Ninguna salida filtra por «{value}», que la consulta pide.",
                )
            )

    return issues


def validate_selections(
    decision: PlannerDecision, catalog: list[VerifiedDataset]
) -> list[SelectedDataset]:
    available = {dataset.dataset_id: dataset for dataset in catalog}
    result = []
    seen = set()

    for selection in decision.datasets:
        _require(selection.dataset_id not in seen, "El modelo repitió un dataset.")
        seen.add(selection.dataset_id)

        dataset = available.get(selection.dataset_id)
        _require(
            dataset is not None, "El modelo seleccionó un dataset fuera del contexto verificado."
        )

        result.append(_selected_dataset(selection, dataset))

    return result


def _selected_dataset(selection: DatasetSelection, dataset: VerifiedDataset) -> SelectedDataset:
    _require(
        selection.version == dataset.verified_version,
        "La versión propuesta no coincide con PostgreSQL.",
    )
    columns = {column.sql_name: column for column in dataset.columns_metadata}
    _require(len(set(selection.columns)) == len(selection.columns), "La selección repite columnas.")
    _require(
        set(selection.columns).issubset(columns),
        "El modelo seleccionó una columna inexistente.",
    )

    names = _with_projection_origins(selection.columns, columns)

    return SelectedDataset(
        **dataset.model_dump(exclude={"columns_metadata"}),
        columns_metadata=[columns[name].model_copy(deep=True) for name in names],
        selection_reason=selection.reason,
    )


def _with_projection_origins(selected: list[str], columns: dict) -> list[str]:
    names = list(selected)

    for name in selected:
        origin = columns[name].derived_from
        if not origin:
            continue
        _require(
            origin in columns, "La proyección numérica no tiene columna de origen verificable."
        )
        if origin not in names:
            names.append(origin)

    return names


def validate_operations(
    decision: PlannerDecision, selected: list[SelectedDataset], temporal_text=()
) -> tuple[list[ValidatedStep], list[ValidationIssue]]:
    return _OperationValidator(decision, selected, temporal_text).validate()


class _OperationValidator:
    def __init__(self, decision: PlannerDecision, selected: list[SelectedDataset], temporal_text):
        self._decision = decision
        self._datasets = {dataset.dataset_id: dataset for dataset in selected}
        self._periods = {(item.dataset_id, item.column) for item in temporal_text}
        self._requested = {item.dataset_id: set(item.columns) for item in decision.datasets}
        self._steps: dict[str, ValidatedStep] = {}
        self._outputs: dict[str, dict[str, OutputField]] = {}
        self._used_datasets: set[UUID] = set()
        self._limited_steps: set[str] = set()
        self._issues: list[ValidationIssue] = []

    def validate(self) -> tuple[list[ValidatedStep], list[ValidationIssue]]:
        for step in self._decision.steps:
            self._check_dependencies(step)

            if step.operation in _COMBINE_OPERATIONS:
                self._check_combination(step)
                return [], self._issues

            fields = self._step_fields(step)
            self._check_filters(step)
            self._check_ordering(step)
            self._check_limit(step)
            self._register(step, fields)

        if self._decision.status == "plannable":
            self._check_plan_outputs()

        return list(self._steps.values()), self._issues

    def _issue(self, code, message, step_id, severity="clarification") -> None:
        self._issues.append(
            ValidationIssue(code=code, message=message, step_id=step_id, severity=severity)
        )

    def _check_dependencies(self, step) -> None:
        _require(step.id not in self._steps, "Los identificadores de pasos deben ser únicos.")
        _require(len(set(step.depends_on)) == len(step.depends_on), "Un paso repite dependencias.")
        _require(
            all(dependency in self._steps for dependency in step.depends_on),
            f"El plan contiene un ciclo o dependencia futura/inexistente: el paso {step.id} depende de {step.depends_on}.",
        )

    def _check_combination(self, step) -> None:
        _require(len(step.depends_on) >= 2, "Combinar requiere al menos dos resultados previos.")

        for key in step.join_on:
            self._resolve(step, key.left)
            self._resolve(step, key.right)

        self._issue(
            "unverified_observation_compatibility",
            "La combinación requiere confirmar claves, granularidad, cardinalidad y solapamientos; el catálogo actual no los garantiza.",
            step.id,
        )

    def _raw_field(self, ref: ColumnRef) -> OutputField:
        _require(
            ref.dataset_id in self._requested and ref.column in self._requested[ref.dataset_id],
            "Una operación usa una columna no seleccionada.",
        )
        column = next(
            column
            for column in self._datasets[ref.dataset_id].columns_metadata
            if column.sql_name == ref.column
        )

        return OutputField(
            name=column.sql_name,
            sql_type=column.sql_type,
            numeric_eligible=column.numeric_eligible is True,
            unit=column.unit,
            unit_status="documented" if column.unit is not None else "unknown",
            sources=[ref],
        )

    def _resolve(self, step, ref) -> OutputField:
        if isinstance(ref, ColumnRef):
            _require(
                step.operation == "read" and ref in step.columns,
                f"Sólo read puede usar referencias directas, declaradas en columns: el paso {step.id} usa la columna {ref.column}.",
            )
            return self._raw_field(ref)

        _require(
            ref.step_id in step.depends_on,
            f"La referencia debe provenir de una dependencia directa: el paso {step.id} usa "
            f"{ref.step_id}.{ref.field}, pero depende de {step.depends_on}.",
        )
        _require(
            ref.field in self._outputs[ref.step_id],
            f"La operación refiere a un resultado inexistente: {ref.step_id} no expone «{ref.field}»; "
            f"expone {sorted(self._outputs[ref.step_id])}.",
        )

        return self._outputs[ref.step_id][ref.field]

    def _step_fields(self, step) -> list[OutputField]:
        _require(not step.join_on, "Las claves de unión sólo corresponden a join.")
        _require(
            (step.operation == "visualize") == (step.visualization is not None),
            "visualize requiere su especificación exclusiva de gráfico.",
        )

        if step.operation == "read":
            return self._read_fields(step)

        return self._dependent_fields(step)

    def _read_fields(self, step) -> list[OutputField]:
        _require(
            step.tool == "sql" and not step.depends_on and bool(step.columns),
            "read requiere SQL, columnas y ninguna dependencia.",
        )
        _require(
            not step.group_by and not step.calculations,
            "read no puede ocultar agrupaciones ni cálculos.",
        )

        ids = {ref.dataset_id for ref in step.columns}
        _require(
            len(ids) == 1,
            f"Cada read debe leer un único dataset: el paso {step.id} mezcla columnas de {len(ids)} datasets. "
            "Usá un read por dataset y una rama independiente para cada uno (read -> sort, aggregate o latest "
            "propios) y poné el último paso de cada rama en output_steps.",
        )
        self._used_datasets.update(ids)

        return [self._raw_field(ref) for ref in step.columns]

    def _dependent_fields(self, step) -> list[OutputField]:
        source = self._single_source(step)
        inherited = list(self._outputs[source].values())

        if step.operation not in _PASSTHROUGH_OPERATIONS:
            _require(
                not step.order_by and step.limit is None,
                "Orden y límite requieren un paso sort explícito después del cálculo.",
            )

        if step.operation == "visualize":
            self._check_visualization(step, source)
            return []

        if step.operation == "sort":
            self._check_sort(step)
            return inherited

        if step.operation == "latest":
            self._check_latest(step, source)
            return inherited

        return self._calculated_fields(step, source)

    def _single_source(self, step) -> str:
        _require(
            len(step.depends_on) == 1 and not step.columns,
            "La operación requiere un resultado previo; no columnas directas.",
        )
        source = step.depends_on[0]
        _require(
            self._steps[source].operation != "visualize",
            "Un gráfico no es una tabla de entrada para cálculos.",
        )

        if step.tool == "sql" and self._steps[source].tool == "python":
            self._issue(
                "materialization_required",
                "El paso SQL queda pendiente: consumir resultados Python requiere una estrategia de materialización todavía no implementada.",
                step.id,
                "warning",
            )

        return source

    def _check_visualization(self, step, source: str) -> None:
        _require(
            step.tool == "python"
            and not step.group_by
            and not step.calculations
            and not step.filters,
            "visualize requiere Python y una entrada explícita sin cálculos ni filtros ocultos.",
        )

        chart = step.visualization
        x = self._resolve(step, chart.x)
        y = self._resolve(step, chart.y) if chart.y is not None else None

        if chart.series is not None:
            self._resolve(step, chart.series)
            _require(
                chart.series != chart.x,
                "La serie del gráfico debe ser un campo distinto del eje x.",
            )

        if chart.chart_type in {"scatter", "histogram"}:
            _require(
                _is_numeric(x), "El eje x del gráfico requiere una variable numérica elegible."
            )

        if chart.chart_type == "line":
            _require(
                _is_numeric(x) or x.sql_type in _TEMPORAL_TYPES,
                "Una línea requiere un eje numérico o temporal verificado; no se interpretan fechas TEXT.",
            )

        if y is not None:
            _require(
                _is_numeric(y), "El eje y del gráfico requiere una variable numérica elegible."
            )
            _require(
                chart.x != chart.y,
                "Los ejes x e y del gráfico deben ser campos distintos: x es la categoría "
                "(por ejemplo, el sitio agrupado) e y el valor calculado.",
            )

        if source in self._limited_steps:
            self._issue(
                "explicitly_limited_chart",
                "El gráfico representa el límite explícito del plan, no todo el dataset.",
                step.id,
                "warning",
            )

    def _check_sort(self, step) -> None:
        _require(
            not step.group_by and not step.calculations and bool(step.order_by),
            "sort sólo debe ordenar/filtrar/limitar un resultado previo.",
        )

    def _check_latest(self, step, source: str) -> None:
        _require(
            not step.group_by
            and not step.calculations
            and step.limit is None
            and len(step.order_by) == 1
            and step.order_by[0].direction == "desc",
            "latest requiere una fecha descendente, sin agregaciones ni límites que corten empates.",
        )

        temporal = self._resolve(step, step.order_by[0].field)
        _require(
            temporal.sql_type in _TEMPORAL_TYPES
            or (self._is_verified_period(temporal) and step.tool == "python"),
            "latest requiere una fecha real DATE/TIMESTAMP o períodos TEXT verificados; no ordena textos libres ni números de fila.",
        )

        if source in self._limited_steps:
            self._issue(
                "limited_latest_input",
                "La fecha más reciente requiere la entrada completa; no se calcula sobre una lectura limitada.",
                step.id,
            )

    def _is_verified_period(self, field: OutputField) -> bool:
        # Un TEXT sólo ordena si su columna fue perfilada como períodos explícitos;
        # Python interpreta cada valor con las mismas reglas y descarta los ambiguos.
        return (
            field.sql_type == "TEXT"
            and len(field.sources) == 1
            and (field.sources[0].dataset_id, field.sources[0].column) in self._periods
        )

    def _calculated_fields(self, step, source: str) -> list[OutputField]:
        _require(
            bool(step.calculations),
            f"El paso {step.id} ({step.operation}) debe declarar cálculos.",
        )
        self._check_calculation_input(step, source)

        if step.operation == "calculate":
            self._check_scalar_calculation(step)

        fields = [self._grouping_field(step, grouping) for grouping in step.group_by]
        for calculation in step.calculations:
            fields.append(self._calculation_field(step, calculation))

        return fields

    def _check_calculation_input(self, step, source: str) -> None:
        reaggregates_single_row = (
            step.operation == "aggregate"
            and self._steps[source].operation == "aggregate"
            and not self._steps[source].group_by
        )
        if reaggregates_single_row:
            # Agregar una única fila ya agregada (contarla, promediarla) no responde nada.
            self._issue(
                "degenerate_aggregate",
                f"El paso {step.id} vuelve a agregar el resultado de {source}, que ya es "
                f"una única fila sin agrupación; publicá {source} en output_steps en lugar de agregarlo otra vez.",
                step.id,
            )

        if source in self._limited_steps:
            self._issue(
                "limited_calculation_input",
                "El cálculo depende de una lectura limitada; hay que precisar si se pretende calcular sobre una muestra.",
                step.id,
            )

    def _check_scalar_calculation(self, step) -> None:
        _require(not step.group_by, "calculate no admite agrupaciones implícitas.")

        functions = {calculation.function for calculation in step.calculations}
        _require(
            not ("correlation" in functions and len(functions) > 1),
            "No mezclar una correlación escalar con cálculos por registro.",
        )

    def _grouping_field(self, step, grouping) -> OutputField:
        _require(step.operation == "aggregate", "Las agrupaciones pertenecen a aggregate.")
        field = self._resolve(step, grouping.field).model_copy(deep=True)

        if grouping.time_grain != "raw":
            if field.sql_type not in _TEMPORAL_TYPES:
                self._issue(
                    "ambiguous_date",
                    "No se puede agrupar por calendario una columna TEXT o de fecha parcial sin una interpretación confirmada.",
                    step.id,
                )
            field.sql_type = "BIGINT" if grouping.time_grain == "year" else "DATE"
            field.numeric_eligible = False
            field.unit = None
            field.unit_status = "dimensionless"

        field.name = grouping.output_name

        return field

    def _calculation_field(self, step, calculation) -> OutputField:
        function = calculation.function
        _require(
            (function in _AGGREGATE_FUNCTIONS) == (step.operation == "aggregate"),
            "La función no corresponde al tipo de operación.",
        )
        if function in _PYTHON_ONLY_FUNCTIONS:
            _require(
                step.tool == "python",
                "La nueva función estadística está habilitada exclusivamente en Python.",
            )

        operands = [self._resolve(step, ref) for ref in calculation.inputs]
        refs = [ref for operand in operands for ref in operand.sources]
        sql_type, unit, unit_status = self._calculation_result(step, function, operands)

        return OutputField(
            name=calculation.output_name,
            sql_type=sql_type,
            numeric_eligible=sql_type in _NUMERIC_TYPES,
            unit=unit,
            unit_status=unit_status,
            sources=list({(ref.dataset_id, ref.column): ref for ref in refs}.values()),
        )

    def _calculation_result(self, step, function: str, operands: list[OutputField]):
        if function in _COUNT_FUNCTIONS:
            return "BIGINT", None, "dimensionless"

        self._check_operands_eligible(step, function, operands)
        sql_type = operands[0].sql_type if function in _EXTREME_FUNCTIONS else "NUMERIC"
        unit, unit_status = operands[0].unit, operands[0].unit_status

        if function in {"difference", "ratio"}:
            self._check_comparable_units(step, operands)

        if function in {"ratio", "correlation"}:
            return sql_type, None, "dimensionless"

        if presentation_unit_status(operands[0]) == "unknown" and sql_type in _NUMERIC_TYPES:
            self._issue(
                "undocumented_unit",
                "El resultado conserva una unidad desconocida; no se infiere a partir del nombre.",
                step.id,
                "warning",
            )

        return sql_type, unit, unit_status

    def _check_operands_eligible(self, step, function: str, operands: list[OutputField]) -> None:
        for operand in operands:
            temporal_extreme = (
                function in _EXTREME_FUNCTIONS and operand.sql_type in _TEMPORAL_TYPES
            )
            if temporal_extreme or _is_numeric(operand):
                continue
            self._issue(
                "ineligible_calculation",
                "El cálculo requiere una columna numérica elegible o una proyección numérica documentada; no se convierte TEXT automáticamente.",
                step.id,
            )

    def _check_comparable_units(self, step, operands: list[OutputField]) -> None:
        compatible = operands[0].unit is not None and operands[0].unit == operands[1].unit
        same_variable = operands[0].sources == operands[1].sources

        if not compatible and not same_variable:
            self._issue(
                "unverified_units",
                "La operación necesita unidades comparables documentadas; no se infieren equivalencias ni conversiones.",
                step.id,
            )

    def _check_filters(self, step) -> None:
        self._check_contradictory_filters(step)

        for predicate in step.filters:
            field = self._resolve(step, predicate.field)
            _require(
                predicate.operator != "ieq" or field.sql_type == "TEXT",
                "ieq sólo admite igualdad literal de columnas TEXT sin distinguir mayúsculas.",
            )

            if predicate.operator in _RANGE_OPERATORS and field.sql_type == "TEXT":
                self._issue(
                    "text_range",
                    "Un rango sobre TEXT necesita confirmar si el orden es textual, numérico o temporal.",
                    step.id,
                )

            for value in predicate.values:
                _check_filter_value(field, value)

    def _check_contradictory_filters(self, step) -> None:
        equalities = {}

        for predicate in step.filters:
            if predicate.operator not in _EQUALITY_OPERATORS:
                continue

            key = predicate.field.model_dump_json()
            values = {str(value).casefold() for value in predicate.values}
            if key in equalities and not equalities[key] & values:
                self._issue(
                    "contradictory_filters",
                    f"El paso {step.id} exige valores incompatibles en el mismo campo "
                    f"({sorted(equalities[key])} y {sorted(values)}); ninguna fila puede cumplir ambos. "
                    "Para varios valores usá in o ramas separadas.",
                    step.id,
                )

            equalities[key] = equalities.get(key, values) & values or values

    def _check_ordering(self, step) -> None:
        for ordering in step.order_by:
            field = self._resolve(step, ordering.field)
            _require(
                not (step.limit == 1 and field.sql_type in _TEMPORAL_TYPES),
                "Un límite de una fila sobre fechas puede elegir arbitrariamente un empate; usá latest para conservar todas las mediciones.",
            )

    def _check_limit(self, step) -> None:
        if step.limit is None:
            return

        ancestors = list(step.depends_on)
        while ancestors:
            ancestor = self._steps[ancestors.pop()]
            _require(
                ancestor.operation != "latest",
                "No se permite recortar con un límite las mediciones empatadas de latest.",
            )
            ancestors.extend(ancestor.depends_on)

    def _register(self, step, fields: list[OutputField]) -> None:
        _require(
            len({field.name for field in fields}) == len(fields),
            "Las columnas de salida de un paso deben tener nombres únicos.",
        )

        self._steps[step.id] = ValidatedStep(**step.model_dump(), output_fields=fields)
        self._outputs[step.id] = {field.name: field for field in fields}

        inherits_limit = any(dependency in self._limited_steps for dependency in step.depends_on)
        if step.limit is not None or inherits_limit:
            self._limited_steps.add(step.id)

    def _check_plan_outputs(self) -> None:
        output_steps = self._decision.output_steps
        _require(
            self._used_datasets == set(self._datasets),
            "Todo dataset seleccionado debe participar en el plan.",
        )
        _require(
            len(set(output_steps)) == len(output_steps),
            "Las salidas finales no deben repetirse.",
        )
        _require(
            all(step_id in self._steps for step_id in output_steps),
            "Una salida final refiere a un paso inexistente.",
        )


def _check_filter_value(field: OutputField, value) -> None:
    if field.sql_type in _NUMERIC_TYPES:
        _require(type(value) in {int, float}, "Un filtro numérico requiere constantes numéricas.")
        return

    if field.sql_type == "BOOLEAN":
        _require(type(value) is bool, "Un filtro booleano requiere valores booleanos.")
        return

    _require(isinstance(value, str), "Un filtro de texto/fecha requiere constantes textuales.")

    parse = _ISO_PARSERS.get(field.sql_type)
    if parse is None:
        return

    try:
        parse(value)
    except ValueError as error:
        raise PlannerValidationError("El filtro temporal no usa una fecha ISO completa.") from error


def validate_execution_plan(plan: PlanningResult, catalog: list[VerifiedDataset]):
    """Reutiliza las verificaciones del Planner con un catálogo fresco y confiable."""
    _require_execution_authorized(plan)

    decision = _revalidation_decision(plan)
    snapshots = [VerifiedDataset.model_validate(item.model_dump()) for item in catalog]
    selected = validate_selections(decision, snapshots)
    _require_unchanged_metadata(plan.selected_datasets, selected)

    operations, issues = validate_operations(decision, selected, plan.temporal_text_columns)
    evidence = plan.context.evidence
    issues.extend(
        validate_temporal_scope(
            operations,
            plan.output_steps,
            require_latest=_latest_required(evidence),
            requested_values=_requested_values(evidence),
        )
    )
    _require(
        not any(issue.severity == "clarification" for issue in issues),
        "La operación requiere resolver una aclaración antes de ejecutarse.",
    )

    schema_changed = any(
        old.output_fields != new.output_fields
        for old, new in zip(plan.operations, operations, strict=True)
    )
    _require(not schema_changed, "El esquema del plan no coincide con el derivado del catálogo.")

    return selected, operations


def _require_execution_authorized(plan: PlanningResult) -> None:
    authorized = plan.status == "plannable" and not any(
        issue.severity == "clarification" for issue in plan.validation_issues
    )
    _require(authorized, "El plan no está autorizado para ejecución.")


def _revalidation_decision(plan: PlanningResult) -> PlannableDecision:
    return PlannableDecision(
        status="plannable",
        reason="Revalidación previa a ejecución.",
        datasets=[_dataset_selection(item) for item in plan.selected_datasets],
        steps=[step.model_dump(exclude={"output_fields"}) for step in plan.operations],
        output_steps=plan.output_steps,
    )


def _dataset_selection(dataset: SelectedDataset) -> DatasetSelection:
    return DatasetSelection(
        dataset_id=dataset.dataset_id,
        version=dataset.verified_version,
        columns=[column.sql_name for column in dataset.columns_metadata],
        reason=dataset.selection_reason,
    )


def _require_unchanged_metadata(proposed_datasets, actual_datasets) -> None:
    for proposed, actual in zip(proposed_datasets, actual_datasets, strict=True):
        changed = any(
            getattr(proposed, field) != getattr(actual, field) for field in _AUTHORIZED_METADATA
        )
        _require(not changed, "Los metadatos autorizados cambiaron; hay que volver a planificar.")


def _latest_required(evidence) -> bool:
    return evidence.get("temporal_requirements", {}).get("operation") == "latest"


def _requested_values(evidence) -> list[str]:
    return [item["text"] for item in evidence.get("query_mentions", []) if item["kind"] == "value"]
