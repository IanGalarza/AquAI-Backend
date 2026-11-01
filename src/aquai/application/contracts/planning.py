from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aquai.application.contracts.retrieval import (
    DatasetRetrievalResult,
    VerifiedDataset,
)

PlanningStatus = Literal["plannable", "needs_clarification", "insufficient_coverage"]
PlanningOutcome = Literal["plannable", "needs_clarification", "insufficient_coverage", "failed"]
Identifier = Annotated[str, Field(pattern=r"^[a-z_][a-z0-9_]{0,62}$")]
MAX_ROW_LIMIT = 10000
MAX_FILTER_VALUES = 100
_MEMBERSHIP = {"eq": "in", "ieq": "in", "ne": "not_in"}


class PlanModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True, allow_inf_nan=False
    )


class ColumnRef(PlanModel):
    kind: Literal["column"]
    dataset_id: UUID
    column: str = Field(
        min_length=1,
        max_length=63,
        description="sql_name exacto de una columna incluida en datasets[].columns para este mismo dataset_id.",
    )


class ResultRef(PlanModel):
    kind: Literal["result"]
    step_id: Identifier
    field: Identifier


ValueRef = Annotated[ColumnRef | ResultRef, Field(discriminator="kind")]


class DatasetSelection(PlanModel):
    dataset_id: UUID
    version: int = Field(ge=1)
    columns: list[str] = Field(
        min_length=1,
        max_length=1024,
        description="Todos los sql_name que usan los read de este dataset, incluidos los necesarios para filtros y orden temporal; nunca nombres originales ni alias.",
    )
    reason: str = Field(min_length=1, max_length=1500)


class FilterSpec(PlanModel):
    field: ValueRef
    operator: Literal[
        "eq", "ieq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_null", "is_not_null"
    ]
    values: list[str | int | float | bool] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)

    #
    # Convierte una igualdad con varios valores en una pertenencia.
    #
    @model_validator(mode="before")
    @classmethod
    def membership(cls, data):
        if not isinstance(data, dict) or len(data.get("values") or []) < 2:
            return data

        operator = _MEMBERSHIP.get(data.get("operator"))
        return {**data, "operator": operator} if operator else data

    #
    # Valida que la cantidad de valores corresponda al operador del filtro.
    #
    @model_validator(mode="after")
    def valid_arity(self):
        count = len(self.values)
        if self.operator in {"is_null", "is_not_null"}:
            valid = count == 0
        elif self.operator in {"in", "not_in"}:
            valid = count > 0
        else:
            valid = count == 1
        if not valid:
            raise ValueError("La cantidad de valores no corresponde al filtro.")
        return self


class GroupingSpec(PlanModel):
    field: ValueRef
    output_name: Identifier
    time_grain: Literal["raw", "day", "month", "year"] = "raw"


class CalculationSpec(PlanModel):
    function: Literal[
        "count",
        "count_distinct",
        "sum",
        "mean",
        "min",
        "max",
        "difference",
        "ratio",
        "correlation",
        "median",
        "stddev",
        "quantile",
    ]
    inputs: list[ValueRef] = Field(default_factory=list, max_length=2)
    output_name: Identifier
    null_policy: Literal["exclude", "preserve"] = "exclude"
    quantile: float | None = Field(default=None, ge=0, le=1)
    ddof: Literal[0, 1] = 1

    #
    # Valida que los operandos, el cuantil y ddof correspondan a la función de cálculo.
    #
    @model_validator(mode="after")
    def valid_arity(self):
        if (self.function == "quantile") != (self.quantile is not None):
            raise ValueError(
                "quantile requiere una probabilidad explícita entre 0 y 1, exclusiva de esa función."
            )
        if self.function != "stddev":
            self.ddof = 1
        expected = 2 if self.function in {"difference", "ratio", "correlation"} else 1
        if self.function == "count" and not self.inputs:
            return self
        if len(self.inputs) != expected:
            raise ValueError("La cantidad de operandos no corresponde al cálculo.")
        return self


class _VisualizationRequest(PlanModel):
    title: str = Field(min_length=1, max_length=200)
    x: ValueRef
    x_label: str = Field(min_length=1, max_length=150)
    y_label: str = Field(min_length=1, max_length=150)


class CartesianVisualizationRequest(_VisualizationRequest):
    chart_type: Literal["line", "bar", "scatter"]
    y: ValueRef
    series: ValueRef | None = None
    null_policy: Literal["exclude", "preserve"] = "exclude"


class HistogramVisualizationRequest(_VisualizationRequest):
    chart_type: Literal["histogram"]
    bins: int = Field(default=10, ge=1, le=100)
    null_policy: Literal["exclude"] = "exclude"
    y: None = None
    series: None = None


VisualizationRequest = Annotated[
    CartesianVisualizationRequest | HistogramVisualizationRequest,
    Field(discriminator="chart_type"),
]


class JoinKey(PlanModel):
    left: ValueRef
    right: ValueRef


class OrderingSpec(PlanModel):
    field: ValueRef
    direction: Literal["asc", "desc"] = "asc"


class PlanStep(PlanModel):
    id: Identifier
    tool: Literal["sql", "python"]
    operation: Literal[
        "read", "aggregate", "calculate", "sort", "latest", "join", "append", "visualize"
    ]
    depends_on: list[Identifier] = Field(default_factory=list, max_length=12)
    columns: list[ColumnRef] = Field(default_factory=list, max_length=1024)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=30)
    group_by: list[GroupingSpec] = Field(default_factory=list, max_length=20)
    calculations: list[CalculationSpec] = Field(default_factory=list, max_length=20)
    join_on: list[JoinKey] = Field(default_factory=list, max_length=10)
    order_by: list[OrderingSpec] = Field(default_factory=list, max_length=20)
    limit: int | None = Field(default=None, ge=1, le=MAX_ROW_LIMIT)
    visualization: VisualizationRequest | None = None
    purpose: str = Field(min_length=1, max_length=1000)


class ReadPlanStep(PlanStep):
    operation: Literal["read"]
    tool: Literal["sql"]
    depends_on: list[Identifier] = Field(default_factory=list, max_length=0)
    columns: list[ColumnRef] = Field(min_length=1, max_length=1024)
    filters: list[FilterSpec] = Field(
        max_length=30,
        description="Filtros explícitos del alcance solicitado. Si hay query_mentions de kind=value, usá sus textos como valores de filtros en las columnas verificadas correspondientes.",
    )
    group_by: list[GroupingSpec] = Field(default_factory=list, max_length=0)
    calculations: list[CalculationSpec] = Field(default_factory=list, max_length=0)
    join_on: list[JoinKey] = Field(default_factory=list, max_length=0)
    visualization: None = None


class DependentPlanStep(PlanStep):
    operation: Literal["aggregate", "calculate", "sort"]
    depends_on: list[Identifier] = Field(min_length=1, max_length=1)
    columns: list[ColumnRef] = Field(default_factory=list, max_length=0)
    visualization: None = None


class AggregatePlanStep(DependentPlanStep):
    """aggregate/calculate sin cálculos no responde nada: la gramática lo impide."""

    operation: Literal["aggregate", "calculate"]
    calculations: list[CalculationSpec] = Field(min_length=1, max_length=20)


class SortPlanStep(DependentPlanStep):
    operation: Literal["sort"]
    order_by: list[OrderingSpec] = Field(min_length=1, max_length=20)
    group_by: list[GroupingSpec] = Field(default_factory=list, max_length=0)
    calculations: list[CalculationSpec] = Field(default_factory=list, max_length=0)


class LatestOrderingSpec(PlanModel):
    field: ResultRef
    direction: Literal["desc"] = "desc"


class LatestPlanStep(DependentPlanStep):
    operation: Literal["latest"]
    order_by: list[LatestOrderingSpec] = Field(min_length=1, max_length=1)
    group_by: list[GroupingSpec] = Field(default_factory=list, max_length=0)
    calculations: list[CalculationSpec] = Field(default_factory=list, max_length=0)
    limit: None = None


class VisualizePlanStep(DependentPlanStep):
    operation: Literal["visualize"]
    tool: Literal["python"]
    visualization: VisualizationRequest
    filters: list[FilterSpec] = Field(default_factory=list, max_length=0)
    group_by: list[GroupingSpec] = Field(default_factory=list, max_length=0)
    calculations: list[CalculationSpec] = Field(default_factory=list, max_length=0)
    join_on: list[JoinKey] = Field(default_factory=list, max_length=0)
    order_by: list[OrderingSpec] = Field(default_factory=list, max_length=0)
    limit: None = None


class CombinePlanStep(PlanStep):
    operation: Literal["join", "append"]
    depends_on: list[Identifier] = Field(min_length=2, max_length=12)
    columns: list[ColumnRef] = Field(default_factory=list, max_length=0)
    visualization: None = None


ProposedStep = Annotated[
    ReadPlanStep
    | AggregatePlanStep
    | SortPlanStep
    | LatestPlanStep
    | VisualizePlanStep
    | CombinePlanStep,
    Field(discriminator="operation"),
]


class PlannerDecision(PlanModel):
    status: PlanningStatus
    reason: str = Field(min_length=1, max_length=3000)
    datasets: list[DatasetSelection] = Field(default_factory=list, max_length=100)
    steps: list[ProposedStep] = Field(default_factory=list, max_length=12)
    output_steps: list[Identifier] = Field(default_factory=list, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=30)
    clarification_question: str | None = Field(default=None, max_length=1500)
    warnings: list[str] = Field(default_factory=list, max_length=50)

    #
    # Valida que el estado de la decisión sea coherente con sus datasets, pasos y avisos.
    #
    @model_validator(mode="after")
    def coherent_status(self):
        if self.status == "plannable":
            if (
                not self.datasets
                or not self.steps
                or not self.output_steps
                or self.clarification_question
                or self.missing_information
            ):
                raise ValueError(
                    "Un plan completo necesita datasets, pasos y salidas, sin aclaraciones pendientes."
                )
        elif self.steps or self.output_steps:
            raise ValueError("Una planificación bloqueada no debe publicar operaciones.")
        if self.status == "needs_clarification" and not self.clarification_question:
            raise ValueError("La aclaración debe incluir una pregunta concreta.")
        if self.status == "insufficient_coverage" and not self.missing_information:
            raise ValueError("Debe identificar la información faltante.")
        return self


class PlannableDecision(PlannerDecision):
    status: Literal["plannable"]
    datasets: list[DatasetSelection] = Field(min_length=1, max_length=100)
    steps: list[ProposedStep] = Field(min_length=1, max_length=12)
    output_steps: list[Identifier] = Field(min_length=1, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=0)
    clarification_question: None = None


class ClarificationDecision(PlannerDecision):
    status: Literal["needs_clarification"]
    steps: list[ProposedStep] = Field(default_factory=list, max_length=0)
    output_steps: list[Identifier] = Field(default_factory=list, max_length=0)
    clarification_question: str = Field(min_length=1, max_length=1500)


class InsufficientCoverageDecision(PlannerDecision):
    status: Literal["insufficient_coverage"]
    steps: list[ProposedStep] = Field(default_factory=list, max_length=0)
    output_steps: list[Identifier] = Field(default_factory=list, max_length=0)
    missing_information: list[str] = Field(min_length=1, max_length=30)
    clarification_question: None = None


PlannerOutput = Annotated[
    PlannableDecision | ClarificationDecision | InsufficientCoverageDecision,
    Field(discriminator="status"),
]


class PlannerReply(PlanModel):
    decision: PlannerOutput
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class PlanningContextRecord(BaseModel):
    prompt_version: str = "aquai-planner-v3"
    evidence: dict[str, JsonValue]
    sha256: str
    size_bytes: int
    budget_bytes: int
    compacted_fields: list[str] = Field(default_factory=list)
    sent_to_model: bool = False


class ValidationIssue(PlanModel):
    code: str
    severity: Literal["warning", "clarification"]
    message: str
    step_id: str | None = None


class SelectedDataset(VerifiedDataset):
    selection_reason: str


class OutputField(PlanModel):
    name: str
    sql_type: str
    numeric_eligible: bool
    unit: str | None
    unit_status: Literal["documented", "unknown", "dimensionless"]
    sources: list[ColumnRef]


class ValidatedStep(PlanStep):
    output_fields: list[OutputField]


class FilterValueResolution(PlanModel):
    dataset_id: UUID
    column: str
    requested: str
    resolved: str | list[str]
    method: Literal["unique_case_insensitive", "contains_words"] = "unique_case_insensitive"


class RequestedThreshold(PlanModel):
    """Comparación numérica pedida («promedio mayor que 500»); si compara una estadística, filtra su resultado."""

    text: str
    operator: Literal["gt", "ge", "lt", "le"]
    value: int | float
    compares_statistic: bool


class RequestedPeriod(PlanModel):
    """Período pedido («durante 2021»): cada salida acota una fecha o un año exactamente a ese intervalo. Un extremo
    ausente queda abierto: «desde 2020» no tiene fin y «antes de 2021», inicio."""

    text: str
    start: date | None = None
    end: date | None = None

    #
    # Valida que el período pedido tenga al menos un extremo.
    #
    @model_validator(mode="after")
    def has_bound(self):
        if self.start is None and self.end is None:
            raise ValueError("Un período pedido necesita al menos un extremo.")
        return self


class TemporalTextColumn(PlanModel):
    """Columna TEXT cuyos valores distintos nombran períodos explícitos verificados."""

    dataset_id: UUID
    column: str
    day_first: bool | None = None
    interpreted_values: int = Field(ge=0)
    uninterpreted_values: list[str] = Field(default_factory=list, max_length=10)
    precisions: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    complete: bool = False


class PlanningResult(BaseModel):
    status: PlanningOutcome
    llm_status: PlanningStatus | None
    reason: str
    llm_reason: str | None = None
    selected_datasets: list[SelectedDataset] = Field(default_factory=list)
    operations: list[ValidatedStep] = Field(default_factory=list)
    output_steps: list[str] = Field(default_factory=list)
    required_tools: list[Literal["sql", "python"]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    warnings: list[str] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    filter_value_resolutions: list[FilterValueResolution] = Field(default_factory=list)
    temporal_text_columns: list[TemporalTextColumn] = Field(default_factory=list)
    attempts: int = 1
    context: PlanningContextRecord
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    verified_at: datetime


class PlanningInput(PlanModel):
    question: str = Field(min_length=1, max_length=6000)
    candidate_limit: int = Field(default=5, ge=1, le=100)


class QueryPlanningState(PlanningInput):
    retrieval: DatasetRetrievalResult | None = None
    planning: PlanningResult | None = None
    status: PlanningOutcome | None = None
