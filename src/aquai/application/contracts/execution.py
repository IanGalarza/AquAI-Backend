from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from aquai.application.contracts.planning import OutputField, QueryPlanningState
from aquai.application.contracts.retrieval import DatasetProvenance, RetrievedColumn

ExecutionStatus = Literal["succeeded", "partial", "failed", "not_executed"]
OperationStatus = Literal[
    "executed",
    "executed_as_dependency",
    "pending_python",
    "unsupported",
    "blocked",
    "error",
    "not_executed",
]
ResultValue = str | int | float | bool | Decimal | date | datetime | None


class ExecutionError(BaseModel):
    code: str
    message: str


class ExecutionDataset(BaseModel):
    dataset_id: UUID
    source_id: UUID
    version: int
    provenance: DatasetProvenance
    columns: list[RetrievedColumn]
    warnings: list[str]


class ValueExclusion(BaseModel):
    output_name: str
    group: dict[str, ResultValue] = Field(default_factory=dict)
    input_rows: int = Field(ge=0)
    valid_rows: int = Field(ge=0)
    null_rows: int = Field(ge=0)
    zero_divisions: int = Field(default=0, ge=0)
    policy: Literal["exclude", "preserve"]


class OperationExecution(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    operation_id: str
    status: OperationStatus
    columns: list[OutputField] = Field(default_factory=list)
    rows: list[dict[str, ResultValue]] = Field(default_factory=list)
    rows_returned: int = 0
    rows_processed: int | None = None
    rows_filtered: int = 0
    materialized: bool = False
    truncated: bool = False
    dataset_ids: list[UUID] = Field(default_factory=list)
    query_operation_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float | None = None
    error: ExecutionError | None = None
    input_operation_ids: list[str] = Field(default_factory=list)
    exclusions: list[ValueExclusion] = Field(default_factory=list)


class SqlExecutionResult(BaseModel):
    status: ExecutionStatus
    plan_fingerprint: str | None = None
    operations: list[OperationExecution] = Field(default_factory=list)
    datasets: list[ExecutionDataset] = Field(default_factory=list)
    output_operations: list[str] = Field(default_factory=list)
    pending_operations: list[str] = Field(default_factory=list)
    rows_returned: int = 0
    duration_ms: float = 0
    started_at: datetime
    finished_at: datetime
    read_only: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: ExecutionError | None = None


class ChartAxis(BaseModel):
    field: OutputField
    label: str


class ChartPoint(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    x: str | int | float | bool
    y: float | None


class HistogramBin(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    lower: float
    upper: float
    count: int = Field(ge=0)
    upper_inclusive: bool


class ChartSeries(BaseModel):
    name: str | None = None
    points: list[ChartPoint] = Field(default_factory=list)
    bins: list[HistogramBin] = Field(default_factory=list)


class DatasetVersionReference(BaseModel):
    dataset_id: UUID
    source_id: UUID
    version: int


class PlotlyFigure(BaseModel):
    """Figura completa de presentación; nunca conserva objetos de la librería."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    renderer: Literal["plotly"] = "plotly"
    renderer_version: str = Field(description="Versión de plotly.py que construyó la figura.")
    plotly_js_version: str
    data: list[dict[str, JsonValue]]
    layout: dict[str, JsonValue]
    config: dict[str, JsonValue]


class VisualizationResult(BaseModel):
    operation_id: str
    chart_type: Literal["line", "bar", "scatter", "histogram"]
    title: str
    x_axis: ChartAxis
    y_axis: ChartAxis
    series_field: OutputField | None = None
    series: list[ChartSeries]
    datasets: list[DatasetVersionReference]
    input_operation_ids: list[str]
    input_rows: int
    excluded_rows: int
    numeric_encoding: Literal["float64_for_display"] = "float64_for_display"
    warnings: list[str] = Field(default_factory=list)
    figure: PlotlyFigure | None = None


class PythonExecutionResult(BaseModel):
    status: ExecutionStatus
    plan_fingerprint: str | None = None
    sql_input_fingerprint: str | None = None
    operations: list[OperationExecution] = Field(default_factory=list)
    datasets: list[ExecutionDataset] = Field(default_factory=list)
    visualizations: list[VisualizationResult] = Field(default_factory=list)
    output_operations: list[str] = Field(default_factory=list)
    pending_operations: list[str] = Field(default_factory=list)
    rows_returned: int = 0
    duration_ms: float = 0
    started_at: datetime
    finished_at: datetime
    input_fingerprint: str | None = None
    decimal_precision: int = 38
    warnings: list[str] = Field(default_factory=list)
    error: ExecutionError | None = None


class QueryExecutionState(QueryPlanningState):
    execution: SqlExecutionResult | None = None
    python_execution: PythonExecutionResult | None = None
    visualizations: list[VisualizationResult] = Field(default_factory=list)
    execution_warnings: list[str] = Field(default_factory=list)
    execution_status: ExecutionStatus = "not_executed"
    pending_operations: list[str] = Field(default_factory=list)
