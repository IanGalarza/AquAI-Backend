"""Resultados acotados de ejecución, sin objetos de persistencia ni SQL libre."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from aquai.application.contracts.planning import OutputField, QueryPlanningState
from aquai.application.contracts.retrieval import DatasetProvenance, RetrievedColumn


ExecutionStatus = Literal["succeeded", "partial", "failed", "not_executed"]
OperationStatus = Literal["executed", "executed_as_dependency", "pending_python", "unsupported", "blocked", "error", "not_executed"]
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


class OperationExecution(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    operation_id: str
    status: OperationStatus
    columns: list[OutputField] = Field(default_factory=list)
    rows: list[dict[str, ResultValue]] = Field(default_factory=list)
    rows_returned: int = 0
    materialized: bool = False
    truncated: bool = False
    dataset_ids: list[UUID] = Field(default_factory=list)
    query_operation_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float | None = None
    error: ExecutionError | None = None


class SqlExecutionResult(BaseModel):
    status: ExecutionStatus
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


class QueryExecutionState(QueryPlanningState):
    execution: SqlExecutionResult | None = None
    execution_status: ExecutionStatus = "not_executed"
    pending_operations: list[str] = Field(default_factory=list)
