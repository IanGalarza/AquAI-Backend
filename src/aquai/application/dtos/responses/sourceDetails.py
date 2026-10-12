"""Objetos estables del detalle; sólo celdas y parámetros conservan JSON variable."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, JsonValue

from aquai.application.contracts.tabular.results import TabularFinding
from aquai.application.contracts.tabular.structure import SampleRow
from aquai.application.dtos.responses.reviewSource import SourceDecisionResponse


class SourceBlock(BaseModel):
    region_id: str
    sheet_name: str
    start_row: int = Field(ge=1)
    end_row: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_column: int = Field(ge=1)
    kind: Literal["observations", "total", "legend", "derived", "summary", "header", "repeated_header", "unknown"]
    reason: str | None = None
    nonempty_cells: int = Field(default=0, ge=0)
    sample: list[SampleRow] = Field(default_factory=list)
    column_indices: list[int] = Field(default_factory=list)


class SourcePreview(BaseModel):
    sheet_name: str
    block_id: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[SampleRow] = Field(default_factory=list)


class SourceFinding(BaseModel):
    id: str
    kind: Literal["review", "technical", "validation"]
    code: str | None = None
    message: str
    sheet_name: str | None = None
    block_id: str | None = None
    cell: str | None = None
    region_id: str | None = None
    range: str | None = None
    requires_review: bool = True
    severity: Literal["warning", "error"] = "warning"


class SourceReviewOption(BaseModel):
    action: str
    description: str
    values: dict[str, JsonValue] = Field(default_factory=dict)


class SourceReview(BaseModel):
    id: str
    review_id: str | None = None
    source_id: UUID | None = None
    revision: int = Field(default=0, ge=0)
    code: str
    message: str
    kind: Literal["review"] = "review"
    sheet_name: str | None = None
    block_id: str | None = None
    cell: str | None = None
    range: str | None = None
    region_id: str | None = None
    options: list[SourceReviewOption] = Field(default_factory=list)
    finding: TabularFinding | None = None
    blocked_reason: str | None = None


class SourceDecisionScope(BaseModel):
    source_id: UUID
    sheet_name: str | None = None
    block_id: str | None = None
    range: str | None = None
    cell: str | None = None
    region_id: str | None = None


class SourceDecision(BaseModel):
    review_id: str
    key: str
    action: str
    values: dict[str, JsonValue] = Field(default_factory=dict)
    scope: SourceDecisionScope
    created_at: str
    response: SourceDecisionResponse


class SourceTechnicalError(BaseModel):
    code: str
    message: str
    sheet_name: str | None = None
    cell: str | None = None
    range: str | None = None
    created_at: str | None = None


class SourceDatasetStorage(BaseModel):
    status: str | None = None
    message: str | None = None
