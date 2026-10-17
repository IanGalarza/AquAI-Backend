"""Proyección del detalle que consume la interfaz, sin metadatos internos."""
from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator
from aquai.application.dtos.responses.sourceState import SourceStateResponse


class ColumnDetail(BaseModel):
    original_name: str
    sql_name: str
    sql_type: str
    unit: str | None = None
    meaning: str | None = None
    numeric_eligible: bool | None = None
    representation: str | None = None
    derived_from: str | None = None
    excluded_count: int | None = Field(default=None, ge=0)
    header_missing: bool = False


class SourceLocator(BaseModel):
    sheet: str | None = None
    range: str | None = None
    has_possible_overlap: bool = False

    @model_validator(mode="before")
    @classmethod
    def project_overlap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {**value, "has_possible_overlap": bool(
                value.get("has_possible_overlap") or value.get("possible_overlap")
            )}
        return value


class ProcessingError(BaseModel):
    code: str | None = None
    message: str | None = None


class ProcessingReport(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    warning_count: int = Field(default=0, ge=0)
    errors: list[str | ProcessingError] = Field(default_factory=list)


class DatasetDetail(BaseModel):
    id: UUID
    name: str
    description: str | None
    source_locator: SourceLocator
    row_count: int = Field(ge=0)
    columns_metadata: list[ColumnDetail]
    quality_warnings: list[str]
    is_active: bool
    queryable: bool

    @model_validator(mode="before")
    @classmethod
    def project_description(cls, value: Any) -> Any:
        if isinstance(value, dict):
            description = value.get("semantic_description")
            if isinstance(description, str) and description.strip():
                return {**value, "description": description.strip()}
        return value


class SourceDetailResponse(SourceStateResponse):
    model_config = ConfigDict(populate_by_name=True)
    original_filename: str
    file_type: str
    file_size_bytes: int
    description: str | None
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime
    processing_report: ProcessingReport
    datasets: list[DatasetDetail]
    queryable: bool
