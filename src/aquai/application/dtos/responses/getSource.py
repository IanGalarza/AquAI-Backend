"""Detalle sin originales binarios ni registros completos."""
from datetime import datetime
from typing import Any
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse


class DatasetDetail(BaseModel):
    id: UUID
    name: str
    description: str | None
    semantic_description: str | None
    source_locator: dict[str, Any]
    physical_table_name: str | None
    version: int
    row_count: int = Field(ge=0)
    columns_metadata: list[dict[str, Any]]
    quality_warnings: list[str]
    is_active: bool
    queryable: bool
    semantic_point_id: UUID


class SourceDetailResponse(SourceAcceptedResponse):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    description: str | None
    is_active: bool = Field(serialization_alias="isActive")
    created_at: datetime
    updated_at: datetime | None
    processing_report: dict[str, Any]
    datasets: list[DatasetDetail]
    queryable: bool
