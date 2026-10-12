"""Filtros y paginación del listado de fuentes."""

from pydantic import BaseModel, ConfigDict, Field

from aquai.application.dtos.responses.uploadSource import ProcessingStatus, ValidationStatus


class ListSourcesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    name: str | None = Field(default=None, max_length=255)
    processing_status: ProcessingStatus | None = None
    validation_status: ValidationStatus | None = None
