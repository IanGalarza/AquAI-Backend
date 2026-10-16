"""Filtros y paginación del listado de fuentes."""
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from aquai.domain.enums.sourceStatus import ProcessingStatus


class SourceSort(StrEnum):
    NEWEST = "newest"
    OLDEST = "oldest"
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"


class ListSourcesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    name: str | None = Field(default=None, max_length=255)
    processing_status: ProcessingStatus | None = None
    sort: SourceSort = SourceSort.NEWEST
