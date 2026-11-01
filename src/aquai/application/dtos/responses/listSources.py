from datetime import datetime

from pydantic import BaseModel, Field

from aquai.application.dtos.responses.sourceState import SourceStateResponse


#
# Elemento de la lista de fuentes con sus datos básicos y cantidad de datasets.
#
class SourceListItem(SourceStateResponse):
    name: str
    file_type: str
    file_size_bytes: int = Field(ge=0)
    created_at: datetime
    is_active: bool = Field(serialization_alias="isActive")
    dataset_count: int = Field(default=0, ge=0)


#
# Respuesta paginada con la lista de fuentes.
#
class SourceListResponse(BaseModel):
    items: list[SourceListItem]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
