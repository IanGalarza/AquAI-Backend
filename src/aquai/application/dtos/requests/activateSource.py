from pydantic import BaseModel, ConfigDict, Field


#
# Cuerpo de la solicitud para activar o desactivar una fuente (campo isActive).
#
class ActivateSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    is_active: bool = Field(alias="isActive", strict=True)
