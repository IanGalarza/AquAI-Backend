from pydantic import ConfigDict, Field

from aquai.application.dtos.responses.sourceState import SourceStateResponse


#
# Respuesta con el estado de activación de una fuente y si se puede consultar.
#
class SourceActivationResponse(SourceStateResponse):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)
    is_active: bool = Field(serialization_alias="isActive")
    queryable: bool
