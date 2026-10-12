from pydantic import BaseModel, ConfigDict, Field


class ActivateSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    is_active: bool = Field(alias="isActive", strict=True)
