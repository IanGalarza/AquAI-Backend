from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AnsweringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RESPONSE_", extra="ignore")
    max_preview_rows: int = Field(default=3, ge=0, le=20)
