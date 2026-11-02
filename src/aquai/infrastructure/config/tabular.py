from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aquai.infrastructure.config.validation import http_base_url, non_empty_name


class TabularSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TABULAR_",
        case_sensitive=False,
        extra="ignore",
    )

    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_cells: int = Field(default=2_000_000, gt=0)
    max_sheets: int = Field(default=30, gt=0, le=30)
    max_expanded_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    sample_rows: int = Field(default=12, gt=0, le=100)
    max_datasets: int = Field(default=100, gt=0, le=1000)
    max_columns: int = Field(default=1024, gt=0, le=1500)
    structure_assistance: bool = True
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = Field(default="aquai_datasets", min_length=1, max_length=200)
    index_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    #
    # Valida que la URL de Qdrant sea HTTP sin parámetros ni fragmentos.
    #
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        return http_base_url(value)

    #
    # Valida que el nombre de la colección no esté vacío.
    #
    @field_validator("qdrant_collection")
    @classmethod
    def validate_qdrant_collection(cls, value: str) -> str:
        return non_empty_name(value)
