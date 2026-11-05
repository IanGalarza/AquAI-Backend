from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aquai.infrastructure.config.validation import http_base_url, non_empty_name


class DocumentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCUMENT_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    max_upload_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    max_pages: int = Field(default=1000, gt=0)
    max_expanded_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    conversion_timeout_seconds: float = Field(default=3600, gt=0, allow_inf_nan=False)
    conversion_threads: int = Field(default=6, gt=0, le=64)
    ocr_max_page_chars: int = Field(default=50, ge=0)
    ocr_min_image_coverage: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    symbol_max_readings: int = Field(default=300, ge=0)
    figure_ocr_max: int = Field(default=60, ge=0)
    chunk_tokenizer: str = Field(default="BAAI/bge-m3", min_length=1, max_length=200)
    chunk_max_tokens: int = Field(default=512, ge=64, le=8192)
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="TABULAR_QDRANT_URL")
    qdrant_collection: str = Field(default="aquai_documents", min_length=1, max_length=200)
    index_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    index_batch_size: int = Field(default=32, gt=0, le=256)

    #
    # Valida que la URL de Qdrant sea HTTP sin parámetros ni fragmentos.
    #
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        return http_base_url(value)

    #
    # Valida que el nombre de la colección y el tokenizador no estén vacíos.
    #
    @field_validator("qdrant_collection", "chunk_tokenizer")
    @classmethod
    def validate_names(cls, value: str) -> str:
        return non_empty_name(value)
