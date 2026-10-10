from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TabularSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TABULAR_",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    default_model: str = Field(default="qwen3:14b", min_length=1, max_length=200)
    llm_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_cells: int = Field(default=2_000_000, gt=0)
    max_sheets: int = Field(default=30, gt=0, le=30)
    max_expanded_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    sample_rows: int = Field(default=12, gt=0, le=100)

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        url = TypeAdapter(AnyHttpUrl).validate_python(value)
        if url.query or url.fragment:
            raise ValueError("La URL de Ollama no debe incluir parámetros ni fragmentos.")
        return str(url).rstrip("/")

    @field_validator("default_model")
    @classmethod
    def validate_default_model(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("El modelo predeterminado no puede estar vacío.")
        return model
