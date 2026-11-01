from pydantic import AnyHttpUrl, Field, TypeAdapter, field_validator, model_validator
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
    default_model: str = Field(default="qwen3.5:9b", min_length=1, max_length=200)
    llm_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    llm_num_ctx: int = Field(default=32768, ge=1024)
    llm_think: bool = False
    llm_keep_alive: str | None = Field(default=None, pattern=r"^-?\d+(?:\.\d+)?(?:ms|s|m|h)?$")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_cells: int = Field(default=2_000_000, gt=0)
    max_sheets: int = Field(default=30, gt=0, le=30)
    max_expanded_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    sample_rows: int = Field(default=12, gt=0, le=100)
    max_datasets: int = Field(default=100, gt=0, le=1000)
    max_columns: int = Field(default=1024, gt=0, le=1500)
    structure_assistance: bool = True
    embedding_model: str = Field(default="nomic-embed-text", min_length=1, max_length=200)
    embedding_timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = Field(default="aquai_datasets", min_length=1, max_length=200)
    index_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    #
    # Valida que la URL sea HTTP sin parámetros ni fragmentos y quita la barra final.
    #
    @field_validator("ollama_base_url", "qdrant_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        url = TypeAdapter(AnyHttpUrl).validate_python(value)
        if url.query or url.fragment:
            raise ValueError("La URL no debe incluir parámetros ni fragmentos.")
        return str(url).rstrip("/")

    #
    # Valida que el nombre de modelo o colección no esté vacío.
    #
    @field_validator("default_model", "embedding_model", "qdrant_collection")
    @classmethod
    def validate_default_model(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("El modelo o colección no puede estar vacío.")
        return model

    #
    # Valida que los modelos de descripción y de embeddings sean distintos.
    #
    @model_validator(mode="after")
    def validate_separate_embedding_model(self) -> "TabularSettings":
        if self.default_model.removesuffix(":latest") == self.embedding_model.removesuffix(
            ":latest"
        ):
            raise ValueError("La descripción y los embeddings requieren modelos diferentes.")
        return self
