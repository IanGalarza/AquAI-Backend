from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SqlExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SQL_EXECUTION_", extra="ignore", hide_input_in_errors=True
    )
    database_url: SecretStr | None = None
    dataset_schema: str = Field(default="public", pattern=r"^[a-z_][a-z0-9_]{0,62}$")
    statement_timeout_ms: int = Field(default=10000, ge=1, le=60000)
    lock_timeout_ms: int = Field(default=2000, ge=1, le=10000)
    max_duration_ms: int = Field(default=30000, ge=1, le=120000)
    max_rows: int = Field(default=1000, ge=1, le=10000)
    max_result_bytes: int = Field(default=1048576, ge=1024, le=10485760)
    max_row_bytes: int = Field(default=65536, ge=128, le=1048576)

    #
    # Convierte una URL vacía en None.
    #
    @field_validator("database_url", mode="before")
    @classmethod
    def empty_url(cls, value):
        return None if value == "" else value

    #
    # Valida que el schema de datasets no sea uno interno del motor.
    #
    @field_validator("dataset_schema")
    @classmethod
    def application_schema(cls, value):
        if value.startswith("pg_") or value == "information_schema":
            raise ValueError("El schema debe pertenecer a AquAI.")
        return value
