from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DATABASE_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
    )

    database_url: SecretStr = Field(validation_alias="DATABASE_URL")

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, value: object) -> SecretStr:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("DATABASE_URL debe contener una URL de conexión a PostgreSQL.")

        try:
            url = make_url(raw_value.strip())
        except (ArgumentError, ValueError) as exc:
            raise ValueError("DATABASE_URL debe contener una URL de conexión válida.") from exc

        if url.drivername in {"postgres", "postgresql"}:
            url = url.set(drivername="postgresql+psycopg")
        if url.drivername != "postgresql+psycopg":
            raise ValueError("DATABASE_URL debe usar el driver postgresql+psycopg.")
        if not url.database:
            raise ValueError("DATABASE_URL debe incluir el nombre de la base de datos.")

        return SecretStr(url.render_as_string(hide_password=False))

    @property
    def sqlalchemy_url(self) -> URL:
        """Devuelve la URL procesada sin mostrarla en la representación de la configuración."""
        return make_url(self.database_url.get_secret_value())
