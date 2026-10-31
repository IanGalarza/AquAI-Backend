from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConversationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CONVERSATIONS_", extra="ignore")
    generation_timeout_seconds: float = Field(default=1500, ge=1, le=3600)
    max_message_bytes: int = Field(default=2097152, ge=16384, le=10485760)
    max_table_rows: int = Field(default=100, ge=0, le=1000)
    max_table_bytes: int = Field(default=262144, ge=1024, le=1048576)
    max_source_details_bytes: int = Field(default=65536, ge=2048, le=262144)
    history_messages: int = Field(default=8, ge=0, le=8)
    # Mensajes anteriores que se leen de PostgreSQL para recuperar análisis que la pregunta menciona;
    # al modelo sólo llegan los pocos seleccionados, nunca toda la ventana.
    history_recall_messages: int = Field(default=40, ge=0, le=200)
    lock_timeout_ms: int = Field(default=2000, ge=1, le=10000)
    statement_timeout_ms: int = Field(default=10000, ge=1, le=60000)
