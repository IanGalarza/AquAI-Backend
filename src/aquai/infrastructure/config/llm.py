from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aquai.infrastructure.config.validation import http_base_url, non_empty_name


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_OLLAMA_",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = "http://localhost:11434"
    keep_alive: str | None = Field(default=None, pattern=r"^-?\d+(?:\.\d+)?(?:ms|s|m|h)?$")

    #
    # Valida que la URL de Ollama sea HTTP sin parámetros ni fragmentos.
    #
    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return http_base_url(value)


class AgentModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = Field(default="qwen3.5:9b", min_length=1, max_length=200)
    num_ctx: int = Field(default=32768, ge=1024)
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    think: bool = False
    timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)

    #
    # Valida que el nombre del modelo no esté vacío.
    #
    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return non_empty_name(value)


class TabularModelSettings(AgentModelSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_TABULAR_")


class RouterModelSettings(AgentModelSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_ROUTER_")

    max_output_tokens: int = Field(default=768, ge=1)
    context_max_bytes: int = Field(default=24000, ge=2048, le=96000)


class GeneralModelSettings(AgentModelSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_GENERAL_")

    max_output_tokens: int = Field(default=1536, ge=1)
    context_max_bytes: int = Field(default=24000, ge=2048, le=96000)


class PlannerModelSettings(AgentModelSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_PLANNER_")

    max_output_tokens: int = Field(default=4096, ge=1)
    context_max_bytes: int = Field(default=96000, ge=2048, le=96000)


class ResponseModelSettings(AgentModelSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_RESPONSE_")

    max_output_tokens: int = Field(default=2048, ge=1)
    context_max_bytes: int = Field(default=24000, ge=2048, le=96000)


class EmbeddingModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLM_EMBEDDING_",
        case_sensitive=False,
        extra="ignore",
    )

    model: str = Field(default="nomic-embed-text", min_length=1, max_length=200)
    timeout_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)

    #
    # Valida que el nombre del modelo no esté vacío.
    #
    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return non_empty_name(value)


class LLMSettings(BaseModel):
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    tabular: TabularModelSettings = Field(default_factory=TabularModelSettings)
    router: RouterModelSettings = Field(default_factory=RouterModelSettings)
    general: GeneralModelSettings = Field(default_factory=GeneralModelSettings)
    planner: PlannerModelSettings = Field(default_factory=PlannerModelSettings)
    response: ResponseModelSettings = Field(default_factory=ResponseModelSettings)
    embedding: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)

    #
    # Devuelve la configuración de los agentes que usan un modelo de chat.
    #
    @property
    def chat_agents(self) -> tuple[AgentModelSettings, ...]:
        return (self.tabular, self.router, self.general, self.planner, self.response)

    #
    # Valida que ningún agente de chat use el modelo de embeddings.
    #
    @model_validator(mode="after")
    def validate_separate_embedding_model(self) -> "LLMSettings":
        embedding_model = self.embedding.model.removesuffix(":latest")
        chat_models = {agent.model.removesuffix(":latest") for agent in self.chat_agents}
        if embedding_model in chat_models:
            raise ValueError("Los agentes de chat y los embeddings requieren modelos diferentes.")
        return self
