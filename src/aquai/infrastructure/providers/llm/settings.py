from functools import lru_cache

from aquai.infrastructure.config.llm import LLMSettings


#
# Devuelve la configuración de Ollama y de los modelos de cada agente, cacheada.
#
@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()
