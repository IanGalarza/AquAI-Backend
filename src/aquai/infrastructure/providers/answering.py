from functools import lru_cache

from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.Services.validateEvidenceService import ValidateEvidenceService
from aquai.infrastructure.config.answering import AnsweringSettings
from aquai.infrastructure.providers.llm.response import OllamaResponseProvider
from aquai.infrastructure.providers.tabular import get_tabular_settings


#
# Devuelve la configuración de redacción de respuestas, cacheada.
#
@lru_cache
def get_answering_settings():
    return AnsweringSettings()


#
# Crea el servicio que valida la evidencia de las respuestas.
#
def create_evidence_validator():
    return ValidateEvidenceService()


#
# Crea el servicio que genera la respuesta con el proveedor de Ollama.
#
def create_response_generator():
    settings = get_answering_settings()

    return GenerateResponseService(
        OllamaResponseProvider(get_tabular_settings(), settings),
        max_preview_rows=settings.max_preview_rows,
    )
