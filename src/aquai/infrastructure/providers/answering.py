from functools import lru_cache

from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.Services.validateEvidenceService import ValidateEvidenceService
from aquai.infrastructure.config.answering import AnsweringSettings
from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider
from aquai.infrastructure.providers.llm.response import OllamaResponseProvider
from aquai.infrastructure.providers.llm.settings import get_llm_settings


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
    models = get_llm_settings()

    return GenerateResponseService(
        OllamaResponseProvider(models.ollama, models.response),
        embeddings=OllamaEmbeddingProvider(models.ollama, models.document_embedding),
        max_preview_rows=get_answering_settings().max_preview_rows,
        draft_attempts=models.response.max_attempts,
        document_draft_attempts=models.response.document_max_attempts,
    )
