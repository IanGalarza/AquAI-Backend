"""Composición de Validator y Response Generator, sin endpoints ni clientes nuevos."""

from functools import lru_cache

from aquai.application.useCases.queries.generateResponseHandler import GenerateResponseHandler
from aquai.application.useCases.queries.validateEvidenceHandler import ValidateEvidenceHandler
from aquai.infrastructure.config.answering import AnsweringSettings
from aquai.infrastructure.providers.llm.response import OllamaResponseProvider
from aquai.infrastructure.providers.tabular import get_tabular_settings


@lru_cache
def get_answering_settings():
    return AnsweringSettings()


def create_evidence_validator():
    return ValidateEvidenceHandler()


def create_response_generator():
    settings = get_answering_settings()
    return GenerateResponseHandler(OllamaResponseProvider(get_tabular_settings(), settings), max_preview_rows=settings.max_preview_rows)
