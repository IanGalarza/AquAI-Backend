import logging
from time import perf_counter

from aquai.application.answering.context import response_language
from aquai.application.answering.hypotheses import unit_hypotheses
from aquai.application.answering.measures import measured_results
from aquai.application.answering.references import (
    chart_lines,
    comparison_facts,
    incompatibility_notes,
    local_references,
    verified_references,
)
from aquai.application.contracts.answering import DocumentContrast
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.documents import DocumentEvidenceProvider

_logger = logging.getLogger(__name__)


class ContrastReferencesService:
    #
    # Guarda el asistente que propone los valores de referencia.
    #
    def __init__(self, evidence: DocumentEvidenceProvider):
        self._evidence = evidence

    #
    # Contrasta los resultados con la evidencia del turno; nada de esto altera el análisis.
    #
    async def contrast(self, state) -> DocumentContrast:
        started = perf_counter()
        language = response_language(state.question, state.response_language)
        question = state.routing.original_question
        retrieval = state.document_retrieval
        results = measured_results(state, language)
        fragments = [item for item in retrieval.fragments if not item.doubts] if retrieval else []
        if not results or not fragments:
            return DocumentContrast()

        contrast = DocumentContrast(model=self._evidence.model)
        drafts = await self._proposed_or_none(results, fragments, contrast)
        places = [
            item.text
            for item in state.routing.data_mentions
            if not isinstance(item, str) and item.kind == "value"
        ]
        contrast.references = _distinct_references(
            [
                *local_references(fragments, results, question, places),
                *verified_references(drafts, fragments, results, question, places),
            ]
        )
        contrast.comparisons = comparison_facts(contrast.references, results, language)
        contrast.hypotheses = unit_hypotheses(results, fragments, language)
        contrast.lines, notes = chart_lines(state, contrast.references, results, question, language)
        contrast.notes = notes or incompatibility_notes(contrast.references, language)
        contrast.duration_ms = round((perf_counter() - started) * 1000, 3)
        _logger.info(
            "Contraste propuestas=%s verificadas=%s hipótesis=%s líneas=%s ms=%s",
            contrast.proposed,
            len(contrast.references),
            len(contrast.hypotheses),
            len(contrast.lines),
            contrast.duration_ms,
        )

        return contrast

    #
    # Pide las referencias al modelo; sin ellas sigue valiendo lo que se lee de los fragmentos.
    #
    async def _proposed_or_none(self, results, fragments, contrast):
        try:
            return await self._proposed(results, fragments, contrast)
        except LLMProviderError as error:
            _logger.warning("No se pudieron extraer referencias: %s", type(error).__name__)
            return []

    #
    # Pide al modelo los valores de referencia de los fragmentos, sin aceptarlos todavía.
    #
    async def _proposed(self, results, fragments, contrast):
        context = {
            "targets": [
                {"variable": item.variable, "column": item.column, "unit": item.unit}
                for item in results[:4]
            ],
            "fragments": [
                {"id": item.evidence_id, "section": item.section[-2:], "text": item.text}
                for item in fragments
            ],
        }
        extraction, usage = await self._evidence.references(
            context, [item.evidence_id for item in fragments]
        )
        contrast.proposed = len(extraction.references)
        contrast.input_tokens = usage.input_tokens
        contrast.output_tokens = usage.output_tokens

        return extraction.references


#
# Quita las referencias repetidas, conservando la primera lectura de cada valor.
#
def _distinct_references(references) -> list:
    found = {}
    for item in references:
        found.setdefault((item.result_field, item.low, item.high, item.evidence_id), item)

    return list(found.values())
