import logging
from time import perf_counter

from aquai.application.answering.context import response_language
from aquai.application.answering.drafting import asks_interpretation
from aquai.application.answering.hypotheses import unit_hypotheses
from aquai.application.answering.measures import measured_results, reference_targets
from aquai.application.answering.references import (
    ambiguity_notes,
    chart_lines,
    comparison_facts,
    incompatibility_notes,
    local_references,
    most_specified,
    reference_option,
    threshold_references,
    threshold_value,
    verified_references,
)
from aquai.application.contracts.answering import DocumentContrast, ReferenceValue
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.documents import DocumentEvidenceProvider
from aquai.application.matching import reference_operator

_logger = logging.getLogger(__name__)
_INCOMPLETE_EXTRACTION = (
    "La extracción de valores de referencia llegó incompleta; sólo se usaron los valores completos."
)
_FAILED_EXTRACTION = "No se pudieron extraer valores de referencia de los fragmentos."
_MAX_TIERS = 3


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
        asked = asks_interpretation(question)
        drafts = await self._proposed_or_none(results, fragments, contrast) if asked else []
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
        contrast.notes = notes or [
            *incompatibility_notes(contrast.references, language),
            *ambiguity_notes(contrast.references, results, language),
        ]
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
    # Verifica, antes de planificar, las guías o límites documentados con que se pide comparar la variable.
    #
    async def documented_thresholds(self, state, retrieval) -> list[ReferenceValue]:
        targets = reference_targets(state)
        fragments = [item for item in retrieval.fragments if not item.doubts]
        if not targets or not fragments:
            return []

        contrast = DocumentContrast(model=self._evidence.model)
        drafts = await self._proposed_or_none(targets, fragments, contrast)
        found = verified_references(drafts, fragments, targets, state.routing.original_question)
        _logger.info("Referencia para comparar propuestas=%s verificadas=%s", contrast.proposed, len(found))

        return most_specified(threshold_references(found))

    #
    # Devuelve el enrutado con la referencia resuelta: cada valor verificado se vuelve un umbral del plan, y
    # si son varios niveles, cada uno con la condición en que rige, se calculan por separado. Varios valores
    # sin condición, o ninguno verificable, pasan a una respuesta documental que presenta lo que dicen.
    #
    @staticmethod
    def resolved_routing(state, retrieval, found):
        routing = state.routing
        options = list(dict.fromkeys(reference_option(item, routing.language) for item in found))
        cited = {"reference_options": options, "reference_documents": list(dict.fromkeys(item.evidence_id for item in found))}
        values = sorted({threshold_value(item) for item in found})
        tiered = len(values) > 1 and all(item.condition for item in found)
        if len(values) == 1 or (tiered and len(values) <= _MAX_TIERS):
            operator = reference_operator(routing.original_question) or "gt"
            mentions = [DataMention(kind="threshold", text=_threshold_text(operator, value)) for value in values]
            return routing.model_copy(update={"data_mentions": [*routing.data_mentions, *mentions], **cited})
        if retrieval.status == "found":
            return routing.model_copy(update={"route": "document", **cited})

        return routing

    #
    # Pide las referencias al modelo; sin ellas sigue valiendo lo que se lee de los fragmentos.
    #
    async def _proposed_or_none(self, results, fragments, contrast):
        try:
            return await self._proposed(results, fragments, contrast)
        except LLMProviderError as error:
            _logger.warning(
                "No se pudieron extraer referencias: %s: %s (tokens de salida=%s)",
                type(error).__name__,
                error,
                getattr(error, "output_tokens", None),
            )
            contrast.warnings.append(_FAILED_EXTRACTION)
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
        if not usage.complete:
            _logger.warning("Extracción de referencias truncada: %s valores completos", contrast.proposed)
            contrast.warnings.append(_INCOMPLETE_EXTRACTION)

        return extraction.references


#
# Escribe el umbral de un valor documentado como una comparación que el Planner aplica tal cual.
#
def _threshold_text(operator, value) -> str:
    words = {"gt": "mayor que", "lt": "menor que"}

    return f"{words[operator]} {format(value, 'f')}"


#
# Quita las referencias repetidas, conservando la primera lectura de cada valor.
#
def _distinct_references(references) -> list:
    found = {}
    for item in references:
        found.setdefault((item.result_field, item.low, item.high, item.evidence_id), item)

    return list(found.values())
