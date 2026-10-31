import asyncio
import re
from time import perf_counter

from pydantic import TypeAdapter, ValidationError

from aquai.application.answering.catalog import catalog_segments
from aquai.application.answering.segments import answer_text, segments_from_text
from aquai.application.contracts.answering import (
    AnswerSegment,
    GenerationRecord,
    QueryResponse,
    ResponseOperationReference,
    SourceReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.routing import GeneralResponseProvider
from aquai.application.interfaces.sources import SourceQueries

# Bloques o firmas de código: AquAI explica en palabras y no entrega programas.
_CODE = re.compile(
    r"```|^\s*(?:def|class|import|from\s+\S+\s+import|SELECT|function|public\s+class)\b",
    re.IGNORECASE | re.MULTILINE,
)
_SEGMENTS = TypeAdapter(list[AnswerSegment])


class GenerateDirectResponseService:
    def __init__(self, provider: GeneralResponseProvider, catalog: SourceQueries | None = None):
        self._provider = provider
        self._catalog = catalog

    async def generate(self, state):
        routing = state.routing
        if routing is None or routing.route == "data":
            raise ValueError("La respuesta directa requiere una ruta no analítica.")

        response = _initial_response(state, routing)

        if routing.route == "needs_clarification":
            return self._clarification_response(response, routing)

        if routing.route == "failed":
            return self._failure(response, "routing_failed", routing.error)

        if routing.route == "catalog":
            return await self._catalog_response(response, routing)

        selected = _selected_history(state, routing)

        if selected is None:
            return self._failure(
                response,
                "routing_failed",
                "La ruta cita mensajes que no están en el historial de la conversación.",
            )

        if routing.route == "historical":
            if len(selected) != 1 or selected[0].analysis is None:
                return self._failure(
                    response,
                    "historical_reference_missing",
                    "La ruta histórica no cita un único resultado anterior recuperable.",
                )

            return self._historical_response(response, routing, selected[0])

        if any(item.scientific for item in selected):
            raise ValueError("La ruta general no puede interpretar resultados científicos.")

        return await self._general_response(response, state, routing, selected)

    def _clarification_response(self, response, routing):
        response.response_type = "clarification"
        response.status = "needs_clarification"
        response.clarification_question = routing.clarification_question
        self._publish(response, routing.clarification_question or "¿Podés precisar la consulta?")

        return response

    async def _catalog_response(self, response, routing):
        """El inventario proviene de PostgreSQL en este turno, nunca del conocimiento del LLM."""
        response.response_type = "catalog"
        if self._catalog is None:
            return self._failure(response, "catalog_unavailable", "El catálogo no está disponible.")

        started = perf_counter()
        try:
            catalog = await asyncio.to_thread(self._catalog.catalog_overview)
        except Exception:
            return self._failure(
                response, "catalog_unavailable", "No se pudo consultar el catálogo de fuentes."
            )

        bound_dataset_ids = [item.dataset_id for item in routing.dataset_bindings]
        response.segments = catalog_segments(catalog, bound_dataset_ids, routing.language)
        response.answer = answer_text(response.segments)
        response.limitations.append(
            "Inventario consultado en PostgreSQL durante esta respuesta; no se leyeron registros de los datasets."
            if routing.language == "es"
            else "Inventory read from PostgreSQL for this answer; no dataset records were read."
        )
        response.timings_ms["catalog"] = round((perf_counter() - started) * 1000, 3)

        return response

    def _historical_response(self, response, routing, message):
        analysis = message.analysis
        response.response_type = "historical"
        response.answer = message.text
        response.historical_message_ids = [message.message_id]
        response.sources = [SourceReference.model_validate(item) for item in analysis.sources]
        response.operation_references = [
            ResponseOperationReference.model_validate(item)
            for item in analysis.operation_references
        ]

        note = (
            "Se reproduce una respuesta anterior con sus versiones originales; "
            "no se ejecutó una nueva consulta."
            if routing.language == "es"
            else "This reproduces a previous answer with its original versions; no new query ran."
        )
        response.limitations.append(note)

        if not message.text_complete:
            response.status = "partial"
            response.limitations.append("El texto histórico disponible está abreviado.")

        segments = self._historical_segments(message, response)
        attached = any(segment.attachment for segment in segments)
        notice = AnswerSegment(kind="note", text=_replay_notice(routing.language, attached))
        response.segments = [*segments, notice]

        return response

    async def _general_response(self, response, state, routing, selected):
        context = {
            "question": state.question,
            "language": routing.language,
            "conversation_context": [{"role": item.role, "text": item.text} for item in selected],
        }
        started = perf_counter()
        response.generation.model = self._provider.model

        try:
            reply = await self._provider.generate(context)
            self._publish_reply(response, reply, routing.language)
        except LLMProviderError as error:
            _record_error_tokens(response.generation, error)
            self._failure(response, "general_generation_failed", None)

        response.generation.duration_ms = round((perf_counter() - started) * 1000, 3)
        response.timings_ms["general"] = response.generation.duration_ms

        return response

    def _publish_reply(self, response, reply, language):
        response.generation.input_tokens = reply.input_tokens
        response.generation.output_tokens = reply.output_tokens

        if reply.draft.language != language or not reply.draft.answer.strip():
            raise LLMProviderError("La respuesta general no respeta el idioma solicitado.")

        self._publish(response, reply.draft.answer.strip())

        if _CODE.search(response.answer):
            self._publish(response, _code_refusal(language))
            response.limitations.append("Se omitió código generado por el modelo.")

        response.generation.model = reply.model
        response.generation.status = "generated"

    @staticmethod
    def _publish(response, text):
        response.answer = text
        response.segments = segments_from_text(text)

    @staticmethod
    def _historical_segments(message, response):
        if message.text_complete and message.segments:
            try:
                return _SEGMENTS.validate_python(message.segments)
            except ValidationError:
                pass

        evidence_ids = [item.evidence_id for item in response.operation_references]

        return segments_from_text(message.text, evidence_ids)

    @classmethod
    def _failure(cls, response, code, message):
        response.status = "failed"
        text = (
            "No pude generar una respuesta válida. Podés volver a intentarlo."
            if response.language == "es"
            else "I could not generate a valid answer. You can try again."
        )
        cls._publish(response, text)

        response.generation.status = "failed"
        response.generation.error = ExecutionError(
            code=code, message=message or "El modelo no completó una respuesta válida."
        )
        response.errors = [response.generation.error]

        return response


def _initial_response(state, routing) -> QueryResponse:
    return QueryResponse(
        response_type="general",
        status="answered",
        validation_status=None,
        answer="",
        language=routing.language,
        routing=routing,
        generation=GenerationRecord(status="deterministic"),
        timings_ms=dict(state.stage_durations_ms),
    )


def _selected_history(state, routing):
    messages = {item.message_id: item for item in state.conversation_history}

    if any(key not in messages for key in routing.history_message_ids):
        return None

    return [messages[key] for key in routing.history_message_ids]


def _replay_notice(language, attached):
    if language == "es":
        suffix = " Sus tablas y gráficos están en ese mensaje." if attached else ""
        return "Te muestro la respuesta anterior; no volví a consultar los datos." + suffix

    suffix = " Its tables and charts are in that message." if attached else ""

    return "This is the previous answer; I did not query the data again." + suffix


def _code_refusal(language):
    if language == "es":
        return (
            "AquAI no genera código de programación. Puedo explicarte el concepto en palabras "
            "o ayudarte a analizar los datos que ya cargaste: promedios, máximos, últimas "
            "mediciones o gráficos."
        )

    return (
        "AquAI does not write programming code. I can explain the concept in words or help "
        "you analyze the loaded data: averages, maxima, latest measurements or charts."
    )


def _record_error_tokens(generation, error):
    for field in ("input_tokens", "output_tokens"):
        value = getattr(error, field, None)
        if value is not None:
            setattr(generation, field, value)
