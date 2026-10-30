"""Respuestas generales y aclaraciones; no usa evidencia científica nueva."""

import asyncio
import re
from time import perf_counter

from aquai.application.answering.catalog import catalog_answer
from aquai.application.contracts.answering import (
    GenerationRecord, QueryResponse, ResponseOperationReference, SourceReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.routing import GeneralResponseProvider
from aquai.application.interfaces.sources import SourceQueries


# Bloques o firmas de código: AquAI explica en palabras y no entrega programas.
_CODE = re.compile(r"```|^\s*(?:def|class|import|from\s+\S+\s+import|SELECT|function|public\s+class)\b",
                   re.IGNORECASE | re.MULTILINE)


class GenerateDirectResponseHandler:
    def __init__(self, provider: GeneralResponseProvider, catalog: SourceQueries | None = None):
        self._provider = provider
        self._catalog = catalog

    async def handle(self, state):
        routing = state.routing
        if routing is None or routing.route == "data":
            raise ValueError("La respuesta directa requiere una ruta no analítica.")
        response = QueryResponse(
            response_type="general", status="answered", validation_status=None,
            answer="", language=routing.language, routing=routing,
            generation=GenerationRecord(status="deterministic"),
            timings_ms=dict(state.stage_durations_ms),
        )
        if routing.route == "needs_clarification":
            response.response_type = "clarification"
            response.status = "needs_clarification"
            response.clarification_question = routing.clarification_question
            response.answer = routing.clarification_question or "¿Podés precisar la consulta?"
            return response
        if routing.route == "failed":
            return self._failure(response, "routing_failed", routing.error)
        if routing.route == "catalog":
            return await self._catalog_response(response, routing)
        messages = {
            item.message_id: item for item in state.conversation_history
        }
        selected = [messages[key] for key in routing.history_message_ids]
        if routing.route == "historical":
            message = selected[0]
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
                if routing.language == "es" else
                "This reproduces a previous answer with its original versions; no new query ran."
            )
            response.limitations.append(note)
            if not message.text_complete:
                response.status = "partial"
                response.limitations.append("El texto histórico disponible está abreviado.")
            return response
        if any(item.scientific for item in selected):
            raise ValueError("La ruta general no puede interpretar resultados científicos.")
        context = {
            "question": state.question, "language": routing.language,
            "conversation_context": [
                {"role": item.role, "text": item.text} for item in selected
            ],
        }
        started = perf_counter()
        response.generation.model = self._provider.model
        try:
            reply = await self._provider.generate(context)
            response.generation.input_tokens = reply.input_tokens
            response.generation.output_tokens = reply.output_tokens
            if reply.draft.language != routing.language or not reply.draft.answer.strip():
                raise LLMProviderError("La respuesta general no respeta el idioma solicitado.")
            response.answer = reply.draft.answer.strip()
            if _CODE.search(response.answer):
                response.answer = (
                    "AquAI no genera código de programación. Puedo explicarte el concepto en palabras "
                    "o ayudarte a analizar los datos que ya cargaste: promedios, máximos, últimas "
                    "mediciones o gráficos." if routing.language == "es" else
                    "AquAI does not write programming code. I can explain the concept in words or help "
                    "you analyze the loaded data: averages, maxima, latest measurements or charts."
                )
                response.limitations.append("Se omitió código generado por el modelo.")
            response.generation.model = reply.model
            response.generation.status = "generated"
        except LLMProviderError as error:
            for field in ("input_tokens", "output_tokens"):
                value = getattr(error, field, None)
                if value is not None:
                    setattr(response.generation, field, value)
            self._failure(response, "general_generation_failed", None)
        response.generation.duration_ms = round((perf_counter() - started) * 1000, 3)
        response.timings_ms["general"] = response.generation.duration_ms
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
            return self._failure(response, "catalog_unavailable", "No se pudo consultar el catálogo de fuentes.")
        response.answer = catalog_answer(
            catalog, [item.dataset_id for item in routing.dataset_bindings], routing.language
        )
        response.limitations.append(
            "Inventario consultado en PostgreSQL durante esta respuesta; no se leyeron registros de los datasets."
            if routing.language == "es" else
            "Inventory read from PostgreSQL for this answer; no dataset records were read."
        )
        response.timings_ms["catalog"] = round((perf_counter() - started) * 1000, 3)
        return response

    @staticmethod
    def _failure(response, code, message):
        response.status = "failed"
        response.answer = (
            "No pude generar una respuesta válida. Podés volver a intentarlo."
            if response.language == "es" else
            "I could not generate a valid answer. You can try again."
        )
        response.generation.status = "failed"
        response.generation.error = ExecutionError(
            code=code, message=message or "El modelo no completó una respuesta válida."
        )
        response.errors = [response.generation.error]
        return response
