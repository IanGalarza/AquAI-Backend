"""Una interpretación LLM; referencias, restricciones y reescritura en Python."""

import asyncio
import logging
from time import perf_counter

from aquai.application.answering.context import response_language
from aquai.application.contracts.routing import DatasetBinding, RoutingResult
from aquai.application.conversations.context import contextual_question, routing_context
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.routing import RoutingProvider
from aquai.application.interfaces.sources import SourceQueries


_logger = logging.getLogger(__name__)


class RouteQueryHandler:
    def __init__(self, provider: RoutingProvider, sources: SourceQueries):
        self._provider = provider
        self._sources = sources

    async def handle(self, state):
        started = perf_counter()
        context, digest, size, omitted = routing_context(
            state.question, state.conversation_history, self._provider.context_budget_bytes
        )
        result = RoutingResult(
            route="failed", language=response_language(state.question, state.response_language),
            original_question=state.question, contextualized_question=state.question,
            request_history=[state.question], reason="No se pudo interpretar la consulta.",
            model=self._provider.model, context_sha256=digest, context_size_bytes=size,
            history_messages_provided=len(context["history"]),
            omitted_history_ids=omitted,
        )
        try:
            reply = await self._provider.interpret(context)
            decision = reply.decision
            result.model = reply.model
            result.input_tokens = reply.input_tokens
            result.output_tokens = reply.output_tokens
            result.language = state.response_language or decision.language
            result.reason = decision.reason
            result.route = decision.route
            result.clarification_question = decision.clarification_question
            provided = {item.message_id: item for item in state.conversation_history}
            allowed = {item["message_id"] for item in context["history"]}
            if (
                len(decision.history_message_ids) != len(set(decision.history_message_ids))
                or any(str(key) not in allowed for key in decision.history_message_ids)
            ):
                raise LLMProviderError("El Router citó mensajes ajenos al contexto recibido.")
            messages = [provided[key] for key in decision.history_message_ids]
            result.history_message_ids = list(decision.history_message_ids)
            result.context_used = bool(messages)
            if decision.context_required and not messages:
                return self._clarify(result, "¿A qué consulta o fuente anterior te referís?")
            if result.route == "needs_clarification":
                question = decision.clarification_question or "¿Podés precisar la consulta?"
                return self._clarify(
                    result, question, from_model=decision.language == result.language
                )
            if result.route == "historical":
                if (
                    len(messages) != 1 or messages[0].analysis is None
                    or not messages[0].text.strip()
                ):
                    return self._clarify(result, "¿Qué resultado anterior querés recuperar?")
                return result
            if result.route == "general":
                if decision.source_mentions or any(item.scientific for item in messages):
                    return self._clarify(
                        result,
                        "¿Querés consultar datos, recuperar un resultado o una explicación general?",
                    )
                return result
            if decision.context_required and (
                not messages or any(item.analysis is None for item in messages)
            ):
                return self._clarify(result, "¿Qué variable, período y fuente querés consultar?")
            if any(item.analysis and not item.analysis.complete for item in messages):
                return self._clarify(
                    result, "El contexto guardado es incompleto. ¿Podés precisar la consulta?"
                )
            question, requests = contextual_question(state.question, messages)
            if len(question) > 6000 or len(requests) > 8:
                return self._clarify(
                    result, "¿Podés resumir la variable, filtros y fuente que querés conservar?"
                )
            result.contextualized_question = question
            result.request_history = requests
            result.question_contextualized = question != state.question
            explicit = self._explicit_mentions(
                decision.source_mentions, state.question, messages
            )
            if explicit:
                bindings = await self._resolve_sources(explicit, state.question)
                result.source_mentions = explicit
            else:
                bindings = [
                    DatasetBinding(
                        dataset_id=source["dataset_id"], source_id=source["source_id"],
                        historical_version=source["version"],
                    )
                    for message in messages if message.analysis
                    for source in message.analysis.sources
                ]
            unique = {item.dataset_id: item for item in bindings}
            result.dataset_bindings = list(unique.values())
            if len(unique) > state.candidate_limit:
                return self._clarify(
                    result, "La selección abarca varios datasets. ¿Cuáles necesitás analizar?"
                )
        except ValueError as error:
            return self._clarify(result, str(error))
        except LLMProviderError as error:
            result.route = "failed"
            result.error = str(error)
            for field in ("input_tokens", "output_tokens"):
                value = getattr(error, field, None)
                if value is not None:
                    setattr(result, field, value)
        finally:
            result.duration_ms = round((perf_counter() - started) * 1000, 3)
            _logger.info(
                "Router route=%s context=%s contextualized=%s model=%s tokens=%s/%s ms=%s",
                result.route, result.context_used, result.question_contextualized,
                result.model, result.input_tokens, result.output_tokens, result.duration_ms,
            )
        return result

    @staticmethod
    def _explicit_mentions(mentions, question, messages):
        names = [
            name.casefold()
            for message in messages if message.analysis
            for source in message.analysis.sources
            for name in (
                source["dataset_name"], source["provenance"]["source_name"],
                source["provenance"]["original_filename"],
            )
        ]
        explicit = []
        for mention in dict.fromkeys(mentions):
            value = mention.strip().casefold()
            if not value:
                raise LLMProviderError("La mención de fuente está vacía.")
            if value in question.casefold():
                explicit.append(mention)
            elif not any(value in name for name in names):
                raise LLMProviderError(
                    "La mención no aparece en la pregunta ni en sus referencias históricas."
                )
            # Un nombre histórico verificado no se convierte en una selección nueva:
            # se conserva el conjunto de identidades de los mensajes ya autorizados.
        return explicit

    async def _resolve_sources(self, mentions, question):
        bindings = []
        for mention in dict.fromkeys(mentions):
            if not mention.strip() or mention.casefold() not in question.casefold():
                raise LLMProviderError("La mención de fuente no aparece en la pregunta original.")
            matches = await asyncio.to_thread(self._sources.matching_datasets, mention)
            exact = [item for item in matches if item.exact]
            names = [item for item in matches if item.name_match]
            matches = exact or names or matches
            sources = {item.source_id for item in matches}
            if not matches:
                raise ValueError(
                    f"No pude identificar la fuente «{mention}». ¿Podés indicar su nombre?"
                )
            if len(sources) != 1:
                names = sorted({item.source_name for item in matches})
                options = "; ".join(names[:8])
                raise ValueError(
                    f"«{mention}» coincide con varias fuentes: {options}. ¿Cuál querés usar?"
                )
            if any(not item.available for item in matches):
                raise ValueError(
                    f"La selección «{mention}» incluye datos inactivos o no consultables. "
                    "¿Podés indicar un dataset disponible?"
                )
            bindings.extend(
                DatasetBinding(dataset_id=item.dataset_id, source_id=item.source_id)
                for item in matches
            )
        return bindings

    @staticmethod
    def _clarify(result, question, *, from_model=False):
        result.route = "needs_clarification"
        result.clarification_question = (
            question if result.language == "es" or from_model else
            "Could you specify the previous question, variable, period, or source you mean?"
        )
        result.dataset_bindings = []
        return result
