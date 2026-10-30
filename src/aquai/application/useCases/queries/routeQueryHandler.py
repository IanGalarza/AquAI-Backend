"""Una interpretación LLM; referencias, restricciones y reescritura en Python."""

import asyncio
import logging
import re
from time import perf_counter
from uuid import UUID

from aquai.application.answering.context import response_language
from aquai.application.contracts.routing import DataMention, DatasetBinding, RoutingResult
from aquai.application.conversations.context import contextual_question, routing_context
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.routing import RoutingProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import contains_phrase, normalize_name


_logger = logging.getLogger(__name__)


def _sheet(dataset_name: str) -> str:
    parts = [part.strip() for part in dataset_name.split(" / ")]
    return parts[1] if len(parts) >= 3 else dataset_name


def catalog_names(catalog) -> list[str]:
    """Nombres reales para que el Router reconozca fuentes; no describe contenidos."""
    return [
        source["name"] + (": " + "; ".join(dict.fromkeys(_sheet(item["name"]) for item in source["datasets"]))
                          if source["datasets"] else "")
        for source in catalog
    ][:60]


class RouteQueryHandler:
    def __init__(self, provider: RoutingProvider, sources: SourceQueries):
        self._provider = provider
        self._sources = sources

    async def handle(self, state):
        started = perf_counter()
        try:
            catalog = await asyncio.to_thread(self._sources.catalog_overview)
        except Exception as error:  # El Router sigue funcionando sin nombres del catálogo.
            _logger.warning("No se pudo leer el catálogo para el Router: %s", type(error).__name__)
            catalog = []
        context, digest, size, omitted = routing_context(
            state.question, state.conversation_history, self._provider.context_budget_bytes,
            catalog_names(catalog),
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
            result.clarification_question = decision.clarification_question
            provided = {item.message_id: item for item in state.conversation_history}
            allowed = {item["message_id"] for item in context["history"]}
            if (
                len(decision.history_message_ids) != len(set(decision.history_message_ids))
                or any(str(key) not in allowed for key in decision.history_message_ids)
            ):
                raise LLMProviderError("El Router citó mensajes ajenos al contexto recibido.")
            messages = self._with_context([provided[key] for key in decision.history_message_ids], provided)
            visible = [item for item in state.conversation_history if str(item.message_id) in allowed]
            last = next((item for item in reversed(visible) if item.role == "assistant"), None)
            route = decision.route
            if route == "needs_clarification":
                if last is not None and last.clarification:
                    # Una aclaración ya respondida no se repite: se continúa con lo acumulado
                    # y el Planner verifica si alcanza contra el catálogo real.
                    messages = self._with_context([last, *messages], provided)
                    route = "data"
                    result.warnings.append("Se continuó la solicitud pendiente con la aclaración recibida.")
                elif not decision.context_required and any(
                        item.kind == "variable" for item in decision.data_mentions):
                    # Una consulta autocontenida sobre una variable no requiere conocer la fuente.
                    route = "data"
            if decision.context_required and not messages and route != "needs_clarification":
                if last is not None and (last.analysis is not None or last.clarification):
                    # Sin una cita explícita, el seguimiento refiere al último resultado o aclaración.
                    messages = self._with_context([last], provided)
                elif route == "data" and not decision.data_mentions:
                    return self._clarify(result, "¿A qué resultado o consulta anterior te referís?")
            result.history_message_ids = [item.message_id for item in messages]
            result.context_used = bool(messages)
            result.data_mentions = self._literal_mentions(decision.data_mentions, state.question, messages, result)
            if messages and not any(item.kind == "variable" for item in result.data_mentions):
                # Un seguimiento sin variable nueva conserva la ya interpretada; los valores no
                # se heredan porque la nueva pregunta o la aclaración pueden reemplazarlos.
                result.data_mentions += [DataMention(text=text, kind="variable") for text in dict.fromkeys(
                    text for item in messages for text in item.variables)]
            if route == "needs_clarification":
                result.route = route
                result.request_history = self._requests(messages, state.question)
                question = decision.clarification_question or "¿Podés precisar qué variable, sitio o período querés consultar?"
                return self._clarify(result, question, from_model=decision.language == result.language)
            explicit = self._explicit_mentions(decision.source_mentions, state.question, result.data_mentions, result)
            inferred, aliases = await self._catalog_references(state.question, result.data_mentions, catalog, explicit)
            if inferred:
                result.warnings.append("Se reconoció como fuente explícita: " + "; ".join(f"«{name}»" for name in inferred) + ".")
                explicit += inferred
            if route == "general" and explicit:
                route = "catalog"  # Una pregunta sobre una fuente nombrada se responde con el catálogo.
            result.route = route
            if route == "historical":
                analytical = [item for item in messages if item.analysis is not None]
                if (
                    len(analytical) != 1 or analytical[0].analysis is None
                    or not analytical[0].text.strip()
                ):
                    return self._clarify(result, "¿Qué resultado anterior querés recuperar?")
                result.history_message_ids = [analytical[0].message_id]
                return result
            if route == "general":
                # El conocimiento general no interpreta resultados científicos anteriores.
                result.history_message_ids = [item.message_id for item in messages if not item.scientific]
                return result
            # En una consulta de inventario, un nombre parcial («ABSA») refiere a una fuente.
            bindings, resolved = self._resolve_sources(explicit, state.question, catalog,
                                                       partial=route == "catalog")
            # El texto literal que nombró la fuente («ABSA») también viaja al Planner.
            result.source_mentions = list(dict.fromkeys([*resolved, *(aliases if resolved else [])]))
            # Un nombre de fuente identificado nunca es, además, un valor para filtrar registros.
            result.data_mentions = [item for item in result.data_mentions if not any(
                contains_phrase(name, item.text) or contains_phrase(item.text, name) for name in resolved)]
            if route == "catalog":
                result.dataset_bindings = list({item.dataset_id: item for item in bindings}.values())
                result.history_message_ids = []
                return result
            if any(item.analysis and not item.analysis.complete for item in messages):
                return self._clarify(
                    result, "El resultado anterior no se conservó completo. ¿Podés indicar de nuevo la variable, el sitio y el período?"
                )
            question, requests = contextual_question(state.question, messages)
            pending = any(item.clarification for item in messages)
            words = state.question.strip(" ¿?¡!.").split()
            if pending and not resolved and 0 < len(words) <= 6 and not any(
                    item.kind == "value" for item in result.data_mentions):
                result.tentative_values = [" ".join(words)]
            if len(question) > 6000 or len(requests) > 8:
                return self._clarify(
                    result, "¿Podés resumir la variable, filtros y fuente que querés conservar?"
                )
            result.contextualized_question = question
            result.request_history = requests
            result.question_contextualized = question != state.question
            if not resolved:
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
    def _with_context(messages, provided):
        """Una aclaración arrastra los mensajes que citó, si siguen en el historial.

        Citar una pregunta del usuario equivale a citar ese intercambio: incluye su respuesta.
        """
        order = list(provided)
        selected = {}
        for message in messages:
            for key in message.context_message_ids:
                if key in provided:
                    selected.setdefault(key, provided[key])
            selected.setdefault(message.message_id, message)
            index = order.index(message.message_id)
            if message.role == "user" and index + 1 < len(order) and provided[order[index + 1]].role == "assistant":
                reply = provided[order[index + 1]]
                selected.setdefault(reply.message_id, reply)
                for key in reply.context_message_ids:
                    if key in provided:
                        selected.setdefault(key, provided[key])
        return sorted(selected.values(), key=lambda item: list(provided).index(item.message_id))

    @staticmethod
    def _requests(messages, question):
        requests = [
            request for message in messages
            for request in (message.pending_requests if message.clarification else
                            message.analysis.request_history if message.analysis else [])
        ]
        return [*dict.fromkeys(item for item in requests if item != question), question][-8:]

    @staticmethod
    def _literal_mentions(mentions, question, messages, result):
        """Sólo conserva fragmentos literales de la pregunta o de sus solicitudes citadas."""
        texts = [question, *RouteQueryHandler._requests(messages, question)]
        kept = {}
        for item in mentions:
            text = item.text.strip()
            if text and any(text.casefold() in value.casefold() for value in texts):
                kept.setdefault((text.casefold(), item.kind), item.model_copy(update={"text": text}))
            else:
                result.warnings.append("Se descartó una mención del Router que no aparece en la conversación.")
        return list(kept.values())

    @staticmethod
    def _explicit_mentions(mentions, question, data_mentions, result):
        values = {value.text.strip().casefold() for value in data_mentions if value.kind == "value"}
        explicit = []
        for mention in dict.fromkeys(item.strip() for item in mentions):
            if not mention or mention.casefold() not in question.casefold():
                # Un nombre ajeno a la pregunta no se convierte en una selección nueva.
                result.warnings.append("Se descartó una mención de fuente que no aparece en la pregunta.")
                continue
            if mention.casefold() not in values or RouteQueryHandler._has_source_cue(mention, question):
                explicit.append(mention)
        return explicit

    async def _catalog_references(self, question, mentions, catalog, explicit):
        """Fuentes nombradas en la pregunta aunque el modelo no las haya marcado como tales."""
        known = {normalize_name(item) for item in explicit}
        found, aliases = [], []
        for source in catalog:
            name = normalize_name(source["name"])
            # Nombres de una sola palabra son demasiado genéricos para reconocerlos sin una señal.
            if len(name.split()) >= 2 and name not in known and contains_phrase(question, name):
                found.append(source["name"])
        # Un «valor» ausente de los registros que aparece en el nombre de una única fuente la nombra.
        for item in mentions:
            if item.kind != "value" or any(contains_phrase(name, item.text) for name in [*explicit, *found]):
                continue
            owners = {source["source_id"]: source for source in catalog
                      if contains_phrase(normalize_name(source["name"]), item.text)}
            if len(owners) == 1 and not await self._stored_anywhere(item.text, catalog):
                found.append(next(iter(owners.values()))["name"])
                aliases.append(item.text)
        return found, aliases

    async def _stored_anywhere(self, value, catalog):
        ids = [dataset["dataset_id"] for source in catalog for dataset in source["datasets"] if dataset["available"]]
        try:
            datasets = await asyncio.to_thread(self._sources.verified_datasets, ids)
            found = await asyncio.to_thread(self._sources.find_text_values, datasets, [value])
        except Exception:
            return True  # Sin poder verificarlo, se conserva como valor y no se reinterpreta.
        return any(isinstance(row["value"], str) and contains_phrase(row["value"], value)
                   for matches in found.values() for row in matches.get(value, []))

    @staticmethod
    def _has_source_cue(mention, question):
        if re.search(r"\.(?:csv|xlsx)\b", mention, re.IGNORECASE):
            return True
        for match in re.finditer(re.escape(mention), question, re.IGNORECASE):
            prefix = question[max(0, match.start() - 100):match.start()]
            if re.search(r"\b(?:archivo|fuente|dataset|conjunto de datos|hoja|tabla|planilla|file|source|workbook|sheet|table)\s+"
                         r"(?:(?:el|la|los|las|de|del|llamad[oa]|denominad[oa]|named|called)\s+)*[«\"“']?\s*$",
                         prefix, re.IGNORECASE):
                return True
        return False

    def _resolve_sources(self, mentions, question, catalog, *, partial=False):
        """Identidad comprobada contra nombres reales; un tema o un sitio no es una fuente."""
        bindings, resolved = [], []
        for mention in dict.fromkeys(mentions):
            target = normalize_name(mention)
            if not target:
                continue
            strict = partial or self._has_source_cue(mention, question)
            try:
                identifier = UUID(mention.strip())
            except ValueError:
                identifier = None
            pairs = [(source, dataset) for source in catalog for dataset in source["datasets"]]
            if identifier is not None:
                matches = [(s, d) for s, d in pairs if identifier in {s["source_id"], d["dataset_id"]}]
            else:
                matches = [(s, d) for s, d in pairs if target in {
                    normalize_name(s["name"]), normalize_name(s["original_filename"])}]
                matches = matches or [(s, d) for s, d in pairs if target in {
                    normalize_name(d["name"]), normalize_name(_sheet(d["name"]))}]
                if not matches and strict:
                    matches = [(s, d) for s, d in pairs if contains_phrase(s["name"], mention)
                               or contains_phrase(d["name"], mention)]
            if not matches:
                if strict or identifier is not None:
                    names = "; ".join(source["name"] for source in catalog[:8])
                    raise ValueError(
                        f"No encontré una fuente llamada «{mention}» entre las cargadas"
                        + (f" ({names})" if names else "") + ". ¿Cuál querés usar?"
                    )
                continue  # Sin identidad comprobada, la mención sigue como texto para recuperar.
            sources = {source["source_id"] for source, _ in matches}
            if len(sources) != 1:
                names = "; ".join(sorted({source["name"] for source, _ in matches})[:8])
                raise ValueError(f"«{mention}» coincide con varias fuentes: {names}. ¿Cuál querés usar?")
            available = [(s, d) for s, d in matches if d["available"]]
            if not available:
                raise ValueError(
                    f"La fuente «{matches[0][0]['name']}» está inactiva o todavía no se puede consultar. "
                    "¿Querés usar otra fuente?"
                )
            bindings.extend(
                DatasetBinding(dataset_id=d["dataset_id"], source_id=s["source_id"]) for s, d in available
            )
            resolved.append(mention)
        return bindings, resolved

    @staticmethod
    def _clarify(result, question, *, from_model=False):
        result.route = "needs_clarification"
        result.clarification_question = (
            question if result.language == "es" or from_model else
            "Could you specify the previous question, variable, period, or source you mean?"
        )
        result.dataset_bindings = []
        return result
