import asyncio
import logging
import re
from time import perf_counter
from uuid import UUID

from aquai.application.answering.context import response_language
from aquai.application.contracts.routing import (
    DataMention,
    DatasetBinding,
    RoutingResult,
)
from aquai.application.conversations.context import contextual_question, routing_context
from aquai.application.errors import LLMProviderError
from aquai.application.interfaces.routing import RoutingProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import contains_phrase, normalize_name, normalize_text
from aquai.application.Rules.vocabulary import (
    SOURCE_CUE_WINDOW,
    SOURCE_CUE_WORDS,
    SOURCE_NAME_CONNECTORS,
    UPLOADED_FILE_EXTENSION,
)

_logger = logging.getLogger(__name__)
SOURCE_CUE = re.compile(
    rf"\b(?:{'|'.join(SOURCE_CUE_WORDS)})\s+"
    rf"(?:(?:{'|'.join(SOURCE_NAME_CONNECTORS)})\s+)*[«\"“']?\s*$",
    re.IGNORECASE,
)


def _sheet(dataset_name: str) -> str:
    parts = [part.strip() for part in dataset_name.split(" / ")]
    return parts[1] if len(parts) >= 3 else dataset_name


def _catalog_name(source) -> str:
    if not source["datasets"]:
        return source["name"]

    sheets = dict.fromkeys(_sheet(item["name"]) for item in source["datasets"])
    return source["name"] + ": " + "; ".join(sheets)


def catalog_names(catalog) -> list[str]:
    """Nombres reales para que el Router reconozca fuentes; no describe contenidos."""
    return [_catalog_name(source) for source in catalog][:60]


class RouteQueryService:
    def __init__(self, provider: RoutingProvider, sources: SourceQueries):
        self._provider = provider
        self._sources = sources

    async def route(self, state):
        started = perf_counter()
        catalog = await self._read_catalog()
        context, digest, size, omitted = routing_context(
            state.question,
            state.conversation_history,
            self._provider.context_budget_bytes,
            catalog_names(catalog),
        )
        result = self._uninterpreted_result(state, context, digest, size, omitted)

        try:
            return await self._interpret(state, context, catalog, result)
        except ValueError as error:
            return self._clarify(result, str(error))
        except LLMProviderError as error:
            self._record_failure(result, error)
        finally:
            self._log_route(result, started)

        return result

    async def _read_catalog(self):
        try:
            return await asyncio.to_thread(self._sources.catalog_overview)
        except Exception as error:  # El Router sigue funcionando sin nombres del catálogo.
            _logger.warning("No se pudo leer el catálogo para el Router: %s", type(error).__name__)
            return []

    def _uninterpreted_result(self, state, context, digest, size, omitted):
        return RoutingResult(
            route="failed",
            language=response_language(state.question, state.response_language),
            original_question=state.question,
            contextualized_question=state.question,
            request_history=[state.question],
            reason="No se pudo interpretar la consulta.",
            model=self._provider.model,
            context_sha256=digest,
            context_size_bytes=size,
            history_messages_provided=len(context["history"]),
            omitted_history_ids=omitted,
        )

    async def _interpret(self, state, context, catalog, result):
        reply = await self._provider.interpret(context)
        decision = reply.decision
        self._apply_reply(result, reply, state.response_language)

        provided = {item.message_id: item for item in state.conversation_history}
        allowed = {item["message_id"] for item in context["history"]}
        self._check_cited_messages(decision.history_message_ids, allowed)

        cited = [provided[key] for key in decision.history_message_ids]
        messages = self._with_context(cited, provided)
        last = self._last_assistant_message(state.conversation_history, allowed)
        route, messages = self._clarification_route(decision, last, messages, provided, result)

        asked = [
            item for item in decision.data_mentions if self._literal(item.text, state.question)
        ]
        if route == "historical" and self._changes(
            asked, decision.source_mentions, state.question, messages
        ):
            route = "data"  # Otro sitio, variable, fuente o cantidad cambia el resultado: no es repetir el anterior.

        if self._needs_implicit_reference(decision, route, asked, messages):
            if last is not None and (last.analysis is not None or last.clarification):
                # Sin una cita explícita, el seguimiento refiere al último resultado o aclaración.
                messages = self._with_context([last], provided)
            elif decision.context_required and route == "data" and not decision.data_mentions:
                return self._clarify(result, "¿A qué resultado o consulta anterior te referís?")

        self._record_mentions(result, decision, state.question, messages)
        if route == "needs_clarification":
            return self._model_clarification(result, decision, route, messages, state.question)

        return await self._resolve_route(state, catalog, result, decision, route, messages)

    async def _resolve_route(self, state, catalog, result, decision, route, messages):
        explicit, aliases = await self._named_sources(decision, state.question, catalog, result)
        if route == "general" and explicit:
            route = "catalog"  # Una pregunta sobre una fuente nombrada se responde con el catálogo.
        result.route = route

        if route == "historical":
            return self._historical_result(result, messages)
        if route == "general":
            return self._general_result(result, messages)

        # En una consulta de inventario, un nombre parcial («ABSA») refiere a una fuente.
        bindings, resolved = self._resolve_sources(
            explicit, state.question, catalog, partial=route == "catalog"
        )
        self._record_sources(result, resolved, aliases)
        if route == "catalog":
            return self._catalog_result(result, bindings)

        return self._data_result(state, result, messages, bindings, resolved)

    @staticmethod
    def _apply_reply(result, reply, preferred_language):
        decision = reply.decision
        result.model = reply.model
        result.input_tokens = reply.input_tokens
        result.output_tokens = reply.output_tokens
        result.language = preferred_language or decision.language
        result.reason = decision.reason
        result.clarification_question = decision.clarification_question

    @staticmethod
    def _check_cited_messages(message_ids, allowed):
        repeated = len(message_ids) != len(set(message_ids))
        if repeated or any(str(key) not in allowed for key in message_ids):
            raise LLMProviderError("El Router citó mensajes ajenos al contexto recibido.")

    @staticmethod
    def _last_assistant_message(history, allowed):
        visible = [item for item in history if str(item.message_id) in allowed]
        return next((item for item in reversed(visible) if item.role == "assistant"), None)

    @staticmethod
    def _clarification_route(decision, last, messages, provided, result):
        if decision.route != "needs_clarification":
            return decision.route, messages

        if last is not None and last.clarification:
            # Una aclaración ya respondida no se repite: se continúa con lo acumulado
            # y el Planner verifica si alcanza contra el catálogo real.
            messages = RouteQueryService._with_context([last, *messages], provided)
            result.warnings.append("Se continuó la solicitud pendiente con la aclaración recibida.")
            return "data", messages

        mentions_variable = any(item.kind == "variable" for item in decision.data_mentions)
        if not decision.context_required and mentions_variable:
            # Una consulta autocontenida sobre una variable no requiere conocer la fuente.
            return "data", messages

        return decision.route, messages

    @staticmethod
    def _needs_implicit_reference(decision, route, asked, messages):
        # Sin variable, valor, fecha ni fuente propios no hay nada que recuperar («Ahora mostrame todos»):
        # la consulta sólo puede continuar el último resultado, aunque el modelo no lo haya marcado.
        bare = (
            route == "data"
            and not decision.source_mentions
            and all(item.kind == "limit" for item in asked)
        )
        return (
            (decision.context_required or bare) and not messages and route != "needs_clarification"
        )

    @staticmethod
    def _record_mentions(result, decision, question, messages):
        result.history_message_ids = [item.message_id for item in messages]
        result.context_used = bool(messages)
        result.data_mentions = RouteQueryService._literal_mentions(
            decision.data_mentions, question, messages, result
        )

        if messages and not any(item.kind == "variable" for item in result.data_mentions):
            # Un seguimiento sin variable nueva conserva la ya interpretada; los valores no
            # se heredan porque la nueva pregunta o la aclaración pueden reemplazarlos.
            variables = dict.fromkeys(text for item in messages for text in item.variables)
            result.data_mentions += [DataMention(text=text, kind="variable") for text in variables]

    @staticmethod
    def _model_clarification(result, decision, route, messages, question):
        result.route = route
        result.request_history = RouteQueryService._requests(messages, question)
        clarification = (
            decision.clarification_question
            or "¿Podés precisar qué variable, sitio o período querés consultar?"
        )
        return RouteQueryService._clarify(
            result, clarification, from_model=decision.language == result.language
        )

    async def _named_sources(self, decision, question, catalog, result):
        explicit = self._explicit_mentions(
            decision.source_mentions, question, result.data_mentions, result
        )
        inferred, aliases = await self._catalog_references(
            question, result.data_mentions, catalog, explicit
        )
        if inferred:
            result.warnings.append(
                "Se reconoció como fuente explícita: "
                + "; ".join(f"«{name}»" for name in inferred)
                + "."
            )
            explicit += inferred
        return explicit, aliases

    @staticmethod
    def _historical_result(result, messages):
        analytical = [item for item in messages if item.analysis is not None]
        recoverable = len(analytical) == 1 and analytical[0].text.strip()
        if not recoverable:
            return RouteQueryService._clarify(result, "¿Qué resultado anterior querés recuperar?")

        result.history_message_ids = [analytical[0].message_id]
        return result

    @staticmethod
    def _general_result(result, messages):
        # El conocimiento general no interpreta resultados científicos anteriores.
        result.history_message_ids = [item.message_id for item in messages if not item.scientific]
        return result

    @staticmethod
    def _record_sources(result, resolved, aliases):
        # El texto literal que nombró la fuente («ABSA») también viaja al Planner.
        named = [*resolved, *aliases] if resolved else []
        result.source_mentions = list(dict.fromkeys(named))

        # Un nombre de fuente identificado, o el alias con que el usuario la nombró («la base FQ»),
        # nunca es, además, un valor para filtrar registros.
        result.data_mentions = [
            item
            for item in result.data_mentions
            if not RouteQueryService._names_any(item.text, result.source_mentions)
        ]

    @staticmethod
    def _names_any(text, names):
        return any(contains_phrase(name, text) or contains_phrase(text, name) for name in names)

    @staticmethod
    def _catalog_result(result, bindings):
        result.dataset_bindings = list(RouteQueryService._by_dataset(bindings).values())
        result.history_message_ids = []
        return result

    def _data_result(self, state, result, messages, bindings, resolved):
        if any(item.analysis and not item.analysis.complete for item in messages):
            return self._clarify(
                result,
                "El resultado anterior no se conservó completo. "
                "¿Podés indicar de nuevo la variable, el sitio y el período?",
            )

        question, requests = contextual_question(state.question, messages)
        self._record_tentative_value(result, state.question, messages, resolved)
        if len(question) > 6000 or len(requests) > 8:
            return self._clarify(
                result, "¿Podés resumir la variable, filtros y fuente que querés conservar?"
            )

        result.contextualized_question = question
        result.request_history = requests
        result.question_contextualized = question != state.question

        if not resolved:
            bindings = self._historical_bindings(messages)
        unique = self._by_dataset(bindings)
        result.dataset_bindings = list(unique.values())
        if len(unique) > state.candidate_limit:
            return self._clarify(
                result, "La selección abarca varios datasets. ¿Cuáles necesitás analizar?"
            )

        return result

    @staticmethod
    def _record_tentative_value(result, question, messages, resolved):
        answers_clarification = any(item.clarification for item in messages)
        words = question.strip(" ¿?¡!.").split()
        short_reply = 0 < len(words) <= 6
        names_value = any(item.kind == "value" for item in result.data_mentions)
        if answers_clarification and not resolved and short_reply and not names_value:
            result.tentative_values = [" ".join(words)]

    @staticmethod
    def _historical_bindings(messages):
        bindings = []
        for message in messages:
            if not message.analysis:
                continue
            for source in message.analysis.sources:
                bindings.append(
                    DatasetBinding(
                        dataset_id=source["dataset_id"],
                        source_id=source["source_id"],
                        historical_version=source["version"],
                    )
                )
        return bindings

    @staticmethod
    def _by_dataset(bindings):
        return {item.dataset_id: item for item in bindings}

    @staticmethod
    def _record_failure(result, error):
        result.route = "failed"
        result.error = str(error)
        for field in ("input_tokens", "output_tokens"):
            value = getattr(error, field, None)
            if value is not None:
                setattr(result, field, value)

    @staticmethod
    def _log_route(result, started):
        result.duration_ms = round((perf_counter() - started) * 1000, 3)
        _logger.info(
            "Router route=%s context=%s contextualized=%s model=%s tokens=%s/%s ms=%s",
            result.route,
            result.context_used,
            result.question_contextualized,
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.duration_ms,
        )

    @staticmethod
    def _with_context(messages, provided):
        """Una aclaración arrastra los mensajes que citó, si siguen en el historial.

        Citar una pregunta del usuario equivale a citar ese intercambio: incluye su respuesta.
        """
        order = list(provided)
        selected = {}
        for message in messages:
            RouteQueryService._select_cited(message, provided, selected)
            selected.setdefault(message.message_id, message)

            reply = RouteQueryService._reply_to(message, order, provided)
            if reply is None:
                continue
            selected.setdefault(reply.message_id, reply)
            RouteQueryService._select_cited(reply, provided, selected)
        return sorted(selected.values(), key=lambda item: order.index(item.message_id))

    @staticmethod
    def _select_cited(message, provided, selected):
        for key in message.context_message_ids:
            if key in provided:
                selected.setdefault(key, provided[key])

    @staticmethod
    def _reply_to(message, order, provided):
        index = order.index(message.message_id)
        if message.role != "user" or index + 1 >= len(order):
            return None

        following = provided[order[index + 1]]
        return following if following.role == "assistant" else None

    @staticmethod
    def _requests(messages, question):
        requests = [
            request
            for message in messages
            for request in RouteQueryService._message_requests(message)
        ]
        return [*dict.fromkeys(item for item in requests if item != question), question][-8:]

    @staticmethod
    def _message_requests(message):
        if message.clarification:
            return message.pending_requests
        if message.analysis:
            return message.analysis.request_history
        return []

    @staticmethod
    def _pending_requests(messages):
        return [
            request
            for message in messages
            if message.clarification
            for request in message.pending_requests
        ]

    @staticmethod
    def _literal_mentions(mentions, question, messages, result):
        """Sólo conserva fragmentos literales de la pregunta o de sus solicitudes citadas.

        Una cantidad sólo vale si se pide ahora o en la solicitud que esperaba una aclaración:
        la de un resultado ya respondido la conserva su plan, y repetirla anularía un cambio («todos»)."""
        texts = [question, *RouteQueryService._requests(messages, question)]
        current = [question, *RouteQueryService._pending_requests(messages)]
        kept = {}
        for item in mentions:
            text = item.text.strip()
            searched = current if item.kind == "limit" else texts
            if not text or not any(RouteQueryService._literal(text, value) for value in searched):
                result.warnings.append(
                    "Se descartó una mención del Router que no aparece en la conversación."
                )
                continue

            kept.setdefault((text.casefold(), item.kind), item.model_copy(update={"text": text}))
        return list(kept.values())

    @staticmethod
    def _changes(mentions, sources, question, messages):
        """Menciones de la pregunta actual que no estaban en las solicitudes citadas («¿Y en Toma Agua?»)."""
        cited = RouteQueryService._requests(messages, question)[:-1]
        asked_sources = [text for text in sources if RouteQueryService._literal(text, question)]
        texts = [*(item.text for item in mentions), *asked_sources]
        return [
            text
            for text in texts
            if not any(RouteQueryService._literal(text, request) for request in cited)
        ]

    @staticmethod
    def _literal(mention, text):
        """La mención aparece en el texto aunque el modelo omita acentos o cambie mayúsculas y puntuación."""
        target = normalize_text(mention)
        return bool(target) and target in normalize_text(text)

    @staticmethod
    def _explicit_mentions(mentions, question, data_mentions, result):
        values = {value.text.strip().casefold() for value in data_mentions if value.kind == "value"}
        explicit = []
        for mention in dict.fromkeys(item.strip() for item in mentions):
            if not mention or mention.casefold() not in question.casefold():
                # Un nombre ajeno a la pregunta no se convierte en una selección nueva.
                result.warnings.append(
                    "Se descartó una mención de fuente que no aparece en la pregunta."
                )
                continue

            is_value = mention.casefold() in values
            if not is_value or RouteQueryService._has_source_cue(mention, question):
                explicit.append(mention)
        return explicit

    async def _catalog_references(self, question, mentions, catalog, explicit):
        """Fuentes nombradas en la pregunta aunque el modelo no las haya marcado como tales."""
        known = {normalize_name(item) for item in explicit}
        found = self._multiword_sources_in(question, catalog, known)
        aliases = []

        # Un «valor» ausente de los registros que aparece en el nombre de una única fuente, o de una de
        # sus hojas («la base FQ»), la nombra.
        for item in mentions:
            if item.kind != "value":
                continue
            if any(contains_phrase(name, item.text) for name in [*explicit, *found]):
                continue

            owners = self._value_owners(item.text, catalog)
            if len(owners) == 1 and not await self._stored_anywhere(item.text, catalog):
                found.append(next(iter(owners.values()))["name"])
                aliases.append(item.text)
        return found, aliases

    @staticmethod
    def _multiword_sources_in(question, catalog, known):
        found = []
        for source in catalog:
            name = normalize_name(source["name"])
            # Nombres de una sola palabra son demasiado genéricos para reconocerlos sin una señal.
            if len(name.split()) >= 2 and name not in known and contains_phrase(question, name):
                found.append(source["name"])
        return found

    @staticmethod
    def _value_owners(value, catalog):
        owners = {}
        for source in catalog:
            names = [
                source["name"],
                *(_sheet(dataset["name"]) for dataset in source["datasets"]),
            ]
            if any(contains_phrase(normalize_name(name), value) for name in names):
                owners[source["source_id"]] = source
        return owners

    async def _stored_anywhere(self, value, catalog):
        dataset_ids = [
            dataset["dataset_id"]
            for source in catalog
            for dataset in source["datasets"]
            if dataset["available"]
        ]
        try:
            datasets = await asyncio.to_thread(self._sources.verified_datasets, dataset_ids)
            found = await asyncio.to_thread(self._sources.find_text_values, datasets, [value])
        except Exception:
            return True  # Sin poder verificarlo, se conserva como valor y no se reinterpreta.

        for matches in found.values():
            for row in matches.get(value, []):
                if isinstance(row["value"], str) and contains_phrase(row["value"], value):
                    return True
        return False

    @staticmethod
    def _has_source_cue(mention, question):
        if UPLOADED_FILE_EXTENSION.search(mention):
            return True

        for match in re.finditer(re.escape(mention), question, re.IGNORECASE):
            prefix = question[max(0, match.start() - SOURCE_CUE_WINDOW) : match.start()]

            if SOURCE_CUE.search(prefix):
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
            identifier = self._as_identifier(mention)
            matches = self._catalog_matches(mention, target, identifier, catalog, strict)
            if not matches:
                if strict or identifier is not None:
                    raise ValueError(self._unknown_source_message(mention, catalog))
                continue  # Sin identidad comprobada, la mención sigue como texto para recuperar.

            available = self._available_matches(mention, matches)
            bindings.extend(
                DatasetBinding(dataset_id=dataset["dataset_id"], source_id=source["source_id"])
                for source, dataset in available
            )
            resolved.append(mention)
        return bindings, resolved

    @staticmethod
    def _as_identifier(mention):
        try:
            return UUID(mention.strip())
        except ValueError:
            return None

    @staticmethod
    def _catalog_matches(mention, target, identifier, catalog, strict):
        pairs = [(source, dataset) for source in catalog for dataset in source["datasets"]]
        if identifier is not None:
            return [
                (source, dataset)
                for source, dataset in pairs
                if identifier in {source["source_id"], dataset["dataset_id"]}
            ]

        by_source = [
            (source, dataset)
            for source, dataset in pairs
            if target
            in {normalize_name(source["name"]), normalize_name(source["original_filename"])}
        ]
        if by_source:
            return by_source

        by_dataset = [
            (source, dataset)
            for source, dataset in pairs
            if target in {normalize_name(dataset["name"]), normalize_name(_sheet(dataset["name"]))}
        ]
        if by_dataset or not strict:
            return by_dataset

        return [
            (source, dataset)
            for source, dataset in pairs
            if contains_phrase(source["name"], mention) or contains_phrase(dataset["name"], mention)
        ]

    @staticmethod
    def _unknown_source_message(mention, catalog):
        names = "; ".join(source["name"] for source in catalog[:8])
        listed = f" ({names})" if names else ""
        return (
            f"No encontré una fuente llamada {mention} entre las cargadas"
            + listed
            + ". ¿Cuál querés usar?"
        )

    @staticmethod
    def _available_matches(mention, matches):
        sources = {source["source_id"] for source, _ in matches}
        if len(sources) != 1:
            names = "; ".join(sorted({source["name"] for source, _ in matches})[:8])
            raise ValueError(f"{mention} coincide con varias fuentes: {names}. ¿Cuál querés usar?")

        available = [(source, dataset) for source, dataset in matches if dataset["available"]]
        if not available:
            raise ValueError(
                f"La fuente {matches[0][0]['name']} está inactiva o todavía no se puede consultar. "
                "¿Querés usar otra fuente?"
            )
        return available

    @staticmethod
    def _clarify(result, question, *, from_model=False):
        keeps_question = result.language == "es" or from_model
        result.route = "needs_clarification"
        result.clarification_question = (
            question
            if keeps_question
            else "Could you specify the previous question, variable, period, or source you mean?"
        )
        result.dataset_bindings = []
        return result
