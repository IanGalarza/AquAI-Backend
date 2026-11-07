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
from aquai.application.matching import (
    contains_any_phrase,
    contains_phrase,
    documented_reference,
    mentioned_count,
    named_documents,
    normalize_name,
    normalize_text,
    numeric_comparisons,
    qualified_period,
    qualified_variable,
    qualifies_variable,
    starts_with_phrase,
    variable_core,
    without_reference_request,
)
from aquai.application.planning.context import chart_requested, requested_statistics
from aquai.application.planning.followUp import requested_period, requested_time_grain
from aquai.application.Rules.evidence import (
    CITATION_QUESTION,
    DOCUMENT_ANAPHORA,
    GUIDELINE_CUES,
    LAST_RESULT,
    OBSERVED_COMPARISON,
    ORGANIZATION_NAMES,
    REFERENTIAL_QUESTION,
    REPEAT_REQUEST,
)
from aquai.application.Rules.vocabulary import (
    ADDITIVE_CUES,
    CATEGORY_CUES,
    CHART_WORDS,
    CONTINUATION_CUES,
    MEASURED_STATISTICS,
    MENTION_CONNECTORS,
    PLACE_WORDS,
    SOURCE_ALIAS_CUES,
    SOURCE_CUE_WINDOW,
    SOURCE_CUE_WORDS,
    SOURCE_NAME_CONNECTORS,
    TIME_GRAIN_CUES,
    UPLOADED_FILE_EXTENSION,
)

_logger = logging.getLogger(__name__)
_MAX_REFERENTIAL_WORDS = 10
_NUMERIC_KINDS = {"value", "limit", "threshold"}
_SCOPE_KINDS = {"value", "dimension", "threshold", "date"}
_CONSERVED_KINDS = {"variable", "value", "dimension", "date"}
_RECORD_SCOPE_KINDS = {"value", "date", "threshold"}
_CALCULATION_KINDS = {"variable", "value", "dimension", "threshold", "limit"}
_ELLIPSIS_WORDS = {*MENTION_CONNECTORS, *(word for cue in CONTINUATION_CUES for word in cue.split())}
_PLACE_NAMES = {normalize_text(word) for word in PLACE_WORDS}
_CHART_FILLERS = {*MENTION_CONNECTORS, "un", "una", "a", "an"}
_CHART_WORD = re.compile(rf"\b{CHART_WORDS}\b", re.IGNORECASE)
SOURCE_ALIAS = re.compile(
    rf"\b(?:{'|'.join(SOURCE_ALIAS_CUES)})\s+(?:(?:{'|'.join(SOURCE_NAME_CONNECTORS)})\s+)*"
    r"(?P<name>[^\W_][\w-]*)",
    re.IGNORECASE,
)
SOURCE_CUE = re.compile(
    rf"\b(?:{'|'.join(SOURCE_CUE_WORDS)})\s+"
    rf"(?:(?:{'|'.join(SOURCE_NAME_CONNECTORS)})\s+)*[«\"“']?\s*$",
    re.IGNORECASE,
)
_SOURCE_WORD = re.compile(rf"\b(?:{'|'.join(SOURCE_CUE_WORDS)})s?\b", re.IGNORECASE)


#
# Extrae el nombre de la hoja del nombre de un dataset.
#
def _sheet(dataset_name: str) -> str:
    parts = [part.strip() for part in dataset_name.split(" / ")]
    return parts[1] if len(parts) >= 3 else dataset_name


#
# Devuelve el nombre de una fuente con sus hojas para mostrarlo en el catálogo.
#
def _catalog_name(source) -> str:
    if not source["datasets"]:
        return source["name"]

    sheets = dict.fromkeys(_sheet(item["name"]) for item in source["datasets"])
    return source["name"] + ": " + "; ".join(sheets)


#
# Devuelve los nombres de las fuentes del catálogo (hasta 60) para que el Router las reconozca.
#
def catalog_names(catalog) -> list[str]:
    return [_catalog_name(source) for source in catalog][:60]


class RouteQueryService:
    #
    # Guarda el proveedor de enrutamiento y las consultas de fuentes que usa el servicio.
    #
    def __init__(self, provider: RoutingProvider, sources: SourceQueries):
        self._provider = provider
        self._sources = sources

    #
    # Decide la ruta de una consulta (datos, catálogo, histórica, general o aclaración).
    #
    async def route(self, state):
        started = perf_counter()
        catalog = await self._read_catalog()
        documents = await self._read_documents()
        context, digest, size, omitted = routing_context(
            state.question,
            state.conversation_history,
            self._provider.context_budget_bytes,
            catalog_names(catalog),
        )
        result = self._uninterpreted_result(state, context, digest, size, omitted)

        try:
            return await self._interpret(state, context, catalog, documents, result)
        except ValueError as error:
            return self._clarify(result, str(error))
        except LLMProviderError as error:
            self._record_failure(result, error)
        finally:
            self._log_route(result, started)

        return result

    #
    # Lee el catálogo de fuentes y devuelve una lista vacía si no está disponible.
    #
    async def _read_catalog(self):
        try:
            return await asyncio.to_thread(self._sources.catalog_overview)
        except Exception as error:
            _logger.warning("No se pudo leer el catálogo para el Router: %s", type(error).__name__)
            return []

    #
    # Lee los documentos consultables y devuelve una lista vacía si no están disponibles.
    #
    async def _read_documents(self):
        try:
            return await asyncio.to_thread(self._sources.document_overview)
        except Exception as error:
            _logger.warning("No se pudo leer los documentos: %s", type(error).__name__)
            return []

    #
    # Crea el resultado de enrutamiento inicial, fallido, antes de interpretar la consulta.
    #
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

    #
    # Interpreta la consulta con el modelo y la resuelve contra el historial y el contexto.
    #
    async def _interpret(self, state, context, catalog, documents, result):
        reply = await self._provider.interpret(context)
        decision = reply.decision
        self._apply_reply(result, reply, state.response_language)

        provided = {item.message_id: item for item in state.conversation_history}
        allowed = {item["message_id"] for item in context["history"]}
        self._check_cited_messages(decision.history_message_ids, allowed)

        cited = [provided[key] for key in decision.history_message_ids]
        messages = self._with_context(cited, provided)
        last = self._last_assistant_message(state.conversation_history, allowed)
        route, messages = self._clarification_route(
            decision, last, messages, provided, result, state.question
        )

        asked = [
            item for item in decision.data_mentions if self._literal(item.text, state.question)
        ]
        if route == "historical" and (
            self._changes(asked, decision.source_mentions, state.question, messages)
            or chart_requested(state.question)
        ):
            route = "data"

        route = self._documentary_route(route, asked, state.question, documents)
        anchor = self._anchor(state.conversation_history, allowed)
        followed = self._documented_reply(last, asked, state.question, route, decision)
        subject = self._period_subject(state.question, asked, messages, anchor, route, decision)
        if self._asks_citations(state.question, last):
            route = "document"
            result.asks_citations = True
            messages = self._with_context([last], provided)
        elif self._repeats_last(route, asked, decision, anchor, state.question, messages, last):
            route = "historical"
            messages = self._with_context([anchor], provided)
        elif followed is not None:
            route = "document"
            messages = self._with_context([followed], provided)
        elif subject is not None:
            route = "data"
            result.requested_period = requested_period(state.question)
            messages = self._with_context([subject], provided)
        elif anchor is not None and self._fragment(decision, route, asked, state.question):
            route = "data"
            messages = self._with_context([anchor], provided)
        elif self._refers_to_last(route, messages, state.question, last):
            messages = self._with_context([last], provided)
        elif self._needs_implicit_reference(decision, route, asked, messages):
            target = last if route == "document" and self._cites_evidence(last) else anchor
            if target is not None:
                messages = self._with_context([target], provided)
            elif decision.context_required and route == "data" and not decision.data_mentions:
                return self._clarify(result, "¿A qué resultado o consulta anterior te referís?")

        self._record_mentions(result, decision, without_reference_request(state.question), messages)
        if route == "needs_clarification":
            return self._model_clarification(result, decision, route, messages, state.question)

        if route == "document":
            return self._document_result(state, result, messages, documents)

        result = await self._resolve_route(state, catalog, result, decision, route, messages)
        if result.route == "data":
            self._record_reference_comparison(result, state.question, documents)

        return result

    #
    # Registra la referencia documental con que se pide comparar los datos y los documentos que la pregunta
    # nombra, para buscar ese valor antes de planificar.
    #
    @staticmethod
    def _record_reference_comparison(result, question, documents):
        reference = documented_reference(question)
        if reference is None:
            return

        result.reference_comparison = reference
        result.document_ids = [item["document_id"] for item in named_documents(question, documents)]
        result.data_mentions = [
            item
            for item in result.data_mentions
            if isinstance(item, str) or item.kind != "value" or not contains_phrase(reference, item.text)
        ]

    #
    # Corrige la ruta de una pregunta científica o que nombra un documento.
    #
    @staticmethod
    def _documentary_route(route, asked, question, documents):
        if route == "general" and any(item.kind == "variable" for item in asked):
            return "document"
        if route in {"general", "catalog", "data"} and not RouteQueryService._calculates(route, asked, question):
            return "document" if named_documents(question, documents) else route

        return route

    #
    # Indica si la pregunta pide calcular o comparar sobre los datos, aunque nombre un documento.
    #
    @staticmethod
    def _calculates(route, asked, question):
        measured = contains_any_phrase(question, MEASURED_STATISTICS) or bool(requested_statistics(question))
        compared = OBSERVED_COMPARISON.search(question) is not None and any(
            item.kind == "variable" for item in asked
        )

        return measured or chart_requested(question) or (route == "data" and compared)

    #
    # Devuelve la respuesta documental que continúa un fragmento («¿Y la turbiedad?»), si la hay.
    #
    @staticmethod
    def _documented_reply(last, asked, question, route, decision):
        documental = last is not None and last.documents and last.analysis is None
        fragment = RouteQueryService._elliptical(question, asked) and all(
            item.kind == "variable" for item in asked
        )
        measured = contains_any_phrase(question, MEASURED_STATISTICS)
        follows = route != "data" or decision.context_required

        return last if documental and fragment and follows and not measured else None

    #
    # Indica si la pregunta pide de dónde salió la última respuesta, que citó documentos.
    #
    @staticmethod
    def _asks_citations(question, last):
        return last is not None and bool(last.documents) and CITATION_QUESTION.search(question)

    #
    # Indica si se pide repetir un resultado sin nombrar cuál: es el último análisis.
    #
    @staticmethod
    def _repeats_last(route, asked, decision, anchor, question, messages, last):
        unnamed = not asked and not decision.source_mentions
        repeats = route == "historical" or REPEAT_REQUEST.search(question) is not None
        latest = anchor is not None and anchor.analysis is not None and anchor is last
        cited = any(item.analysis is not None for item in messages)
        names_last = LAST_RESULT.search(question) is not None

        return repeats and unnamed and latest and (names_last or not cited)

    #
    # Indica si una pregunta documental breve remite a la última respuesta («¿Eso es normal?»).
    #
    @staticmethod
    def _refers_to_last(route, messages, question, last):
        refers = REFERENTIAL_QUESTION.search(question) is not None
        brief = len(question.split()) <= _MAX_REFERENTIAL_WORDS

        return (
            route == "document"
            and not messages
            and refers
            and brief
            and RouteQueryService._cites_evidence(last)
        )

    #
    # Indica si un mensaje del asistente tiene resultados o documentos que el seguimiento retoma.
    #
    @staticmethod
    def _cites_evidence(message):
        return message is not None and (message.analysis is not None or bool(message.documents))

    #
    # Prepara la ruta documental: conserva el contexto citado y los documentos a los que se refiere.
    #
    def _document_result(self, state, result, messages, documents):
        result.route = "document"
        result.request_history = self._requests(messages, state.question)
        result.document_ids = self._referred_documents(state.question, messages, documents)

        return result

    #
    # Devuelve los documentos que la pregunta nombra o a los que remite con «ese paper».
    #
    @staticmethod
    def _referred_documents(question, messages, documents):
        named = [item["document_id"] for item in named_documents(question, documents)]
        if named or DOCUMENT_ANAPHORA.search(question) is None:
            return named

        cited = [item["document_id"] for message in messages for item in message.documents]
        return list(dict.fromkeys(UUID(str(item)) for item in cited if item))

    #
    # Resuelve la ruta decidida: fuentes nombradas, valores guardados y resultado de cada ruta.
    #
    async def _resolve_route(self, state, catalog, result, decision, route, messages):
        explicit, aliases = await self._named_sources(decision, state.question, catalog, result)
        explicit = await self._stored_values(explicit, state.question, catalog, result)
        if route == "general" and explicit:
            route = "catalog"
        result.route = route

        if route == "historical":
            return self._historical_result(result, messages, state.conversation_history)
        if route == "general":
            return self._general_result(result, messages)

        bindings, resolved = self._resolve_sources(
            explicit, state.question, catalog, partial=route == "catalog"
        )
        self._record_sources(result, resolved, aliases)
        if route == "catalog" and self._restricts_records(result):
            route = result.route = "data"
            result.warnings.append(
                "La pregunta acota registros por un valor, una fecha o un umbral: "
                "se consultaron los datos y no el inventario de fuentes."
            )
        if route == "catalog":
            return self._catalog_result(result, bindings)

        return self._data_result(state, result, messages, bindings, resolved)

    #
    # Reinterpreta como valores de los registros los nombres que no son fuentes del catálogo.
    #
    async def _stored_values(self, explicit, question, catalog, result):
        kept, found = [], []
        for mention in explicit:
            unknown = not self._catalog_matches(
                mention, normalize_name(mention), self._as_identifier(mention), catalog, True
            )
            if (
                unknown
                and not self._has_source_cue(mention, question)
                and await self._stored_anywhere(mention, catalog)
            ):
                found.append(DataMention(text=mention, kind="value"))
                continue
            kept.append(mention)

        if found:
            additive = contains_any_phrase(question, ADDITIVE_CUES)
            replaced = [
                item for item in result.conserved_mentions if item.kind == "value" and not additive
            ]
            result.data_mentions = [
                *(item for item in result.data_mentions if item not in replaced),
                *(item for item in found if item not in result.data_mentions),
            ]
            result.conserved_mentions = [
                item for item in result.conserved_mentions if item not in replaced
            ]
            result.warnings.append(
                "Se interpretó como valor de los registros, no como fuente: "
                + "; ".join(item.text for item in found)
                + "."
            )
        return kept

    #
    # Indica si las menciones acotan registros por un valor, una fecha o un umbral.
    #
    @staticmethod
    def _restricts_records(result):
        return any(item.kind in _RECORD_SCOPE_KINDS for item in result.data_mentions)

    #
    # Copia al resultado el modelo, los tokens, el idioma y el motivo de la respuesta del Router.
    #
    @staticmethod
    def _apply_reply(result, reply, preferred_language):
        decision = reply.decision
        result.model = reply.model
        result.input_tokens = reply.input_tokens
        result.output_tokens = reply.output_tokens
        result.language = preferred_language or decision.language
        result.reason = decision.reason
        result.clarification_question = decision.clarification_question

    #
    # Verifica que los mensajes citados existan en el contexto y no estén repetidos.
    #
    @staticmethod
    def _check_cited_messages(message_ids, allowed):
        repeated = len(message_ids) != len(set(message_ids))
        if repeated or any(str(key) not in allowed for key in message_ids):
            raise LLMProviderError("El Router citó mensajes ajenos al contexto recibido.")

    #
    # Devuelve el último mensaje del asistente visible en el contexto.
    #
    @staticmethod
    def _last_assistant_message(history, allowed):
        visible = [item for item in history if str(item.message_id) in allowed]
        return next((item for item in reversed(visible) if item.role == "assistant"), None)

    #
    # Decide la ruta cuando el modelo pidió aclaración, continuando una aclaración ya respondida.
    #
    @staticmethod
    def _clarification_route(decision, last, messages, provided, result, question):
        if decision.route != "needs_clarification":
            return decision.route, messages

        if last is not None and last.clarification:
            messages = RouteQueryService._with_context([last, *messages], provided)
            result.warnings.append("Se continuó la solicitud pendiente con la aclaración recibida.")
            return "data", messages

        if RouteQueryService._refers_to_last("document", [], question, last):
            result.warnings.append("La pregunta breve remite a la última respuesta, que tiene evidencia citable.")
            return "document", RouteQueryService._with_context([last], provided)

        mentions_variable = any(item.kind == "variable" for item in decision.data_mentions)
        if not decision.context_required and mentions_variable:
            return "data", messages

        return decision.route, messages

    #
    # Devuelve el último análisis verificado o aclaración visible del asistente.
    #
    @staticmethod
    def _anchor(history, allowed):
        replies = [
            item for item in history if item.role == "assistant" and str(item.message_id) in allowed
        ]
        return next(
            (item for item in reversed(replies) if item.analysis is not None or item.clarification),
            None,
        )

    #
    # Devuelve el análisis cuyo período se pregunta en un seguimiento sin condiciones propias.
    #
    @staticmethod
    def _period_subject(question, asked, messages, anchor, route, decision):
        own = [item for item in asked if item.kind in _CALCULATION_KINDS]
        about_source = (
            route == "catalog"
            or _SOURCE_WORD.search(question) is not None
            or any(RouteQueryService._literal(text, question) for text in decision.source_mentions)
        )
        if own or about_source or chart_requested(question) or requested_period(question) is None:
            return None
        cited = [item for item in messages if item.analysis is not None]
        if route == "document" and (any(item.documents for item in messages) or not cited):
            return None

        if len(cited) == 1:
            return cited[0]

        return anchor if anchor is not None and anchor.analysis is not None else None

    #
    # Indica si la consulta continúa el último resultado por no tener condiciones propias.
    #
    @staticmethod
    def _needs_implicit_reference(decision, route, asked, messages):
        bare = (
            route == "data"
            and not decision.source_mentions
            and all(item.kind == "limit" for item in asked)
        )
        return (
            (decision.context_required or bare) and not messages and route != "needs_clarification"
        )

    #
    # Indica si la pregunta es un fragmento que continúa el último análisis.
    #
    @staticmethod
    def _fragment(decision, route, asked, question):
        presentation = chart_requested(question) and all(
            RouteQueryService._presentation(item) for item in asked
        )
        names_source = any(
            RouteQueryService._literal(text, question) for text in decision.source_mentions
        )
        return (
            route in {"data", "needs_clarification"}
            and not names_source
            and (presentation or RouteQueryService._elliptical(question, asked))
        )

    #
    # Indica si la pregunta se compone sólo de menciones y palabras de enlace.
    #
    @staticmethod
    def _elliptical(question, asked):
        covered = {word for item in asked for word in normalize_text(item.text).split()}
        words = normalize_text(question).split()

        return bool(asked) and all(word in covered or word in _ELLIPSIS_WORDS for word in words)

    #
    # Registra el historial citado y las menciones propias y conservadas de la consulta.
    #
    @staticmethod
    def _record_mentions(result, decision, question, messages):
        result.history_message_ids = [item.message_id for item in messages]
        result.context_used = bool(messages)
        mentions = RouteQueryService._literal_mentions(
            decision.data_mentions, question, messages, result
        )
        mentions = RouteQueryService._comparison_mentions(
            mentions, [question, *RouteQueryService._pending_requests(messages)]
        )
        mentions = RouteQueryService._without_chart_phrases(mentions)
        mentions = RouteQueryService._without_reference_values(mentions)
        mentions = RouteQueryService._qualified_variables(
            RouteQueryService._separated_variables(mentions), question
        )
        mentions = RouteQueryService._asked_dimensions(mentions, question)
        own, conserved = RouteQueryService._conserved_scope(mentions, question, messages)
        result.data_mentions = [*own, *conserved]
        result.conserved_mentions = conserved

    #
    # Quita los valores que nombran la referencia con que se compara («OMS para uso recreativo», «el nivel
    # guía»): describen un documento, no registros que filtrar.
    #
    @staticmethod
    def _without_reference_values(mentions):
        return [
            item
            for item in mentions
            if not (
                item.kind == "value"
                and (ORGANIZATION_NAMES.search(item.text) or GUIDELINE_CUES.search(item.text))
            )
        ]

    #
    # Completa las variables y fechas con los calificativos que siguen en la pregunta.
    #
    @staticmethod
    def _qualified_variables(mentions, question):
        qualify = {"variable": qualified_variable, "date": qualified_period}
        qualified = [
            item.model_copy(update={"text": qualify[item.kind](item.text, question)})
            if item.kind in qualify
            else item
            for item in mentions
        ]
        variables = [item.text for item in qualified if item.kind == "variable"]
        return [
            item
            for item in qualified
            if not (item.kind == "value" and any(qualifies_variable(item.text, text) for text in variables))
        ]

    #
    # Agrega como dimensión el lugar que la pregunta consulta («qué sitios») si falta.
    #
    @staticmethod
    def _asked_dimensions(mentions, question):
        words = normalize_text(question).split()
        asked = [
            following
            for word, following in zip(words, words[1:])
            if word in CATEGORY_CUES and following in _PLACE_NAMES
        ]
        known = [item.text for item in mentions if item.kind in {"dimension", "value"}]
        added = [
            DataMention(text=word, kind="dimension")
            for word in dict.fromkeys(asked)
            if not any(contains_phrase(text, word, plural_insensitive=True) for text in known)
        ]
        return [*mentions, *added]

    #
    # Quita las menciones que sólo piden una presentación de gráfico.
    #
    @staticmethod
    def _without_chart_phrases(mentions):
        return [item for item in mentions if not RouteQueryService._presentation(item)]

    #
    # Indica si una mención pide una presentación (gráfico) y no un dato.
    #
    @staticmethod
    def _presentation(item):
        uncounted = item.kind == "limit" and mentioned_count(item.text) is None
        return RouteQueryService._names_chart(item.text) or (
            uncounted and _CHART_WORD.search(item.text) is not None
        )

    #
    # Indica si un texto sólo nombra un gráfico.
    #
    @staticmethod
    def _names_chart(text):
        remainder = _CHART_WORD.sub(" ", text)
        words = [word for word in normalize_text(remainder).split() if word not in _CHART_FILLERS]
        return remainder != text and not words

    #
    # Separa las menciones propias de las conservadas del análisis citado.
    #
    @staticmethod
    def _conserved_scope(mentions, question, messages):
        current = [question, *RouteQueryService._pending_requests(messages)]
        own = [
            item
            for item in mentions
            if any(RouteQueryService._literal(item.text, text) for text in current)
        ]
        if not messages:
            return own, []

        asked = {item.kind for item in own}
        continues = (
            "variable" not in asked
            or starts_with_phrase(question, CONTINUATION_CUES)
            or any(item.clarification for item in messages)
        )
        additive = contains_any_phrase(question, ADDITIVE_CUES)
        known = {(normalize_text(item.text), item.kind) for item in own}
        conserved = {}
        for item in (item for message in messages for item in message.scope):
            key = (normalize_text(item.text), item.kind)
            if key not in known and RouteQueryService._conserves(item.kind, asked, continues, additive):
                conserved.setdefault(key, item.model_copy())
        return own, list(conserved.values())

    #
    # Indica si una mención de cierto tipo se conserva en un seguimiento.
    #
    @staticmethod
    def _conserves(kind, asked, continues, additive):
        if kind not in _CONSERVED_KINDS:
            return False
        if kind == "variable":
            return kind not in asked

        replaced = kind in asked and not (additive and kind == "value")
        return continues and not replaced

    #
    # Devuelve la aclaración del modelo, o una genérica, como resultado de enrutamiento.
    #
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

    #
    # Reúne las fuentes nombradas explícitamente y las reconocidas en el catálogo.
    #
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

    #
    # Prepara el resultado histórico si hay un único análisis anterior recuperable.
    #
    @staticmethod
    def _historical_result(result, messages, history):
        provided = {item.message_id: item for item in history}
        originals = {}
        for item in messages:
            if item.analysis is not None:
                original = provided.get(item.replay_of, item)
                originals.setdefault(original.message_id, original)
        analytical = list(originals.values())
        recoverable = len(analytical) == 1 and analytical[0].text.strip()
        if not recoverable:
            return RouteQueryService._clarify(result, "¿Qué resultado anterior querés recuperar?")

        result.history_message_ids = [analytical[0].message_id]
        return result

    #
    # Prepara el resultado de la ruta general sin mensajes científicos.
    #
    @staticmethod
    def _general_result(result, messages):
        result.history_message_ids = [item.message_id for item in messages if not item.scientific]
        return result

    #
    # Registra las fuentes resueltas y sus alias, y quita de las menciones las que las nombran.
    #
    @staticmethod
    def _record_sources(result, resolved, aliases):
        named = [*resolved, *aliases] if resolved else []
        result.source_mentions = list(dict.fromkeys(named))

        result.data_mentions = [
            item
            for item in result.data_mentions
            if not RouteQueryService._names_any(item.text, result.source_mentions)
        ]

    #
    # Indica si el texto coincide con alguno de los nombres.
    #
    @staticmethod
    def _names_any(text, names):
        return any(contains_phrase(name, text) or contains_phrase(text, name) for name in names)

    #
    # Prepara el resultado de la ruta de catálogo con sus datasets vinculados.
    #
    @staticmethod
    def _catalog_result(result, bindings):
        result.dataset_bindings = list(RouteQueryService._by_dataset(bindings).values())
        result.history_message_ids = []
        return result

    #
    # Prepara el resultado de la ruta de datos: contextualiza la pregunta y vincula datasets.
    #
    def _data_result(self, state, result, messages, bindings, resolved):
        if any(item.analysis and not item.analysis.complete for item in messages):
            return self._clarify(
                result,
                "El resultado anterior no se conservó completo. "
                "¿Podés indicar de nuevo la variable, el sitio y el período?",
            )

        if self._changes_topic(result, state.question, messages, resolved):
            messages = self._without_context(result)

        question, requests = contextual_question(
            without_reference_request(state.question),
            messages,
            redefined=self._redefines(result, state.question),
        )
        self._record_tentative_value(result, state.question, messages, resolved)
        if len(question) > 6000 or len(requests) > 8:
            return self._clarify(
                result, "¿Podés resumir la variable, filtros y fuente que querés conservar?"
            )

        result.contextualized_question = question
        result.request_history = requests
        result.question_contextualized = question != state.question
        if self._lacks_variable(result, [state.question, *self._pending_requests(messages)]):
            return self._clarify(result, "¿Qué variable querés analizar?")

        if not resolved:
            bindings = self._historical_bindings(messages)
            result.reselects_datasets = bool(bindings) and self._reselects(result, messages)
            result.inherits_named_sources = bool(bindings) and any(
                item.named_sources for item in messages
            )
        unique = self._by_dataset(bindings)
        result.dataset_bindings = list(unique.values())
        if len(unique) > state.candidate_limit:
            return self._clarify(
                result, "La selección abarca varios datasets. ¿Cuáles necesitás analizar?"
            )

        return result

    #
    # Indica si la pregunta cambia de fuente y de variable, lo que la hace un tema nuevo.
    #
    @staticmethod
    def _changes_topic(result, question, messages, resolved):
        own_variable = any(
            item.kind == "variable"
            for item in result.data_mentions
            if item not in result.conserved_mentions
        )
        answers_clarification = any(item.clarification for item in messages)
        continues = starts_with_phrase(question, CONTINUATION_CUES)

        return bool(messages and resolved and own_variable) and not (
            answers_clarification or continues
        )

    #
    # Indica si la pregunta define otro cálculo (estadística, agrupación o período) propio.
    #
    @staticmethod
    def _redefines(result, question):
        regroups = any(
            item.kind in {"dimension", "date"}
            for item in result.data_mentions
            if item not in result.conserved_mentions
        )
        return bool(
            requested_statistics(question)
            or contains_any_phrase(question, MEASURED_STATISTICS)
            or requested_time_grain(question)
            or regroups
        )

    #
    # Indica si los datasets del análisis citado deben volver a elegirse por el nuevo valor o variable.
    #
    @staticmethod
    def _reselects(result, messages):
        changes = any(
            item.kind in {"value", "variable"}
            for item in result.data_mentions
            if item not in result.conserved_mentions
        )
        return changes and not any(item.named_sources for item in messages)

    #
    # Quita del resultado el contexto conservado y trata la pregunta como una consulta nueva.
    #
    @staticmethod
    def _without_context(result):
        result.data_mentions = [
            item for item in result.data_mentions if item not in result.conserved_mentions
        ]
        result.conserved_mentions = []
        result.history_message_ids = []
        result.context_used = False
        result.warnings.append(
            "La pregunta nombra otra fuente y otra variable: se trató como una consulta nueva."
        )
        return []

    #
    # Indica si se pide una estadística sin variable que calcular.
    #
    @staticmethod
    def _lacks_variable(result, texts):
        measured = any(contains_any_phrase(text, MEASURED_STATISTICS) for text in texts)
        return measured and not any(item.kind == "variable" for item in result.data_mentions)

    #
    # Registra como valor tentativo la respuesta breve a una aclaración.
    #
    @staticmethod
    def _record_tentative_value(result, question, messages, resolved):
        answers_clarification = any(item.clarification for item in messages)
        words = question.strip(" ¿?¡!.").split()
        short_reply = 0 < len(words) <= 6
        names_value = any(item.kind == "value" for item in result.data_mentions)
        if answers_clarification and not resolved and short_reply and not names_value:
            result.tentative_values = [" ".join(words)]

    #
    # Construye los vínculos a datasets con las fuentes de los análisis anteriores citados.
    #
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

    #
    # Deja un único vínculo por dataset.
    #
    @staticmethod
    def _by_dataset(bindings):
        return {item.dataset_id: item for item in bindings}

    #
    # Registra el fallo del Router en el resultado, con sus tokens.
    #
    @staticmethod
    def _record_failure(result, error):
        result.route = "failed"
        result.error = str(error)
        for field in ("input_tokens", "output_tokens"):
            value = getattr(error, field, None)
            if value is not None:
                setattr(result, field, value)

    #
    # Registra la duración de la ruta y escribe su resumen en el log.
    #
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

    #
    # Amplía los mensajes citados con los que ellos citan y con sus respuestas.
    #
    @staticmethod
    def _with_context(messages, provided):
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

    #
    # Agrega a la selección los mensajes que cita un mensaje y existen en el historial.
    #
    @staticmethod
    def _select_cited(message, provided, selected):
        for key in message.context_message_ids:
            if key in provided:
                selected.setdefault(key, provided[key])

    #
    # Devuelve la respuesta del asistente que sigue a un mensaje del usuario, si existe.
    #
    @staticmethod
    def _reply_to(message, order, provided):
        index = order.index(message.message_id)
        if message.role != "user" or index + 1 >= len(order):
            return None

        following = provided[order[index + 1]]
        return following if following.role == "assistant" else None

    #
    # Reúne las solicitudes anteriores y la pregunta actual, hasta un máximo de ocho.
    #
    @staticmethod
    def _requests(messages, question):
        requests = [
            request
            for message in messages
            for request in RouteQueryService._message_requests(message)
        ]
        return [*dict.fromkeys(item for item in requests if item != question), question][-8:]

    #
    # Devuelve las solicitudes de un mensaje: las pendientes si es aclaración, o las de su análisis.
    #
    @staticmethod
    def _message_requests(message):
        if message.clarification:
            return message.pending_requests
        if message.analysis:
            return message.analysis.request_history
        return []

    #
    # Devuelve las solicitudes pendientes de las aclaraciones citadas.
    #
    @staticmethod
    def _pending_requests(messages):
        return [
            request
            for message in messages
            if message.clarification
            for request in message.pending_requests
        ]

    #
    # Conserva sólo las menciones literales de la pregunta o de sus solicitudes citadas.
    #
    @staticmethod
    def _literal_mentions(mentions, question, messages, result):
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

    #
    # Convierte los números comparados («mayor que 500») en umbrales y recupera su variable.
    #
    @staticmethod
    def _comparison_mentions(mentions, texts):
        mentions = [
            item for item in mentions if item.kind != "threshold" or numeric_comparisons(item.text)
        ]
        found = [(text, item.text) for text in texts for item in numeric_comparisons(text)]
        if not found:
            return mentions

        kept, variables = [], []
        for item in mentions:
            compared = [
                comparison
                for text, comparison in found
                if RouteQueryService._overlaps(item.text, comparison, text)
            ]
            copied = item.kind == "threshold" or any(char.isdigit() for char in item.text)
            if item.kind not in _NUMERIC_KINDS or not compared or not copied:
                kept.append(item)
            elif item.kind == "threshold":
                variables.extend(RouteQueryService._compared_variable(item.text, compared[0]))

        known = {item.text.casefold() for item in kept if item.kind == "variable"}
        added = {
            text.casefold(): DataMention(text=text, kind="variable")
            for text in variables
            if text.casefold() not in known
        }
        thresholds = {
            comparison.casefold(): DataMention(text=comparison, kind="threshold")
            for _, comparison in found
        }
        return [*kept, *added.values(), *thresholds.values()]

    #
    # Devuelve la variable que antecede a la comparación en un umbral copiado.
    #
    @staticmethod
    def _compared_variable(mention, comparison):
        index = mention.casefold().find(comparison.casefold())
        prefix = mention[:index].strip() if index > 0 else ""

        return [prefix] if variable_core(prefix) else []

    #
    # Separa de la variable las otras menciones y la agrupación temporal que el modelo copió.
    #
    @staticmethod
    def _separated_variables(mentions):
        others = [item.text for item in mentions if item.kind in _SCOPE_KINDS]
        separated = {}
        for item in mentions:
            text, kind = item.text, item.kind
            for other in others if kind == "variable" else ():
                text = RouteQueryService._without_edge(text, other)
            if kind == "variable":
                text = RouteQueryService._without_time_grain(text)
            words = normalize_text(text).split()
            names_category = kind == "value" and bool(words) and words[0] in CATEGORY_CUES
            if names_category:
                kind = "dimension"
            if kind == "dimension":
                text = RouteQueryService._trimmed(text) or text
            separated.setdefault(
                (text.casefold(), kind), item.model_copy(update={"text": text, "kind": kind})
            )
        return list(separated.values())

    #
    # Quita de una variable la agrupación temporal final («por mes»).
    #
    @staticmethod
    def _without_time_grain(text):
        words = (RouteQueryService._trimmed(text) or text).split()
        for cue in (cue for cues in TIME_GRAIN_CUES.values() for cue in cues):
            size = len(cue.split())
            if len(words) > size and normalize_text(" ".join(words[-size:])) == cue:
                return RouteQueryService._trimmed(" ".join(words[:-size])) or text

        return text

    #
    # Quita de un texto otra mención ubicada en uno de sus extremos.
    #
    @staticmethod
    def _without_edge(text, other):
        index = text.casefold().find(other.casefold())
        if index == 0:
            remainder = text[len(other) :]
        elif index > 0 and index + len(other) == len(text):
            remainder = text[:index]
        else:
            return text

        return RouteQueryService._trimmed(remainder) or text

    #
    # Quita los conectores del inicio y del final de un texto.
    #
    @staticmethod
    def _trimmed(text):
        words = text.split()
        while words and normalize_text(words[0]) in MENTION_CONNECTORS:
            words.pop(0)
        while words and normalize_text(words[-1]) in MENTION_CONNECTORS:
            words.pop()
        return " ".join(words)

    #
    # Indica si una mención y una comparación se solapan en el texto.
    #
    @staticmethod
    def _overlaps(mention, comparison, text):
        words = normalize_text(text).split()
        spans = RouteQueryService._spans(words, normalize_text(mention).split())
        compared = RouteQueryService._spans(words, normalize_text(comparison).split())

        return any(
            start < other_end and other_start < end
            for start, end in spans
            for other_start, other_end in compared
        )

    #
    # Devuelve las posiciones donde una secuencia de palabras aparece en una lista.
    #
    @staticmethod
    def _spans(words, target):
        size = len(target)
        return [
            (index, index + size)
            for index in range(len(words) - size + 1)
            if size and words[index : index + size] == target
        ]

    #
    # Devuelve las menciones de la pregunta actual que no estaban en las solicitudes citadas.
    #
    @staticmethod
    def _changes(mentions, sources, question, messages):
        cited = RouteQueryService._requests(messages, question)[:-1]
        asked_sources = [text for text in sources if RouteQueryService._literal(text, question)]
        texts = [*(item.text for item in mentions), *asked_sources]
        return [
            text
            for text in texts
            if not any(RouteQueryService._literal(text, request) for request in cited)
        ]

    #
    # Indica si la mención aparece en el texto, sin importar acentos, mayúsculas ni puntuación.
    #
    @staticmethod
    def _literal(mention, text):
        target = normalize_text(mention)
        return bool(target) and target in normalize_text(text)

    #
    # Filtra las menciones de fuente que aparecen en la pregunta, descartando valores sin señal de fuente.
    #
    @staticmethod
    def _explicit_mentions(mentions, question, data_mentions, result):
        values = {value.text.strip().casefold() for value in data_mentions if value.kind == "value"}
        explicit = []
        for mention in dict.fromkeys(item.strip() for item in mentions):
            if not mention or mention.casefold() not in question.casefold():
                result.warnings.append(
                    "Se descartó una mención de fuente que no aparece en la pregunta."
                )
                continue

            is_value = mention.casefold() in values
            if not is_value or RouteQueryService._has_source_cue(mention, question):
                explicit.append(mention)
        return explicit

    #
    # Detecta fuentes nombradas en la pregunta que el modelo no marcó como tales.
    #
    async def _catalog_references(self, question, mentions, catalog, explicit):
        known = {normalize_name(item) for item in explicit}
        found = self._multiword_sources_in(question, catalog, known)
        aliases = []

        values = [item.text for item in mentions if item.kind == "value"]
        for text in dict.fromkeys([*values, *self._cued_names(question)]):
            if any(contains_phrase(name, text) for name in [*explicit, *found]):
                continue

            owners = self._value_owners(text, catalog)
            if len(owners) == 1 and not await self._stored_anywhere(text, catalog):
                found.append(next(iter(owners.values()))["name"])
                aliases.append(text)
        return found, aliases

    #
    # Devuelve los nombres que siguen a expresiones como «los datos de» o «la base».
    #
    @staticmethod
    def _cued_names(question):
        return [match.group("name") for match in SOURCE_ALIAS.finditer(question)]

    #
    # Devuelve las fuentes de nombre compuesto que la pregunta menciona.
    #
    @staticmethod
    def _multiword_sources_in(question, catalog, known):
        found = []
        for source in catalog:
            name = normalize_name(source["name"])
            if len(name.split()) >= 2 and name not in known and contains_phrase(question, name):
                found.append(source["name"])
        return found

    #
    # Devuelve las fuentes cuyo nombre o cuyas hojas contienen el valor.
    #
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

    #
    # Indica si el valor figura en los registros de algún dataset disponible.
    #
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
            return True

        for matches in found.values():
            for row in matches.get(value, []):
                if isinstance(row["value"], str) and contains_phrase(row["value"], value):
                    return True
        return False

    #
    # Indica si la mención está precedida por una señal de fuente (archivo, base).
    #
    @staticmethod
    def _has_source_cue(mention, question):
        if UPLOADED_FILE_EXTENSION.search(mention):
            return True

        for match in re.finditer(re.escape(mention), question, re.IGNORECASE):
            prefix = question[max(0, match.start() - SOURCE_CUE_WINDOW) : match.start()]

            if SOURCE_CUE.search(prefix):
                return True

        return False

    #
    # Resuelve las menciones de fuente contra el catálogo y devuelve los vínculos a datasets.
    #
    def _resolve_sources(self, mentions, question, catalog, *, partial=False):
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
                continue

            available = self._available_matches(mention, matches)
            bindings.extend(
                DatasetBinding(dataset_id=dataset["dataset_id"], source_id=source["source_id"])
                for source, dataset in available
            )
            resolved.append(mention)
        return bindings, resolved

    #
    # Devuelve el UUID de una mención o None si no lo es.
    #
    @staticmethod
    def _as_identifier(mention):
        try:
            return UUID(mention.strip())
        except ValueError:
            return None

    #
    # Busca en el catálogo los datasets que coinciden con la mención por id o por nombre.
    #
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

    #
    # Construye el mensaje de que no existe una fuente con ese nombre.
    #
    @staticmethod
    def _unknown_source_message(mention, catalog):
        names = "; ".join(source["name"] for source in catalog[:8])
        listed = f" ({names})" if names else ""
        return (
            f"No encontré una fuente llamada {mention} entre las cargadas"
            + listed
            + ". ¿Cuál querés usar?"
        )

    #
    # Devuelve los datasets disponibles de la fuente coincidente, rechazando coincidencias ambiguas o inactivas.
    #
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

    #
    # Convierte el resultado en una solicitud de aclaración, con la pregunta en el idioma adecuado.
    #
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
