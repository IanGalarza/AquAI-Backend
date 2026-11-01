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
    mentioned_count,
    normalize_name,
    normalize_text,
    numeric_comparisons,
    qualified_period,
    qualified_variable,
    qualifies_variable,
    starts_with_phrase,
    variable_core,
)
from aquai.application.planning.context import chart_requested, requested_statistics
from aquai.application.planning.followUp import requested_period, requested_time_grain
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
# Menciones que el modelo confunde con un umbral cuando copian su número.
_NUMERIC_KINDS = {"value", "limit", "threshold"}
# Menciones que acotan o agrupan y que el modelo a veces copia dentro de la variable.
_SCOPE_KINDS = {"value", "dimension", "threshold", "date"}
# Condiciones de un análisis citado que un seguimiento conserva si no las cambia; umbrales y cantidades
# viajan con sus operaciones verificadas.
_CONSERVED_KINDS = {"variable", "value", "dimension", "date"}
# Menciones que acotan registros: un inventario de fuentes no las responde.
_RECORD_SCOPE_KINDS = {"value", "date", "threshold"}
# Menciones que cambian lo calculado: una pregunta por el período del resultado no las incluye.
_CALCULATION_KINDS = {"variable", "value", "dimension", "threshold", "limit"}
# Palabras que no agregan un pedido propio a un fragmento («¿Y en Punta Piedras?»).
_ELLIPSIS_WORDS = {*MENTION_CONNECTORS, *(word for cue in CONTINUATION_CUES for word in cue.split())}
_PLACE_NAMES = {normalize_text(word) for word in PLACE_WORDS}
_CHART_FILLERS = {*MENTION_CONNECTORS, "un", "una", "a", "an"}
_CHART_WORD = re.compile(rf"\b{CHART_WORDS}\b", re.IGNORECASE)
# «los datos de ABSA», «la base FQ»: la palabra siguiente puede abreviar el nombre de una fuente.
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
# «¿Cuándo se cargó el archivo?» pregunta por una fuente, no por el período de un resultado.
_SOURCE_WORD = re.compile(rf"\b(?:{'|'.join(SOURCE_CUE_WORDS)})s?\b", re.IGNORECASE)


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
        # Otro sitio, variable, fuente, cantidad o un gráfico cambian el resultado: no es repetirlo.
        if route == "historical" and (
            self._changes(asked, decision.source_mentions, state.question, messages)
            or chart_requested(state.question)
        ):
            route = "data"

        anchor = self._anchor(state.conversation_history, allowed)
        subject = self._period_subject(state.question, asked, messages, anchor, route, decision)
        if subject is not None:
            # Se responde con las fechas de ese resultado, recalculado con sus mismas operaciones: repetir
            # sus párrafos no contesta el año, y planificar de nuevo puede cambiar lo que se calculó.
            route = "data"
            result.requested_period = requested_period(state.question)
            messages = self._with_context([subject], provided)
        elif anchor is not None and self._fragment(decision, route, asked, state.question):
            # Un fragmento («¿Y el pH?», «La conductividad») continúa el último análisis o aclaración,
            # aunque el modelo cite otro mensaje, pida aclarar o el último turno haya fallado.
            route = "data"
            messages = self._with_context([anchor], provided)
        elif self._needs_implicit_reference(decision, route, asked, messages):
            if anchor is not None:
                # Sin una cita explícita, el seguimiento refiere al último resultado o aclaración.
                messages = self._with_context([anchor], provided)
            elif decision.context_required and route == "data" and not decision.data_mentions:
                return self._clarify(result, "¿A qué resultado o consulta anterior te referís?")

        self._record_mentions(result, decision, state.question, messages)
        if route == "needs_clarification":
            return self._model_clarification(result, decision, route, messages, state.question)

        return await self._resolve_route(state, catalog, result, decision, route, messages)

    async def _resolve_route(self, state, catalog, result, decision, route, messages):
        explicit, aliases = await self._named_sources(decision, state.question, catalog, result)
        explicit = await self._stored_values(explicit, state.question, catalog, result)
        if route == "general" and explicit:
            route = "catalog"  # Una pregunta sobre una fuente nombrada se responde con el catálogo.
        result.route = route

        if route == "historical":
            return self._historical_result(result, messages, state.conversation_history)
        if route == "general":
            return self._general_result(result, messages)

        # En una consulta de inventario, un nombre parcial («ABSA») refiere a una fuente.
        bindings, resolved = self._resolve_sources(
            explicit, state.question, catalog, partial=route == "catalog"
        )
        self._record_sources(result, resolved, aliases)
        if route == "catalog" and self._restricts_records(result):
            # «¿Cuántas muestras hay de Bagliardi?» cuenta registros de un sitio: el inventario no lo responde.
            route = result.route = "data"
            result.warnings.append(
                "La pregunta acota registros por un valor, una fecha o un umbral: "
                "se consultaron los datos y no el inventario de fuentes."
            )
        if route == "catalog":
            return self._catalog_result(result, bindings)

        return self._data_result(state, result, messages, bindings, resolved)

    async def _stored_values(self, explicit, question, catalog, result):
        """Un nombre que no es el de ninguna fuente pero figura en los registros («Bagliardi») es un valor que
        acota la consulta: descartarlo como fuente perdería ese filtro."""
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
            # Como cualquier valor nombrado ahora, reemplaza al conservado del análisis anterior.
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

    @staticmethod
    def _restricts_records(result):
        return any(item.kind in _RECORD_SCOPE_KINDS for item in result.data_mentions)

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
    def _anchor(history, allowed):
        """Último análisis verificado o aclaración visible: un turno fallido no corta la conversación."""
        replies = [
            item for item in history if item.role == "assistant" and str(item.message_id) in allowed
        ]
        return next(
            (item for item in reversed(replies) if item.analysis is not None or item.clarification),
            None,
        )

    @staticmethod
    def _period_subject(question, asked, messages, anchor, route, decision):
        """Análisis verificado cuyo período pregunta un seguimiento sin condiciones propias («¿Y de qué año es?»):
        el único citado o, si no, el último. Una pregunta por una fuente o por el catálogo («¿Cuándo se cargó el
        archivo?», «¿Qué fechas abarca ABSA?») no lo es."""
        own = [item for item in asked if item.kind in _CALCULATION_KINDS]
        about_source = (
            route == "catalog"
            or _SOURCE_WORD.search(question) is not None
            or any(RouteQueryService._literal(text, question) for text in decision.source_mentions)
        )
        if own or about_source or chart_requested(question) or requested_period(question) is None:
            return None

        cited = [item for item in messages if item.analysis is not None]
        if len(cited) == 1:
            return cited[0]

        return anchor if anchor is not None and anchor.analysis is not None else None

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
    def _fragment(decision, route, asked, question):
        """Sólo menciones y conectores («¿Y en Punta Piedras?») u otra presentación sin condiciones propias
        («Ahora mostralo en un gráfico de líneas»): no hay un pedido nuevo que buscar."""
        presentation = chart_requested(question) and all(
            RouteQueryService._presentation(item) for item in asked
        )
        # Una fuente que el modelo agregó sin que figure en la pregunta no es un pedido propio.
        names_source = any(
            RouteQueryService._literal(text, question) for text in decision.source_mentions
        )
        return (
            route in {"data", "needs_clarification"}
            and not names_source
            and (presentation or RouteQueryService._elliptical(question, asked))
        )

    @staticmethod
    def _elliptical(question, asked):
        covered = {word for item in asked for word in normalize_text(item.text).split()}
        words = normalize_text(question).split()

        return bool(asked) and all(word in covered or word in _ELLIPSIS_WORDS for word in words)

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
        mentions = RouteQueryService._qualified_variables(
            RouteQueryService._separated_variables(mentions), question
        )
        mentions = RouteQueryService._asked_dimensions(mentions, question)
        own, conserved = RouteQueryService._conserved_scope(mentions, question, messages)
        result.data_mentions = [*own, *conserved]
        result.conserved_mentions = conserved

    @staticmethod
    def _qualified_variables(mentions, question):
        """«¿Y la temperatura del aire?»: el calificativo que sigue a la variable en la pregunta es parte de ella aunque
        el modelo lo omita; sin él se respondería con la temperatura del agua. Lo mismo el extremo de una fecha:
        «desde 2020» no es «2020»."""
        qualify = {"variable": qualified_variable, "date": qualified_period}
        qualified = [
            item.model_copy(update={"text": qualify[item.kind](item.text, question)})
            if item.kind in qualify
            else item
            for item in mentions
        ]
        # El modelo a veces repite el calificativo como valor («aire»): no es un sitio ni otro valor registrado.
        variables = [item.text for item in qualified if item.kind == "variable"]
        return [
            item
            for item in qualified
            if not (item.kind == "value" and any(qualifies_variable(item.text, text) for text in variables))
        ]

    @staticmethod
    def _asked_dimensions(mentions, question):
        """«¿Qué sitios…?», «por estación»: el lugar que se pregunta es una dimensión aunque el modelo no la marque.
        Sin ella, un dataset sin esa columna se agruparía por cualquier otra (un identificador de muestra)."""
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

    @staticmethod
    def _without_chart_phrases(mentions):
        """«gráfico de líneas» pide una presentación: no es una variable, un valor ni una cantidad de filas."""
        return [item for item in mentions if not RouteQueryService._presentation(item)]

    @staticmethod
    def _presentation(item):
        # El modelo a veces copia todo el pedido («Ahora mostralo en un gráfico») como una cantidad sin número.
        uncounted = item.kind == "limit" and mentioned_count(item.text) is None
        return RouteQueryService._names_chart(item.text) or (
            uncounted and _CHART_WORD.search(item.text) is not None
        )

    @staticmethod
    def _names_chart(text):
        remainder = _CHART_WORD.sub(" ", text)
        words = [word for word in normalize_text(remainder).split() if word not in _CHART_FILLERS]
        return remainder != text and not words

    @staticmethod
    def _conserved_scope(mentions, question, messages):
        """Un seguimiento conserva las condiciones del análisis citado que no cambia y reemplaza las que nombra.

        El alcance vigente es el que interpretó el turno citado; lo que el modelo copió de solicitudes anteriores no
        es un pedido nuevo ni lo reemplaza: «¿Y en Punta Piedras?» cambia el sitio y conserva la variable, y un
        sitio ya reemplazado no vuelve. Valores, dimensiones y fechas sólo se conservan si la pregunta continúa la
        anterior: no nombra otra variable o empieza como continuación («¿Y la turbiedad?»). Un conector aditivo
        («compará con …») suma valores en lugar de reemplazarlos."""
        current = [question, *RouteQueryService._pending_requests(messages)]
        own = [
            item
            for item in mentions
            if any(RouteQueryService._literal(item.text, text) for text in current)
        ]
        if not messages:
            return own, []

        asked = {item.kind for item in own}
        # Responder una aclaración completa la solicitud pendiente: nunca empieza otra.
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

    @staticmethod
    def _conserves(kind, asked, continues, additive):
        if kind not in _CONSERVED_KINDS:
            return False
        if kind == "variable":
            return kind not in asked

        replaced = kind in asked and not (additive and kind == "value")
        return continues and not replaced

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
    def _historical_result(result, messages, history):
        # Una respuesta que reprodujo otra remite al original, con sus párrafos y citas.
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

        if self._changes_topic(result, state.question, messages, resolved):
            messages = self._without_context(result)

        question, requests = contextual_question(
            state.question, messages, redefined=self._redefines(result, state.question)
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

    @staticmethod
    def _changes_topic(result, question, messages, resolved):
        """Otra fuente y otra variable en una pregunta propia es un tema nuevo aunque el modelo cite el anterior:
        no hereda filtros, agrupaciones ni datasets («¿Cuál es la conductividad máxima en la base FQ?»)."""
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

    @staticmethod
    def _redefines(result, question):
        """Una estadística, una agrupación o un período propios definen otro cálculo sobre la misma selección."""
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

    @staticmethod
    def _reselects(result, messages):
        """«¿Y en Bagliardi?» después de Punta Lara: los datasets de aquel resultado se eligieron por Punta Lara, no
        por el usuario, y otros pueden registrar la variable en Bagliardi. Si el usuario nombró las fuentes, se
        conservan."""
        changes = any(
            item.kind in {"value", "variable"}
            for item in result.data_mentions
            if item not in result.conserved_mentions
        )
        return changes and not any(item.named_sources for item in messages)

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

    @staticmethod
    def _lacks_variable(result, texts):
        """Un promedio o un máximo sin variable no tiene qué calcular: el Planner elegiría una por su cuenta."""
        measured = any(contains_any_phrase(text, MEASURED_STATISTICS) for text in texts)
        return measured and not any(item.kind == "variable" for item in result.data_mentions)

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
    def _comparison_mentions(mentions, texts):
        """Un número comparado («mayor que 500») es un umbral, no un valor almacenado ni una cantidad de filas.

        El modelo suele copiarlo como value («500»), lo que restringiría la recuperación a registros con ese
        texto. Como los límites, sólo cuentan los de la pregunta actual o de la solicitud que esperaba aclaración.
        Un superlativo sin número («mayor») no es un umbral."""
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
            # Sólo el número copiado es parte del umbral: un sitio dentro de la comparación sigue acotando
            # registros («mayor que 8 en Punta Lara en promedio»).
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

    @staticmethod
    def _compared_variable(mention, comparison):
        """Un umbral copiado con su variable («pH promedio mayor que 8») conserva esa variable: pH."""
        index = mention.casefold().find(comparison.casefold())
        prefix = mention[:index].strip() if index > 0 else ""

        return [prefix] if variable_core(prefix) else []

    @staticmethod
    def _separated_variables(mentions):
        """El modelo copia a veces la variable con otra mención («turbiedad promedio en Punta Lara»).

        Esa otra mención ya figura aparte: se recorta de la variable si está en un extremo, así el texto sigue
        siendo literal y el Retriever puede encontrar su columna. Una dimensión pierde sus conectores
        («por sitio» → sitio) por la misma razón."""
        others = [item.text for item in mentions if item.kind in _SCOPE_KINDS]
        separated = {}
        for item in mentions:
            text, kind = item.text, item.kind
            for other in others if kind == "variable" else ():
                text = RouteQueryService._without_edge(text, other)
            if kind == "variable":
                # «turbiedad promedio por mes»: la agrupación temporal la lee el Planner de la pregunta.
                text = RouteQueryService._without_time_grain(text)
            # «cada punto de muestreo» o «qué sitios» nombran una categoría, nunca un valor almacenado.
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

    @staticmethod
    def _without_time_grain(text):
        words = (RouteQueryService._trimmed(text) or text).split()
        for cue in (cue for cues in TIME_GRAIN_CUES.values() for cue in cues):
            size = len(cue.split())
            if len(words) > size and normalize_text(" ".join(words[-size:])) == cue:
                return RouteQueryService._trimmed(" ".join(words[:-size])) or text

        return text

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

    @staticmethod
    def _trimmed(text):
        words = text.split()
        while words and normalize_text(words[0]) in MENTION_CONNECTORS:
            words.pop(0)
        while words and normalize_text(words[-1]) in MENTION_CONNECTORS:
            words.pop()
        return " ".join(words)

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

    @staticmethod
    def _spans(words, target):
        size = len(target)
        return [
            (index, index + size)
            for index in range(len(words) - size + 1)
            if size and words[index : index + size] == target
        ]

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
        # sus hojas («la base FQ»), la nombra; también la palabra que sigue a «los datos de» o «la base»
        # aunque el modelo no la haya copiado («En los datos de ABSA, …»).
        values = [item.text for item in mentions if item.kind == "value"]
        for text in dict.fromkeys([*values, *self._cued_names(question)]):
            if any(contains_phrase(name, text) for name in [*explicit, *found]):
                continue

            owners = self._value_owners(text, catalog)
            if len(owners) == 1 and not await self._stored_anywhere(text, catalog):
                found.append(next(iter(owners.values()))["name"])
                aliases.append(text)
        return found, aliases

    @staticmethod
    def _cued_names(question):
        return [match.group("name") for match in SOURCE_ALIAS.finditer(question)]

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
