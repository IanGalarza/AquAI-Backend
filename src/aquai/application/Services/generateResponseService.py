from time import perf_counter

from aquai.application.answering.blocked import blocked_explanations
from aquai.application.answering.context import (
    build_response_context,
    response_language,
)
from aquai.application.answering.coverage import coverage_notes, period_notes, unit_notes
from aquai.application.answering.segments import answer_text
from aquai.application.contracts.answering import (
    AnswerSegment,
    GenerationRecord,
    QueryResponse,
    QueryResponseState,
    ResponseOperationReference,
    ResponseResultReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.execution import evidence_fingerprint
from aquai.application.interfaces.answering import ResponseProvider

_RANK = {"result": 0, "meaning": 1, "summary": 2, "chart": 3, "scope": 4, "detail": 5}

_MESSAGES = {
    "es": {
        "validated": "",
        "partial": "Pude responder sólo una parte de la consulta; el resto no se pudo calcular de forma verificable.",
        "needs_clarification": "Necesito que me aclares algo para continuar.",
        "insufficient_coverage": "No encontré datos suficientes para responder esta pregunta en las fuentes que revisé. Esto no significa que la información no exista: puede estar en otra fuente o con otro nombre.",
        "failed": "No pude armar un análisis confiable para esta pregunta con los datos disponibles. Probá reformularla indicando la variable, el sitio o la fuente que querés usar.",
        "blocked": "No pude completar el análisis con los datos disponibles.",
        "generation_failed": "No pude redactar la respuesta completa. Estos son los resultados verificados:",
    },
    "en": {
        "validated": "",
        "partial": "I could only answer part of the question; the rest could not be calculated in a verifiable way.",
        "needs_clarification": "I need you to clarify something to continue.",
        "insufficient_coverage": "I did not find enough data to answer this question in the sources I reviewed. This does not mean the information does not exist: it may be in another source or under another name.",
        "failed": "I could not build a reliable analysis for this question with the available data. Try rephrasing it with the variable, site or source you want to use.",
        "blocked": "I could not complete the analysis with the available data.",
        "generation_failed": "I could not write the complete answer. These are the verified results:",
    },
}


#
# Calcula la posición de una oración según el orden de su evidencia y el tipo de enunciado.
#
def _position(entry, order):
    evidence_position = order.get(entry["evidence_ids"][0], len(order))
    kind_position = _RANK.get(entry.get("kind"), len(_RANK))

    return evidence_position, kind_position


class GenerateResponseService:
    #
    # Guarda el proveedor de redacción y los límites de filas y de oraciones deterministas.
    #
    def __init__(
        self, provider: ResponseProvider, *, max_preview_rows=3, deterministic_sentences=24
    ):
        self._provider = provider
        self._max_preview_rows = max_preview_rows
        self._deterministic_sentences = deterministic_sentences

    #
    # Genera la respuesta final a partir de la validación y la evidencia, con textos verificados.
    #
    async def generate(self, state: QueryResponseState) -> QueryResponse:
        started = perf_counter()
        validation = state.validation
        language = response_language(state.question, state.response_language)

        if validation is None:
            raise ValueError("Response Generator requiere el resultado del Validator.")

        response = _initial_response(state, validation, language)
        plan = state.planning
        _record_planner_usage(response, plan)
        _record_tool_timings(response, state)

        if _evidence_changed(state, validation, plan):
            self._reject_changed_evidence(response, language)
            return self._finish(response, started)

        if _cannot_answer(validation):
            self._publish_unanswered(response, state, validation, language)
            return self._finish(response, started)

        steps = {step.id: step for step in plan.operations}
        refs = _references_by_relevance(validation, steps)
        _attach_evidence(response, state, validation, steps)

        context, language = build_response_context(
            state, self._provider.context_budget_bytes, self._max_preview_rows
        )
        _attach_context(response, context, language)

        catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
        entries = [(entry, entry["direct"]) for entry in self._ordered_entries(catalog, refs)]
        introduction = _introduction(context)
        notes = [
            *unit_notes(state, language),
            *coverage_notes(state, language),
            *period_notes(state, language),
        ]
        coverage = [AnswerSegment(kind="note", text=text) for text in notes]

        if validation.status == "partial":
            self._publish_partial(response, state, language, introduction, entries)
        elif len(catalog) <= self._deterministic_sentences:
            self._publish(response, [*introduction, *self._segments(entries)])
        else:
            await self._publish_model_draft(
                response,
                context,
                catalog=catalog,
                refs=refs,
                introduction=introduction,
                entries=entries,
                language=language,
            )

        if coverage:
            self._publish(response, [*response.segments, *coverage])
        return self._finish(response, started)

    #
    # Marca la respuesta como fallida porque la evidencia cambió después de validarse.
    #
    def _reject_changed_evidence(self, response, language):
        response.status = "failed"
        response.errors.append(
            ExecutionError(
                code="evidence_changed", message="La evidencia cambió después de su validación."
            )
        )
        self._publish(response, [AnswerSegment(text=self._message("failed", language))])

    #
    # Publica la explicación de por qué no se pudo responder, con los motivos concretos.
    #
    def _publish_unanswered(self, response, state, validation, language):
        reasons = blocked_explanations(state, language)
        explanation = AnswerSegment(
            text=self._unanswered(validation, language, explained=bool(reasons))
        )
        notes = [AnswerSegment(kind="note", text=text) for text in reasons]

        self._publish(response, [explanation, *notes])

    #
    # Publica una respuesta parcial indicando primero lo que no se pudo calcular.
    #
    def _publish_partial(self, response, state, language, introduction, entries):
        missing = blocked_explanations(state, language) or [self._message("partial", language)]
        notes = [AnswerSegment(kind="note", text=text) for text in missing]

        self._publish(response, [*notes, *introduction, *self._segments(entries)])

    #
    # Redacta la respuesta con el modelo y, si falla, publica los textos verificados con un aviso.
    #
    async def _publish_model_draft(
        self, response, context, *, catalog, refs, introduction, entries, language
    ):
        response.generation.model = self._provider.model
        model_started = perf_counter()

        try:
            drafted = await self._draft(response.generation, context, catalog, refs)
            self._publish(response, [*introduction, *self._segments(drafted)])
            response.generation.status = "generated"
        except LLMProviderError as error:
            _record_generation_failure(response, error, context)
            notice = AnswerSegment(kind="note", text=self._message("generation_failed", language))
            self._publish(response, [notice, *introduction, *self._segments(entries)])

        response.generation.duration_ms = round((perf_counter() - model_started) * 1000, 3)

    #
    # Pide al modelo la redacción y devuelve las oraciones de evidencia elegidas, ya comprobadas.
    #
    async def _draft(self, generation, context, catalog, refs):
        if context.size_bytes > context.budget_bytes:
            raise LLMProviderError("La evidencia mínima supera el presupuesto de redacción.")

        context.sent_to_model = True
        reply = await self._provider.generate(context)
        generation.model = reply.model
        generation.input_tokens = reply.input_tokens
        generation.output_tokens = reply.output_tokens

        sentences = reply.draft.sentences
        _check_sentence_selection(sentences, catalog)

        return self._drafted_entries(sentences, catalog, refs)

    #
    # Convierte las oraciones elegidas por el modelo en entradas ordenadas, rechazando citas no autorizadas.
    #
    def _drafted_entries(self, sentences, catalog, refs):
        drafted = []

        for sentence in self._ordered(sentences, catalog, refs):
            entry = catalog[sentence.sentence_id]
            if _cites_unauthorized_evidence(sentence, entry, refs):
                raise LLMProviderError("La redacción cita evidencia no autorizada.")
            drafted.append((entry, entry[sentence.variant]))

        return drafted

    #
    # Ordena las oraciones del catálogo según la relevancia de su evidencia.
    #
    @staticmethod
    def _ordered_entries(catalog, refs):
        order = {key: index for index, key in enumerate(refs)}

        return sorted(catalog.values(), key=lambda entry: _position(entry, order))

    #
    # Ordena las oraciones elegidas por el modelo según la relevancia de su evidencia.
    #
    @staticmethod
    def _ordered(sentences, catalog, refs):
        order = {key: index for index, key in enumerate(refs)}

        return sorted(sentences, key=lambda item: _position(catalog[item.sentence_id], order))

    #
    # Agrupa las oraciones consecutivas con la misma evidencia en segmentos de respuesta.
    #
    @staticmethod
    def _segments(entries):
        segments = []

        for entry, text in entries:
            previous = segments[-1] if segments else None
            if previous is not None and previous.evidence_ids == entry["evidence_ids"]:
                previous.text += " " + text
                previous.attachment = previous.attachment or entry.get("attachment")
                continue

            segments.append(
                AnswerSegment(
                    text=text,
                    evidence_ids=list(entry["evidence_ids"]),
                    attachment=entry.get("attachment"),
                )
            )

        return segments

    #
    # Guarda los segmentos en la respuesta y compone su texto completo.
    #
    @staticmethod
    def _publish(response, segments):
        response.segments = segments
        response.answer = answer_text(segments)

    #
    # Compone el mensaje de una consulta sin respuesta, con sus motivos y preguntas pendientes.
    #
    @classmethod
    def _unanswered(cls, validation, language, *, explained=False):
        status = "blocked" if explained and validation.status == "failed" else validation.status
        text = cls._message(status, language) or cls._message("failed", language)

        if any(item.code == "incomplete_materialization" for item in validation.issues):
            text += (
                " Uno de los resultados llegó incompleto, así que no lo usé para responder."
                if language == "es"
                else " One of the results was incomplete, so I did not use it to answer."
            )

        if validation.clarification_question:
            text += " " + validation.clarification_question
        elif validation.missing_information:
            text += _missing_information_text(validation)

        return text

    #
    # Devuelve el mensaje fijo asociado a un estado de validación y a un idioma.
    #
    @staticmethod
    def _message(status, language):
        messages = _MESSAGES[language]

        return messages.get(status, messages["failed"])

    #
    # Registra la duración de la generación en la respuesta y la devuelve.
    #
    @staticmethod
    def _finish(response, started):
        response.timings_ms["response"] = round((perf_counter() - started) * 1000, 3)

        return response


#
# Crea la respuesta inicial con el estado, las advertencias y los errores de la validación.
#
def _initial_response(state, validation, language) -> QueryResponse:
    return QueryResponse(
        status=validation.status,
        validation_status=validation.status,
        answer="",
        language=language,
        warnings=list(validation.warnings),
        limitations=list(validation.limitations),
        errors=list(validation.errors),
        clarification_question=validation.clarification_question,
        validation_issues=list(validation.issues),
        missing_information=list(validation.missing_information),
        pending_operations=list(validation.pending_operations),
        generation=GenerationRecord(status="deterministic"),
        timings_ms={
            **getattr(state, "stage_durations_ms", {}),
            "validator": validation.duration_ms,
        },
    )


#
# Registra en la respuesta el modelo y los tokens usados por el Planner.
#
def _record_planner_usage(response, plan):
    if plan is None:
        return

    response.planner_model = plan.model
    response.planner_input_tokens = plan.input_tokens
    response.planner_output_tokens = plan.output_tokens


#
# Registra en la respuesta la duración de las ejecuciones SQL y Python.
#
def _record_tool_timings(response, state):
    for name, execution in (("sql", state.execution), ("python", state.python_execution)):
        if execution is not None:
            response.timings_ms[name] = execution.duration_ms


#
# Indica si la evidencia o la pregunta cambiaron respecto de lo que se validó.
#
def _evidence_changed(state, validation, plan) -> bool:
    current_fingerprint = evidence_fingerprint(plan, state.execution, state.python_execution)
    if validation.evidence_fingerprint != current_fingerprint:
        return True

    return plan is not None and plan.context.evidence.get("question") != state.question


#
# Indica si la validación no deja evidencia utilizable para responder.
#
def _cannot_answer(validation) -> bool:
    return validation.status not in {"validated", "partial"} or not validation.usable_evidence


#
# Indexa la evidencia utilizable por id, con los resultados derivados antes que las lecturas.
#
def _references_by_relevance(validation, steps):
    def is_read(item):
        return steps[item.operation_id].operation == "read"

    ordered = sorted(validation.usable_evidence, key=is_read)

    return {item.evidence_id: item for item in ordered}


#
# Adjunta a la respuesta las fuentes, las referencias de operaciones, las tablas y los gráficos.
#
def _attach_evidence(response, state, validation, steps):
    usable = validation.usable_evidence

    response.sources = list(validation.sources)
    response.operation_references = [_operation_reference(item, steps) for item in usable]
    response.results = [_result_reference(item, steps) for item in usable if item.kind == "table"]
    response.visualizations = _authorized_visualizations(state, validation)


#
# Construye la referencia a la operación que respalda una evidencia.
#
def _operation_reference(item, steps) -> ResponseOperationReference:
    return ResponseOperationReference(
        evidence_id=item.evidence_id,
        operation_id=item.operation_id,
        tool=item.tool,
        dataset_ids=item.dataset_ids,
        input_operation_ids=steps[item.operation_id].depends_on,
    )


#
# Construye la referencia al resultado tabular de una evidencia, con sus filas y columnas.
#
def _result_reference(item, steps) -> ResponseResultReference:
    location = "execution.operations" if item.tool == "sql" else "python_execution.operations"

    return ResponseResultReference(
        evidence_id=item.evidence_id,
        operation_id=item.operation_id,
        rows=item.rows,
        columns=steps[item.operation_id].output_fields,
        location=location,
    )


#
# Devuelve los gráficos de la ejecución Python que la validación autorizó.
#
def _authorized_visualizations(state, validation):
    authorized_charts = {
        item.operation_id for item in validation.usable_evidence if item.kind == "visualization"
    }
    if not state.python_execution:
        return []

    return [
        chart
        for chart in state.python_execution.visualizations
        if chart.operation_id in authorized_charts
    ]


#
# Adjunta el contexto de redacción a la respuesta y avisa si se resumieron filas extensas.
#
def _attach_context(response, context, language):
    response.generation.context = context

    if context.omitted_rows:
        response.limitations.append(
            "El texto resume los resultados extensos; la tabla conserva todas sus filas."
            if language == "es"
            else "The text summarizes long results; the table keeps every row."
        )


#
# Devuelve la introducción del contexto como segmento, si la hay.
#
def _introduction(context):
    if not context.evidence.get("introduction"):
        return []

    return [AnswerSegment(text=context.evidence["introduction"])]


#
# Verifica que el modelo haya elegido cada oración del catálogo exactamente una vez.
#
def _check_sentence_selection(sentences, catalog):
    selected = [item.sentence_id for item in sentences]
    repeated = len(selected) != len(set(selected))

    if repeated or set(selected) != set(catalog):
        raise LLMProviderError("El modelo omitió, repitió o inventó enunciados de evidencia.")


#
# Indica si una oración cita evidencia distinta de la de su entrada o no autorizada.
#
def _cites_unauthorized_evidence(sentence, entry, refs) -> bool:
    if sentence.evidence_ids != entry["evidence_ids"]:
        return True

    return any(key not in refs for key in sentence.evidence_ids)


#
# Registra el fallo de generación del modelo, con sus tokens y el error correspondiente.
#
def _record_generation_failure(response, error, context):
    response.status = "failed"
    generation = response.generation
    generation.status = "failed"
    generation.model = getattr(error, "model", None) or generation.model

    for field in ("input_tokens", "output_tokens"):
        value = getattr(error, field, None)
        if value is not None:
            setattr(generation, field, value)

    generation.error = _generation_error(context)
    response.errors.append(generation.error)


#
# Crea el error de generación según si el contexto llegó o no a enviarse al modelo.
#
def _generation_error(context) -> ExecutionError:
    if context.sent_to_model:
        return ExecutionError(
            code="unverified_generation",
            message="El modelo no produjo una redacción verificable; no se publica su propuesta.",
        )

    return ExecutionError(
        code="response_context_limit",
        message="La evidencia mínima supera el presupuesto de redacción; no se llamó al modelo.",
    )


#
# Une los enunciados verificados de información faltante para añadirlos a la respuesta.
#
def _missing_information_text(validation):
    return "".join(f" {item.strip()}" for item in validation.missing_information)
