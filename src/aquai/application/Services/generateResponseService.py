import math
from time import perf_counter

from aquai.application.answering.blocked import blocked_explanations
from aquai.application.answering.context import (
    build_response_context,
    response_language,
)
from aquai.application.answering.coverage import (
    coverage_notes,
    period_notes,
    text_measurement_notes,
    unit_notes,
)
from aquai.application.answering.drafting import (
    analytical_request,
    apply_support,
    backed_citations,
    cited_references,
    documental_request,
    documented_claims,
    listed_values,
    pages_text,
    published_segments,
    retry_record,
    review_draft,
    support_passages,
)
from aquai.application.answering.segments import answer_text
from aquai.application.answering.wording import file_stem
from aquai.application.contracts.answering import (
    AnswerSegment,
    DocumentContrast,
    DocumentReference,
    DocumentTrace,
    GenerationRecord,
    QueryResponse,
    QueryResponseState,
    ResponseOperationReference,
    ResponseResultReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.execution import evidence_fingerprint
from aquai.application.interfaces.answering import ResponseProvider, TextEmbeddings

_MIN_SUPPORT = 0.7
_MIN_LISTED_SUPPORT = 0.45
_RANK = {"result": 0, "meaning": 1, "summary": 2, "chart": 3, "scope": 4, "detail": 5}

_MESSAGES = {
    "es": {
        "validated": "",
        "partial": "Pude responder sólo una parte de la consulta; el resto no se pudo calcular de forma verificable.",
        "needs_clarification": "Necesito que me aclares algo para continuar.",
        "insufficient_coverage": "No encontré datos suficientes para responder esta pregunta en las fuentes que revisé. Esto no significa que la información no exista: puede estar en otra fuente o con otro nombre.",
        "failed": "No pude armar un análisis confiable para esta pregunta con los datos disponibles. Probá reformularla indicando la variable, el sitio o la fuente que querés usar.",
        "blocked": "No pude completar el análisis con los datos disponibles.",
        "no_documents": "No encontré en los documentos cargados información suficiente para responder esta pregunta.",
        "documents_unavailable": "No pude consultar los documentos en este momento. Probá de nuevo en un rato.",
        "unwritten": "Encontré documentos relacionados, pero no pude redactar una respuesta verificable con ellos.",
        "no_interpretation": "No encontré en los documentos cargados evidencia específica para interpretar este resultado.",
        "interpretation_unavailable": "No pude consultar los documentos para interpretar este resultado.",
        "interpretation_unverified": "Encontré documentos relacionados, pero no pude incorporar su interpretación de forma verificable.",
    },
    "en": {
        "validated": "",
        "partial": "I could only answer part of the question; the rest could not be calculated in a verifiable way.",
        "needs_clarification": "I need you to clarify something to continue.",
        "insufficient_coverage": "I did not find enough data to answer this question in the sources I reviewed. This does not mean the information does not exist: it may be in another source or under another name.",
        "failed": "I could not build a reliable analysis for this question with the available data. Try rephrasing it with the variable, site or source you want to use.",
        "blocked": "I could not complete the analysis with the available data.",
        "no_documents": "I did not find enough information in the loaded documents to answer this question.",
        "documents_unavailable": "I could not query the documents right now. Please try again shortly.",
        "unwritten": "I found related documents, but I could not write a verifiable answer from them.",
        "no_interpretation": "I found no specific evidence in the loaded documents to interpret this result.",
        "interpretation_unavailable": "I could not query the documents to interpret this result.",
        "interpretation_unverified": "I found related documents, but I could not include their interpretation in a verifiable way.",
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
    # Guarda el proveedor de redacción y los límites de filas y de intentos.
    #
    def __init__(
        self,
        provider: ResponseProvider,
        *,
        embeddings: TextEmbeddings | None = None,
        max_preview_rows=3,
        draft_attempts=2,
    ):
        self._provider = provider
        self._embeddings = embeddings
        self._max_preview_rows = max_preview_rows
        self._draft_attempts = draft_attempts

    #
    # Genera la respuesta final a partir de la validación, la evidencia y los documentos del turno.
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

        contrast = state.document_contrast or DocumentContrast()
        notes = [
            *unit_notes(state, language),
            *coverage_notes(state, language),
            *period_notes(state, language),
            *contrast.notes,
        ]
        written = await self._written(response, state, context, notes, contrast, language)
        body = written if written is not None else self._verified_text(context, refs)
        if validation.status == "partial":
            missing = blocked_explanations(state, language) or [self._message("partial", language)]
            body = [*(AnswerSegment(kind="note", text=text) for text in missing), *body]

        closing = [*notes, *_interpretation_notes(state, response, written, language, self._message)]
        self._publish(response, [*body, *(AnswerSegment(kind="note", text=item) for item in closing)])
        _record_trace(response, state)

        return self._finish(response, started)

    #
    # Genera la respuesta de una pregunta documental a partir de los fragmentos recuperados.
    #
    async def generate_documental(self, state: QueryResponseState) -> QueryResponse:
        started = perf_counter()
        routing = state.routing
        language = routing.language
        retrieval = state.document_retrieval
        response = _document_response(state, language)

        if routing.asks_citations:
            self._publish_citations(response, state, language)
            return self._finish(response, started)

        if retrieval is None or retrieval.status != "found":
            self._publish_without_documents(response, retrieval, language)
            _record_trace(response, state)
            return self._finish(response, started)

        request = documental_request(
            state,
            language,
            self._provider.context_budget_bytes,
            _previous_answers(state),
            _retaken_analyses(state),
        )
        segments = await self._reviewed_segments(response, request, language)
        if segments is None:
            self._publish_unwritten(response, language)
        else:
            response.document_references = cited_references(segments, retrieval.fragments)
            self._publish(response, segments)

        _record_trace(response, state)

        return self._finish(response, started)

    #
    # Redacta una respuesta analítica; si no se puede verificar devuelve None.
    #
    async def _written(self, response, state, context, notes, contrast, language):
        if state.routing is not None and state.routing.requested_period:
            return None

        request = analytical_request(
            state,
            context,
            language,
            self._provider.context_budget_bytes,
            notes=notes,
            comparisons=contrast.comparisons,
            ambiguities=contrast.hypotheses,
        )
        segments = await self._reviewed_segments(response, request, language)
        if segments is None:
            return None

        lines = _authorized_lines(response, contrast)
        retrieval = state.document_retrieval
        fragments = retrieval.fragments if retrieval is not None else []
        response.document_references = cited_references(
            segments, fragments, [item.evidence_id for item in lines]
        )
        response.reference_values = contrast.references
        response.reference_lines = lines
        response.hypotheses = contrast.hypotheses

        return segments

    #
    # Pide la redacción, la verifica y reintenta una vez con los problemas encontrados.
    #
    async def _reviewed_segments(self, response, request, language):
        generation = response.generation
        generation.model = self._provider.model
        started = perf_counter()
        try:
            reviewed = await self._best_draft(response, request)
            reviewed = await self._audited(response, reviewed, request)
        except LLMProviderError as error:
            _record_generation_error(generation, error, request)
            return None
        finally:
            generation.duration_ms = round((perf_counter() - started) * 1000, 3)

        if not reviewed.accepted:
            return None

        generation.status = "generated"

        return published_segments(reviewed, request, language)

    #
    # Quita las oraciones que no se parecen a lo que dicen los fragmentos que citan.
    #
    async def _audited(self, response, reviewed, request):
        claims = documented_claims(reviewed.accepted)
        if not claims or self._embeddings is None:
            return reviewed

        passages = {id(item[0]): support_passages(item[0], request) for item in claims}
        texts = list(
            dict.fromkeys(
                [*(text for _, text in claims), *(line for lines in passages.values() for line in lines)]
            )
        )
        vectors = dict(zip(texts, await self._embeddings.embed(texts), strict=True))
        unsupported = [
            (segment, text)
            for segment, text in claims
            if not _supported(vectors, text, passages[id(segment)], _MIN_SUPPORT)
            and not (
                listed_values(text, segment, request)
                and _supported(vectors, text, passages[id(segment)], _MIN_LISTED_SUPPORT)
            )
        ]
        audited = apply_support(reviewed, unsupported)
        vectors |= await self._rewritten_vectors(audited, vectors)
        cited = backed_citations(
            audited, request, lambda text, lines: _supported(vectors, text, lines, _MIN_SUPPORT)
        )
        _record_rejections(response, _new_rejections(reviewed, cited))

        return cited

    #
    # Calcula los vectores de las oraciones que cambiaron al quitar lo que no tenía respaldo.
    #
    async def _rewritten_vectors(self, reviewed, vectors):
        claims = documented_claims(reviewed.accepted)
        missing = list(dict.fromkeys(text for _, text in claims if text not in vectors))
        if not missing:
            return {}

        return dict(zip(missing, await self._embeddings.embed(missing), strict=True))

    #
    # Devuelve el borrador con menos problemas entre los intentos permitidos.
    #
    async def _best_draft(self, response, request):
        generation = response.generation
        record = request.record
        best = None
        for _ in range(self._draft_attempts):
            record.sent_to_model = True
            generation.context = record
            try:
                reply = await self._provider.generate(record)
            except LLMProviderError:
                if best is None:
                    raise
                return best

            _add_tokens(generation, reply)
            reviewed = review_draft(reply.draft, request.evidence)
            _record_rejections(response, reviewed)
            if best is None or reviewed.score > best.score:
                best = reviewed
            if reviewed.sufficient(request.evidence):
                return reviewed

            record = retry_record(request, reviewed)

        return best

    #
    # Devuelve los enunciados verificados del catálogo, sin redacción del modelo.
    #
    def _verified_text(self, context, refs):
        catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
        entries = [(entry, entry["direct"]) for entry in self._ordered_entries(catalog, refs)]

        return [*_introduction(context), *self._segments(entries)]

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
        reasons = [*blocked_explanations(state, language), *text_measurement_notes(state, language)]
        explanation = AnswerSegment(
            text=self._unanswered(validation, language, explained=bool(reasons))
        )
        notes = [AnswerSegment(kind="note", text=text) for text in reasons]

        self._publish(response, [explanation, *notes])

    #
    # Responde de dónde salió la respuesta anterior con sus citas guardadas, sin consultar nada.
    #
    def _publish_citations(self, response, state, language):
        references = _previous_references(state)
        response.document_references = references
        lead = "La respuesta anterior se apoya en:" if language == "es" else "The previous answer relies on:"
        self._publish(
            response,
            [
                AnswerSegment(
                    role="documented",
                    text=lead,
                    items=[_citation_text(item) for item in references],
                    document_ids=[item.evidence_id for item in references],
                )
            ],
        )

    #
    # Publica el aviso de que no se pudo redactar una respuesta documental verificable.
    #
    def _publish_unwritten(self, response, language):
        technical = response.generation.error is not None
        response.status = "failed" if technical else "insufficient_coverage"
        response.generation.status = "failed" if technical else "deterministic"
        self._publish(response, [AnswerSegment(text=self._message("unwritten", language))])

    #
    # Publica el aviso de una pregunta documental sin evidencia suficiente.
    #
    def _publish_without_documents(self, response, retrieval, language):
        unavailable = retrieval is not None and retrieval.status == "unavailable"
        key = "documents_unavailable" if unavailable else "no_documents"
        response.status = "failed" if unavailable else "insufficient_coverage"
        self._publish(response, [AnswerSegment(text=self._message(key, language))])

    #
    # Ordena las oraciones del catálogo según la relevancia de su evidencia.
    #
    @staticmethod
    def _ordered_entries(catalog, refs):
        order = {key: index for index, key in enumerate(refs)}

        return sorted(catalog.values(), key=lambda entry: _position(entry, order))

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
# Devuelve las líneas de referencia de los gráficos autorizados.
#
def _authorized_lines(response, contrast):
    charts = {item.operation_id for item in response.visualizations}

    return [item for item in contrast.lines if item.operation_id in charts]


#
# Devuelve el aviso de un resultado que no se pudo interpretar con los documentos.
#
def _interpretation_notes(state, response, written, language, message):
    retrieval = state.document_retrieval
    if retrieval is None or retrieval.purpose != "enrichment" or retrieval.status == "skipped":
        return []
    if retrieval.status == "unavailable":
        return [message("interpretation_unavailable", language)]
    if retrieval.status == "found" and (written is None or not response.document_references):
        return [message("interpretation_unverified", language)]
    if not response.document_references:
        return [message("no_interpretation", language)]

    return []


#
# Registra las decisiones de la recuperación documental en la respuesta.
#
def _record_trace(response, state):
    retrieval = state.document_retrieval
    if retrieval is None:
        return

    trace = response.document_trace or DocumentTrace(
        purpose=retrieval.purpose, status=retrieval.status
    )
    trace.purpose = retrieval.purpose
    trace.status = retrieval.status
    trace.queries = retrieval.queries
    trace.candidates = retrieval.candidates
    trace.discarded = retrieval.discarded
    trace.fragments = len(retrieval.fragments)
    trace.selected = [
        {
            "evidence_id": item.evidence_id,
            "file_name": item.file_name,
            "page": item.page_start,
            "doubts": item.doubts,
        }
        for item in retrieval.fragments
    ]
    trace.warnings = retrieval.warnings
    response.document_trace = trace


#
# Registra en la traza los párrafos que la verificación rechazó.
#
def _record_rejections(response, reviewed):
    retrieval_trace = response.document_trace or DocumentTrace(purpose="enrichment", status="skipped")
    retrieval_trace.rejected_segments.extend(
        {"role": role, "code": problem.code, "detail": problem.detail[:120]}
        for role, problem in reviewed.rejections
    )
    retrieval_trace.rejected_segments.extend(
        {"role": "result", "code": "missing_fact", "detail": key} for key in reviewed.missing_facts
    )
    response.document_trace = retrieval_trace


#
# Indica si una oración se parece lo suficiente a alguna oración de su evidencia.
#
def _supported(vectors, text, passages, threshold) -> bool:
    claim = vectors[text]

    return any(_cosine(claim, vectors[line]) >= threshold for line in passages)


#
# Devuelve el coseno entre dos vectores.
#
def _cosine(left, right) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))

    return dot / norm if norm else 0.0


#
# Devuelve una revisión sólo con los rechazos que agregó la auditoría.
#
def _new_rejections(before, after):
    return type(after)([], after.rejections[len(before.rejections) :], [])


#
# Suma los tokens de un intento de redacción.
#
def _add_tokens(generation, reply):
    generation.model = reply.model
    generation.input_tokens = (generation.input_tokens or 0) + (reply.input_tokens or 0)
    generation.output_tokens = (generation.output_tokens or 0) + (reply.output_tokens or 0)


#
# Registra que el modelo no produjo una redacción, sin alterar el resultado verificado.
#
def _record_generation_error(generation, error, request):
    for field in ("input_tokens", "output_tokens"):
        value = getattr(error, field, None)
        if value is not None:
            setattr(generation, field, value)

    sent = request.record.sent_to_model
    generation.error = ExecutionError(
        code="unverified_generation" if sent else "response_context_limit",
        message=(
            "El modelo no produjo una redacción verificable; se publican los resultados verificados."
            if sent
            else "La evidencia supera el presupuesto de redacción; no se llamó al modelo."
        ),
    )


#
# Crea la respuesta inicial de una pregunta documental.
#
def _document_response(state, language) -> QueryResponse:
    return QueryResponse(
        response_type="document",
        status="answered",
        validation_status=None,
        answer="",
        language=language,
        routing=state.routing,
        generation=GenerationRecord(status="deterministic"),
        timings_ms=dict(state.stage_durations_ms),
    )


#
# Devuelve lo ya respondido con documentos en los mensajes que la pregunta retoma, con sus citas.
#
def _previous_answers(state) -> list[str]:
    cited = set(state.routing.history_message_ids)

    return [
        " ".join([*_verified_paragraphs(item), *(_cited_label(document) for document in item.documents)])
        for item in state.conversation_history
        if item.message_id in cited and item.role == "assistant" and item.analysis is None
    ]


#
# Devuelve los análisis que la pregunta retoma, cuyos resultados el redactor recibe como hechos.
#
def _retaken_analyses(state) -> list:
    cited = set(state.routing.history_message_ids)

    return [
        item
        for item in state.conversation_history
        if item.message_id in cited and item.analysis is not None
    ]


#
# Devuelve los párrafos de resultado de un mensaje, sin sus hipótesis ni interpretaciones.
#
def _verified_paragraphs(message) -> list[str]:
    results = [
        item.get("text", "")
        for item in message.segments
        if item.get("role") == "result" or item.get("evidence_ids")
    ]

    return results or [message.text]


#
# Devuelve las referencias de los mensajes que la pregunta retoma.
#
def _previous_references(state) -> list[DocumentReference]:
    cited = set(state.routing.history_message_ids)
    stored = {
        item["evidence_id"]: item
        for message in state.conversation_history
        if message.message_id in cited
        for item in message.documents
    }

    return [DocumentReference.model_validate(item) for item in stored.values()]


#
# Devuelve la cita legible de una referencia: archivo, páginas y sección.
#
def _citation_text(reference) -> str:
    pages = pages_text(reference.page_start, reference.page_end)
    section = reference.section[-1] if reference.section else ""

    return ", ".join(item for item in (file_stem(reference.file_name), pages, section) if item)


#
# Devuelve el archivo y las páginas de un fragmento citado en un turno anterior.
#
def _cited_label(document) -> str:
    return f"{document.get('file_name')} p. {document.get('page_start')}-{document.get('page_end')}"


#
# Une los enunciados verificados de información faltante para añadirlos a la respuesta.
#
def _missing_information_text(validation):
    return "".join(f" {item.strip()}" for item in validation.missing_information)
