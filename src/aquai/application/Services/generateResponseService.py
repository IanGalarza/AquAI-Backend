import math
from time import perf_counter

from aquai.application.answering.blocked import blocked_explanations, request_gaps
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
    ATTACHMENT_LEADS,
    analytical_request,
    apply_support,
    asks_documents,
    backed_citations,
    cited_lines,
    cited_references,
    claim_variants,
    documental_request,
    documented_claims,
    listed_values,
    pages_text,
    published_segments,
    retry_record,
    review_draft,
    support_passages,
    support_phrases,
)
from aquai.application.answering.readings import number_readings
from aquai.application.answering.segments import answer_text
from aquai.application.answering.verification import Rejection, paraphrase_drift
from aquai.application.answering.wording import file_stem, join_words
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
from aquai.application.matching import (
    asked_reference,
    names_subject,
    reference_operator,
    variable_core,
    variable_variants,
    without_medium,
)
from aquai.application.Rules.evidence import CITATION_YEAR, EXPLANATION_CUES, LITERATURE_CLAUSE

_MIN_SUPPORT = 0.6
# Un ítem de lista comparado con una frase corta de la evidencia necesita más parecido que con una oración: las
# frases cortas se parecen más entre sí, y un ítem inventado del mismo tema llega a 0,68 (calibrado con bge-m3).
_MIN_PHRASE_SUPPORT = 0.7
_UNCOVERED_LITERATURE = {
    "es": "La consulta pide qué dice la bibliografía sobre {topic}, pero los documentos que pude verificar no tratan "
    "ese tema, así que esa parte queda sin responder.",
    "en": "The question asks what the literature says about {topic}, but the documents I could verify do not address "
    "it, so that part remains unanswered.",
}
_UNANSWERED_REFERENCE = {
    "es": "La consulta pide comparar con valores de referencia, pero no pude verificar en los documentos recuperados "
    "un valor comparable con este resultado, así que esa comparación queda sin responder.",
    "en": "The question asks for a comparison with reference values, but I could not verify in the retrieved "
    "documents a value comparable with this result, so that comparison remains unanswered.",
}
_MIN_LISTED_SUPPORT = 0.45
_ALIGNMENT_MARGIN = 0.03
_PAIR_MARGIN = 0.03
_ALIGNED_ROLES = {"documented", "relation"}
_STUDY_SCOPE = "alcance del estudio"
_STUDY_SCOPE_SUFFIX = {"es": ", según el estudio citado", "en": ", according to the cited study"}
_RANK = {"result": 0, "pattern": 1, "meaning": 2, "summary": 3, "chart": 4, "scope": 5, "detail": 6}

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
        "dropped_claims": "Parte de lo que encontré en los documentos no se pudo verificar contra los fragmentos citados y quedó afuera, así que la respuesta puede estar incompleta.",
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
        "dropped_claims": "Part of what I found in the documents could not be verified against the cited fragments and was left out, so the answer may be incomplete.",
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
        document_draft_attempts=2,
    ):
        self._provider = provider
        self._embeddings = embeddings
        self._max_preview_rows = max_preview_rows
        self._draft_attempts = draft_attempts
        self._document_draft_attempts = document_draft_attempts

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
            if not await self._answered_from_documents(response, state, language):
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
        response.reference_values = contrast.references
        notes = [
            *unit_notes(state, language),
            *coverage_notes(state, language),
            *period_notes(state, language),
            *contrast.notes,
        ]
        thresholds = _threshold_notes(state.routing, language)
        missing = []
        if validation.status == "partial":
            missing = [*blocked_explanations(state, language), *request_gaps(state, language)]
            missing = missing or [self._message("partial", language)]
        told = [*notes, *(item.text for item in thresholds), *missing]
        written = await self._written(response, state, context, told, contrast, language)
        body = written if written is not None else [*self._verified_text(context, refs), *_compared(contrast)]
        body = [*(AnswerSegment(kind="note", text=text) for text in missing), *body]

        interpretation = _interpretation_notes(state, body, language, self._message)
        unanswered = [
            *_unanswered_reference(state, contrast, thresholds, response, language),
            *_uncovered_literature(state, body, language),
        ]
        closing = [
            AnswerSegment(kind="note", text=item, document_ids=contrast.note_documents.get(item, []))
            for item in [*notes, *unanswered, *interpretation]
        ]
        if response.status == "validated" and (unanswered or _unexplained(state, interpretation)):
            response.status = "partial"
        self._publish(response, [*body, *thresholds, *closing])
        _cite_notes(response, state, [*([] if written is not None else body), *thresholds, *closing])
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

        closing = _reference_notes(routing, language)
        request = documental_request(
            state,
            language,
            self._provider.context_budget_bytes,
            _previous_answers(state),
            _retaken_analyses(state),
            notes=[item.text for item in closing],
        )
        segments = await self._reviewed_segments(response, request, language, self._document_draft_attempts)
        if segments is None:
            self._publish_unwritten(response, language, closing)
        else:
            response.document_references = cited_references(segments, retrieval.fragments)
            self._publish(response, [*segments, *closing])

        _record_trace(response, state)

        return self._finish(response, started)

    #
    # Responde la parte documental de una consulta mixta cuyo cálculo no se pudo verificar: publica lo que los
    # documentos respaldan y avisa que el cálculo quedó sin hacer y por qué. Devuelve False si no hay documentos
    # para la pregunta o no se pudo redactar con ellos; entonces se publica sólo el motivo.
    #
    async def _answered_from_documents(self, response, state, language) -> bool:
        retrieval = state.document_retrieval
        if retrieval is None or retrieval.purpose != "question" or retrieval.status != "found":
            return False

        request = documental_request(state, language, self._provider.context_budget_bytes, [])
        segments = await self._reviewed_segments(response, request, language, self._draft_attempts)
        _record_trace(response, state)
        if segments is None:
            return False

        reasons = [*blocked_explanations(state, language), *text_measurement_notes(state, language)]
        notes = [AnswerSegment(kind="note", text=text) for text in [self._message("partial", language), *reasons]]
        response.status = "partial"
        response.document_references = cited_references(segments, retrieval.fragments)
        self._publish(response, [notes[0], *segments, *notes[1:]])

        return True

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
        segments = await self._reviewed_segments(response, request, language, self._draft_attempts)
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
    # Pide la redacción, la verifica y reintenta con los problemas encontrados hasta agotar los intentos.
    #
    async def _reviewed_segments(self, response, request, language, attempts):
        generation = response.generation
        generation.model = self._provider.model
        started = perf_counter()
        try:
            reviewed = await self._best_draft(response, request, attempts)
        except LLMProviderError as error:
            _record_generation_error(generation, error, request)
            return None
        finally:
            generation.duration_ms = round((perf_counter() - started) * 1000, 3)

        if not reviewed.accepted:
            return None

        generation.status = "generated"
        published = published_segments(reviewed, request, language)
        if reviewed.dropped:
            published.append(AnswerSegment(kind="note", text=self._message("dropped_claims", language)))

        return published

    #
    # Quita las oraciones que no se parecen a lo que dicen los fragmentos que citan o que pierden o agregan
    # algo respecto de la oración del fragmento que parafrasean; sin embeddings, no publica lo que no auditó.
    #
    async def _audited(self, response, reviewed, request):
        claims = documented_claims(reviewed.accepted)
        if not claims or self._embeddings is None:
            return reviewed

        passages = {id(item[0]): support_passages(item[0], request) for item in claims}
        phrases = {id(segment): support_phrases(passages[id(segment)]) for segment, _ in claims if segment.items}
        texts = list(
            dict.fromkeys(
                [
                    *(variant for _, text in claims for variant in claim_variants(text)),
                    *(line for lines in [*passages.values(), *phrases.values()] for line in lines),
                ]
            )
        )
        try:
            vectors = dict(zip(texts, await self._embeddings.embed(texts), strict=True))
            unsupported = self._unsupported(vectors, claims, passages, request, phrases)
            drifted, scoped = self._drifted(vectors, claims, request, unsupported)
            audited = apply_support(
                reviewed, [*unsupported, *drifted], request.evidence.required_values, scoped
            )
            vectors |= await self._rewritten_vectors(audited, vectors)
        except LLMProviderError:
            return self._unaudited(response, reviewed, claims)

        cited = backed_citations(
            audited, request, lambda text, lines: _best_support(vectors, text, lines), _MIN_SUPPORT
        )
        _record_rejections(response, _new_rejections(reviewed, cited))

        return cited

    #
    # Devuelve las oraciones que no se parecen lo suficiente a lo que citan.
    #
    @staticmethod
    def _unsupported(vectors, claims, passages, request, phrases=None) -> list:
        phrases = phrases or {}
        return [
            (segment, text)
            for segment, text in claims
            if not _resembles(vectors, text, passages[id(segment)], _MIN_SUPPORT)
            and not _phrase_supported(vectors, segment, text, phrases)
            and not (
                listed_values(text, segment, request)
                and _resembles(vectors, text, passages[id(segment)], _MIN_LISTED_SUPPORT)
            )
        ]

    #
    # Devuelve las oraciones con respaldo que, frente a la oración del fragmento que parafrasean, pierden una
    # acotación o agregan una causa, una generalidad o una conclusión estadística. Una oración que sólo pierde el
    # alcance del estudio («in this study») no se quita: se acota con «según el estudio citado», que es lo que
    # expresaba, y se devuelve aparte como reescritura.
    #
    @staticmethod
    def _drifted(vectors, claims, request, unsupported) -> tuple[list, dict]:
        rejected = {(id(segment), text) for segment, text in unsupported}
        language = request.evidence.language
        drifted, scoped = [], {}
        for segment, text in claims:
            if segment.role not in _ALIGNED_ROLES or (id(segment), text) in rejected:
                continue

            lines = cited_lines(segment, request)
            numbered, contributing = _numbered_lines(CITATION_YEAR.sub(" ", text), lines, language)
            aligned = (
                [*_closest(vectors, text, numbered), *contributing]
                if numbered
                else _aligned_lines(vectors, text, lines)
            )
            if not aligned:
                continue

            check = _drift_check(vectors, segment, text, aligned, request)
            problem = check(text)
            rewritten = _scoped_claim(text, language) if problem is not None and problem.family == _STUDY_SCOPE else None
            if rewritten is not None and check(rewritten) is None:
                scoped[(id(segment), text)] = rewritten
            elif problem is not None:
                drifted.append((segment, text, problem))

        return drifted, scoped

    #
    # Quita las oraciones documentales que no se pudieron auditar, sin descartar lo ya verificado.
    #
    @staticmethod
    def _unaudited(response, reviewed, claims):
        unaudited = [(segment, text, Rejection("unaudited_claim", text[:120])) for segment, text in claims]
        audited = apply_support(reviewed, unaudited)
        _record_rejections(response, _new_rejections(reviewed, audited))

        return audited

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
    async def _best_draft(self, response, request, attempts):
        generation = response.generation
        record = request.record
        best = None
        for _ in range(attempts):
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
            reviewed = await self._audited(response, reviewed, request)
            if best is None or reviewed.score(request.evidence) > best.score(request.evidence):
                best = reviewed
            if reviewed.sufficient(request.evidence):
                return best

            record = retry_record(request, reviewed)

        return best

    #
    # Devuelve los enunciados verificados del catálogo, sin redacción del modelo.
    #
    def _verified_text(self, context, refs):
        catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
        entries = [(entry, entry["direct"]) for entry in self._ordered_entries(catalog, refs)]
        lead = ATTACHMENT_LEADS[context.evidence["language"]]

        return [*_introduction(context), *self._segments(entries, lead)]

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
    def _publish_unwritten(self, response, language, closing=()):
        technical = response.generation.error is not None
        response.status = "failed" if technical else "insufficient_coverage"
        response.generation.status = "failed" if technical else "deterministic"
        self._publish(response, [AnswerSegment(text=self._message("unwritten", language)), *closing])

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
    # Agrupa las oraciones consecutivas con la misma evidencia en segmentos de respuesta. Una oración que ya se
    # publicó (dos salidas con el mismo resultado) no se repite: su tabla, si la tiene, va con el rótulo neutro.
    #
    @staticmethod
    def _segments(entries, lead):
        segments = []

        for entry, text in entries:
            said = next((item for item in segments if text in item.text), None)
            attachment = entry.get("attachment")
            if said is not None and attachment and attachment != said.attachment:
                segments.append(AnswerSegment(text=lead, evidence_ids=list(entry["evidence_ids"]), attachment=attachment))
            if said is not None:
                continue

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
# Devuelve el aviso de un resultado que no se pudo interpretar con los documentos, sólo si se pidió.
#
def _interpretation_notes(state, segments, language, message):
    retrieval = state.document_retrieval
    question = state.routing.original_question if state.routing else state.question
    if retrieval is None or retrieval.purpose != "enrichment" or retrieval.status == "skipped":
        return []
    if not asks_documents(state, question):
        return []
    if any(item.role in _ALIGNED_ROLES and item.document_ids for item in segments):
        return []
    if retrieval.status == "unavailable":
        return [message("interpretation_unavailable", language)]
    if retrieval.status == "found":
        return [message("interpretation_unverified", language)]

    return [message("no_interpretation", language)]


#
# Devuelve los contrastes verificados con referencias documentales como párrafos citados: si la redacción falla,
# el texto determinístico no debe perderlos.
#
def _compared(contrast) -> list[AnswerSegment]:
    return [
        AnswerSegment(
            kind="paragraph",
            role="relation",
            text=item.text,
            evidence_ids=[item.evidence_id],
            document_ids=[item.document_id],
        )
        for item in contrast.comparisons
    ]


#
# Devuelve el aviso de un contraste con valores de referencia que la consulta pidió y la respuesta no hizo ni
# explicó: sin él, una respuesta a medias se publicaría como completa.
#
def _unanswered_reference(state, contrast, thresholds, response, language) -> list[str]:
    question = state.routing.original_question if state.routing else state.question
    explained = contrast.comparisons or contrast.notes or thresholds or response.reference_lines
    if asked_reference(question) is None or explained:
        return []

    return [_UNANSWERED_REFERENCE[language]]


#
# Devuelve el aviso de un pedido a la bibliografía sobre variables del análisis («qué dice la bibliografía
# sobre el efecto de la turbidez en las cianobacterias») que ningún párrafo documentado trata: citar documentos
# de otro tema no lo responde.
#
def _uncovered_literature(state, segments, language) -> list[str]:
    question = state.routing.original_question if state.routing else state.question
    clauses = [match.group() for match in LITERATURE_CLAUSE.finditer(question)]
    mentions = [item.text for item in state.routing.data_mentions if item.kind == "variable"] if state.routing else []
    terms = {text: _variable_terms(text) for text in mentions}
    topic = [text for text, names in terms.items() if any(names_subject(clause, names) for clause in clauses)]
    documented = [item.text for item in segments if item.role in _ALIGNED_ROLES]
    if not topic or any(all(names_subject(text, terms[item]) for item in topic) for text in documented):
        return []

    return [_UNCOVERED_LITERATURE[language].format(topic=join_words(topic, language))]


#
# Devuelve los nombres con que un texto puede tratar una variable: sus sinónimos, sin el medio ni la
# estadística, y los de su primera palabra («cianobacterias» por «cianobacterias totales»).
#
def _variable_terms(mention) -> list[str]:
    core = without_medium(variable_core(mention)) or mention
    head = core.split()[0] if core.split() else core

    return list(dict.fromkeys([*variable_variants(core), *variable_variants(head)]))


#
# Indica si la consulta pidió explicar o interpretar con la bibliografía y no quedó ningún párrafo documentado:
# los resultados se conservan, pero la respuesta es parcial.
#
def _unexplained(state, interpretation) -> bool:
    question = state.routing.original_question if state.routing else state.question
    named = state.routing is not None and bool(state.routing.document_ids)

    return bool(interpretation) and bool(
        named or EXPLANATION_CUES.search(question) or LITERATURE_CLAUSE.search(question)
    )


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
    trace.unselected = retrieval.unselected
    contrast = getattr(state, "document_contrast", None)
    trace.warnings = [*retrieval.warnings, *(contrast.warnings if contrast else [])]
    if contrast is not None:
        trace.rejected_segments.extend(contrast.rejected)
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
    return _best_support(vectors, text, passages) >= threshold


#
# Indica si una oración, tal cual o sin la frase que atribuye su fuente, se parece a su evidencia.
#
def _resembles(vectors, text, passages, threshold) -> bool:
    return any(_supported(vectors, variant, passages, threshold) for variant in claim_variants(text))


#
# Indica si el ítem de una lista se parece lo suficiente a una frase de su evidencia, que es lo que un ítem suele
# traducir («contacto total del cuerpo» es «whole-body contact»); una oración se compara sólo con oraciones.
#
def _phrase_supported(vectors, segment, text, phrases) -> bool:
    return text in segment.items and _resembles(vectors, text, phrases.get(id(segment), []), _MIN_PHRASE_SUPPORT)


#
# Devuelve el mayor parecido entre una oración y las de su evidencia.
#
def _best_support(vectors, text, passages) -> float:
    claim = vectors[text]

    return max((_cosine(claim, vectors[line]) for line in passages if line in vectors), default=0.0)


#
# Devuelve la oración del fragmento que una oración parafrasea y las de ese fragmento casi igual de
# parecidas, con su sección y los pares con sus vecinas; nada si ninguna se le parece lo suficiente.
#
def _aligned_lines(vectors, text, lines) -> list[tuple[str, str, tuple]]:
    claim = vectors[text]
    scored = [
        (_cosine(claim, vectors[line]), key, line, heading, pairs)
        for key, line, heading, pairs in lines
        if line in vectors
    ]
    if not scored:
        return []

    best, document, *_ = max(scored, key=lambda item: item[0])
    if best < _MIN_LISTED_SUPPORT:
        return []

    return [
        (line, heading, pairs)
        for score, key, line, heading, pairs in scored
        if key == document and score >= best - _ALIGNMENT_MARGIN
    ]


#
# Devuelve las oraciones de los fragmentos de donde salen las cifras de una oración, con su sección y los
# pares con sus vecinas: las que contienen más de sus números y, aparte, las que aportan las cifras que esas
# no tienen; una oración que funde dos frases debe conservar las acotaciones de ambas. El año de una cita
# («Sathicq et al. (2014)») se quita antes: no es una cifra del hallazgo y la alinearía con los datos de
# publicación del fragmento.
#
def _numbered_lines(text, lines, language) -> tuple[list, list]:
    wanted = [reading.values for reading in number_readings(text, language)]
    if not wanted:
        return [], []

    found = [set().union(*(reading.values for reading in number_readings(line))) for _, line, _, _ in lines]
    counts = [sum(bool(values & numbers) for values in wanted) for numbers in found]
    most = max(counts, default=0)
    if not most:
        return [], []

    primary = [index for index, count in enumerate(counts) if count == most]
    covered = {position for index in primary for position, values in enumerate(wanted) if values & found[index]}
    contributing = []
    for position, values in enumerate(wanted):
        holder = next((index for index, numbers in enumerate(found) if values & numbers), None)
        if position in covered or holder is None:
            continue
        contributing.append(holder)
        covered |= {item for item, other in enumerate(wanted) if other & found[holder]}

    return [lines[index][1:] for index in primary], [lines[index][1:] for index in contributing]


#
# Devuelve, entre las oraciones que contienen las cifras, la que más se parece a la oración y las casi igual
# de parecidas: una cifra común puede figurar en oraciones que hablan de otra cosa.
#
def _closest(vectors, text, lines) -> list[tuple[str, str, tuple]]:
    claim = vectors[text]
    scored = [(_cosine(claim, vectors[line]) if line in vectors else 0.0, item) for item in lines for line in item[:1]]
    best = max(score for score, _ in scored)

    return [item for score, item in scored if score >= best - _ALIGNMENT_MARGIN]


#
# Devuelve la revisión de lo que una oración pierde o agrega frente a las oraciones del fragmento que parafrasea,
# aplicable también a una reescritura suya: el pasaje se elige con la oración original. Las acotaciones de otra
# cláusula sólo se ignoran si todos los fragmentos citados están en el idioma de la respuesta.
#
def _drift_check(vectors, segment, text, aligned, request):
    evidence = request.evidence
    earlier = _earlier_text(segment, text)
    relation = segment.role == "relation"
    same_language = all(evidence.document_languages.get(key) == evidence.language for key in segment.document_ids)
    passages = [(_paraphrased(vectors, text, line, pairs), heading) for line, heading, pairs in aligned]

    def check(claim):
        problems = (
            paraphrase_drift(claim, passage, heading, earlier, evidence.question, relation, same_language)
            for passage, heading in passages
        )
        return next((item for item in problems if item is not None), None)

    return check


#
# Devuelve la oración acotada al estudio que la sostiene, con el cierre que tenía.
#
def _scoped_claim(text, language) -> str:
    body = text.rstrip()
    end = body[-1] if body[-1:] in {".", "!", "?"} else ""
    core = body[:-1] if end else body

    return f"{core}{_STUDY_SCOPE_SUFFIX[language]}{end or '.'}"


#
# Devuelve lo que una oración parafrasea: la oración del fragmento o, si se le parece claramente más, el
# par que forma con una vecina porque la resume entera; sólo contra eso se revisa qué pierde o agrega.
#
def _paraphrased(vectors, text, line, pairs) -> str:
    claim = vectors[text]
    alone = _cosine(claim, vectors[line]) if line in vectors else 0.0
    scored = [(_cosine(claim, vectors[item]), item) for item in pairs if item in vectors]
    score, pair = max(scored, default=(0.0, line))

    return pair if score >= alone + _PAIR_MARGIN else line


#
# Devuelve lo que el párrafo dice antes de una de sus oraciones.
#
def _earlier_text(segment, text) -> str:
    position = segment.text.find(text)

    return segment.text[:position] if position > 0 else ""


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
# Devuelve el aviso de los umbrales documentados con que se calculó, con su uso, su condición y su cita.
#
def _threshold_notes(routing, language) -> list[AnswerSegment]:
    options = routing.reference_options if routing is not None else []
    if not options or routing.route != "data" or not routing.reference_comparison:
        return []

    listed = join_words(options, language)
    if language == "en":
        text = (
            f"The threshold I used is the reference value the document gives: {listed}."
            if len(options) == 1
            else f"The thresholds I used are the reference values the documents give: {listed}."
        )
    else:
        text = (
            f"El umbral que usé es el valor de referencia que da el documento: {listed}."
            if len(options) == 1
            else f"Los umbrales que usé son los valores de referencia que dan los documentos: {listed}."
        )

    return [AnswerSegment(kind="note", text=text, document_ids=list(routing.reference_documents))]


#
# Agrega a las referencias de la respuesta los fragmentos que citan las notas: los umbrales usados y las
# referencias que no se compararon.
#
def _cite_notes(response, state, notes):
    retrieval = state.document_retrieval
    cited = [item for item in notes if item.document_ids]
    if not cited or retrieval is None:
        return

    known = {item.evidence_id for item in response.document_references}
    added = [item for item in cited_references(cited, retrieval.fragments) if item.evidence_id not in known]
    response.document_references = [*response.document_references, *added]


#
# Devuelve el aviso de una comparación con una referencia documental que no se pudo calcular: qué valores dan
# los documentos para elegir uno, o que no hay uno verificable.
#
def _reference_notes(routing, language) -> list[AnswerSegment]:
    if not routing.reference_comparison:
        return []

    options = routing.reference_options
    if not options:
        text = (
            "No encontré en los documentos un valor de referencia verificable, en la misma unidad que estos "
            "datos, con el cual hacer la comparación."
            if language == "es"
            else "I found no verifiable reference value in the documents, in the same unit as these data, "
            "to make the comparison."
        )
        return [AnswerSegment(kind="note", text=text)]

    listed = join_words(options, language)
    example = options[0].split(" (")[0]
    below = reference_operator(routing.original_question) == "lt"
    compared = ("están por debajo de" if below else "superan") if language == "es" else ("are below" if below else "exceed")
    text = (
        f"Para hacer el cálculo con los datos necesito un único valor: los documentos dan {listed}. "
        f"Decime cuál querés usar, por ejemplo preguntando cuántas muestras {compared} {example}."
        if language == "es"
        else f"To run the calculation on the data I need a single value: the documents give {listed}. "
        f"Tell me which one to use, for example by asking how many samples {compared} {example}."
    )

    return [AnswerSegment(kind="note", text=text)]


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
