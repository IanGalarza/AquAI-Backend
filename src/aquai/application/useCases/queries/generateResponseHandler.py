"""Respuesta fundamentada: el LLM organiza enunciados; Python conserva los hechos."""

from time import perf_counter

from aquai.application.answering.blocked import blocked_explanations
from aquai.application.answering.context import build_response_context, response_language
from aquai.application.answering.segments import answer_text
from aquai.application.contracts.answering import (
    AnswerSegment, GenerationRecord, QueryResponse, QueryResponseState, ResponseOperationReference,
    ResponseResultReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.execution import evidence_fingerprint
from aquai.application.interfaces.answering import ResponseProvider


# Cada evidencia presenta primero su resultado y después su alcance y detalles.
_RANK = {"result": 0, "meaning": 1, "summary": 2, "chart": 3, "scope": 4, "detail": 5}


def _position(entry, order):
    return order.get(entry["evidence_ids"][0], len(order)), _RANK.get(entry.get("kind"), len(_RANK))


class GenerateResponseHandler:
    def __init__(self, provider: ResponseProvider, *, max_preview_rows=3, deterministic_sentences=24):
        self._provider, self._max_preview_rows = provider, max_preview_rows
        self._deterministic_sentences = deterministic_sentences

    async def handle(self, state: QueryResponseState) -> QueryResponse:
        started = perf_counter()
        validation = state.validation
        language = response_language(state.question, state.response_language)
        if validation is None:
            raise ValueError("Response Generator requiere el resultado del Validator.")
        response = QueryResponse(status=validation.status, validation_status=validation.status,
            answer="", language=language, warnings=list(validation.warnings), limitations=list(validation.limitations),
            errors=list(validation.errors), clarification_question=validation.clarification_question,
            validation_issues=list(validation.issues), missing_information=list(validation.missing_information),
            pending_operations=list(validation.pending_operations), generation=GenerationRecord(status="deterministic"),
            timings_ms={**getattr(state, "stage_durations_ms", {}), "validator": validation.duration_ms})
        plan = state.planning
        if plan is not None:
            response.planner_model, response.planner_input_tokens, response.planner_output_tokens = plan.model, plan.input_tokens, plan.output_tokens
        for name, item in (("sql", state.execution), ("python", state.python_execution)):
            if item is not None:
                response.timings_ms[name] = item.duration_ms
        if (validation.evidence_fingerprint != evidence_fingerprint(plan, state.execution, state.python_execution)
                or (plan is not None and plan.context.evidence.get("question") != state.question)):
            response.status = "failed"
            response.errors.append(ExecutionError(code="evidence_changed", message="La evidencia cambió después de su validación."))
            self._publish(response, [AnswerSegment(text=self._message("failed", language))])
            return self._finish(response, started)
        if validation.status not in {"validated", "partial"} or not validation.usable_evidence:
            reasons = blocked_explanations(state, language)
            self._publish(response, [AnswerSegment(text=self._unanswered(validation, language, explained=bool(reasons))),
                                     *(AnswerSegment(kind="note", text=text) for text in reasons)])
            return self._finish(response, started)

        steps = {step.id: step for step in plan.operations}
        # Los resultados derivados (promedio, máximo, último) responden; las filas leídas los respaldan.
        refs = {item.evidence_id: item for item in sorted(
            validation.usable_evidence, key=lambda item: steps[item.operation_id].operation == "read")}
        response.sources = list(validation.sources)
        response.operation_references = [ResponseOperationReference(evidence_id=item.evidence_id,
            operation_id=item.operation_id, tool=item.tool, dataset_ids=item.dataset_ids,
            input_operation_ids=steps[item.operation_id].depends_on) for item in validation.usable_evidence]
        response.results = [ResponseResultReference(evidence_id=item.evidence_id, operation_id=item.operation_id,
            rows=item.rows, columns=steps[item.operation_id].output_fields,
            location="execution.operations" if item.tool == "sql" else "python_execution.operations")
            for item in validation.usable_evidence if item.kind == "table"]
        authorized_charts = {item.operation_id for item in validation.usable_evidence if item.kind == "visualization"}
        response.visualizations = [chart for chart in state.python_execution.visualizations if chart.operation_id in authorized_charts] if state.python_execution else []
        context, language = build_response_context(state, self._provider.context_budget_bytes, self._max_preview_rows)
        response.generation.context = context
        if context.omitted_rows:
            response.limitations.append("El texto resume los resultados extensos; la tabla conserva todas sus filas."
                                        if language == "es" else "The text summarizes long results; the table keeps every row.")
        catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
        entries = [(entry, entry["direct"]) for entry in self._ordered_entries(catalog, refs)]
        introduction = [AnswerSegment(text=context.evidence["introduction"])] if context.evidence.get("introduction") else []
        if validation.status == "partial":
            # Lo que no se pudo hacer se explica primero, con su motivo concreto cuando la ejecución lo informa.
            missing = blocked_explanations(state, language) or [self._message("partial", language)]
            self._publish(response, [*(AnswerSegment(kind="note", text=text) for text in missing), *introduction,
                                     *self._segments(entries)])
            return self._finish(response, started)
        if len(catalog) <= self._deterministic_sentences:
            # El orden (resultado, alcance, detalles) ya está determinado y los textos son verificados;
            # el modelo sólo elegía entre ellos y demoraba entre 4 y 24 s. Un umbral menor reactiva el proveedor.
            self._publish(response, [*introduction, *self._segments(entries)])
            return self._finish(response, started)
        response.generation.model = self._provider.model
        model_started = perf_counter()
        try:
            if context.size_bytes > context.budget_bytes:
                raise LLMProviderError("La evidencia mínima supera el presupuesto de redacción.")
            context.sent_to_model = True
            reply = await self._provider.generate(context)
            response.generation.model = reply.model
            response.generation.input_tokens, response.generation.output_tokens = reply.input_tokens, reply.output_tokens
            selected = [item.sentence_id for item in reply.draft.sentences]
            if len(selected) != len(set(selected)) or set(selected) != set(catalog):
                raise LLMProviderError("El modelo omitió, repitió o inventó enunciados de evidencia.")
            drafted = []
            for sentence in self._ordered(reply.draft.sentences, catalog, refs):
                entry = catalog[sentence.sentence_id]
                if (sentence.evidence_ids != entry["evidence_ids"]
                        or any(key not in refs for key in sentence.evidence_ids)):
                    raise LLMProviderError("La redacción cita evidencia no autorizada.")
                drafted.append((entry, entry[sentence.variant]))
            self._publish(response, [*introduction, *self._segments(drafted)])
            response.generation.status = "generated"
        except LLMProviderError as error:
            response.status = "failed"
            response.generation.status = "failed"
            response.generation.model = getattr(error, "model", None) or response.generation.model
            for field in ("input_tokens", "output_tokens"):
                value = getattr(error, field, None)
                if value is not None:
                    setattr(response.generation, field, value)
            response.generation.error = ExecutionError(code="unverified_generation", message="El modelo no produjo una redacción verificable; no se publica su propuesta.") if context.sent_to_model else ExecutionError(
                code="response_context_limit", message="La evidencia mínima supera el presupuesto de redacción; no se llamó al modelo.")
            response.errors.append(response.generation.error)
            # El reemplazo seguro no publica texto del LLM y deja explícito el fallo.
            self._publish(response, [AnswerSegment(kind="note", text=self._message("generation_failed", language)),
                                     *introduction, *self._segments(entries)])
        response.generation.duration_ms = round((perf_counter() - model_started) * 1000, 3)
        return self._finish(response, started)

    @staticmethod
    def _ordered_entries(catalog, refs):
        order = {key: index for index, key in enumerate(refs)}
        return sorted(catalog.values(), key=lambda entry: _position(entry, order))

    @staticmethod
    def _ordered(sentences, catalog, refs):
        order = {key: index for index, key in enumerate(refs)}
        return sorted(sentences, key=lambda item: _position(catalog[item.sentence_id], order))

    @staticmethod
    def _segments(entries):
        """Un párrafo por evidencia: su resultado, alcance y detalles, seguido de su tabla o gráfico."""
        segments = []
        for entry, text in entries:
            previous = segments[-1] if segments else None
            if previous is not None and previous.evidence_ids == entry["evidence_ids"]:
                previous.text += " " + text
                previous.attachment = previous.attachment or entry.get("attachment")
            else:
                segments.append(AnswerSegment(text=text, evidence_ids=list(entry["evidence_ids"]),
                                              attachment=entry.get("attachment")))
        return segments

    @staticmethod
    def _publish(response, segments):
        response.segments = segments
        response.answer = answer_text(segments)

    @classmethod
    def _unanswered(cls, validation, language, *, explained=False):
        # Con un motivo concreto, la sugerencia genérica de reformular contradiría la explicación que sigue.
        status = "blocked" if explained and validation.status == "failed" else validation.status
        text = cls._message(status, language) or cls._message("failed", language)
        if any(item.code == "incomplete_materialization" for item in validation.issues):
            text += (" Uno de los resultados llegó incompleto, así que no lo usé para responder."
                     if language == "es" else " One of the results was incomplete, so I did not use it to answer.")
        if validation.clarification_question:
            text += " " + validation.clarification_question
        elif validation.missing_information:
            text += (" Me faltó información sobre: " if language == "es" else " I was missing information about: ") \
                + "; ".join(item.rstrip(". ") for item in validation.missing_information) + "."
        return text

    @staticmethod
    def _message(status, language):
        messages = {
            "es": {"validated": "",
                   "partial": "Pude responder sólo una parte de la consulta; el resto no se pudo calcular de forma verificable.",
                   "needs_clarification": "Necesito que me aclares algo para continuar.",
                   "insufficient_coverage": "No encontré datos suficientes para responder esta pregunta en las fuentes que revisé. Esto no significa que la información no exista: puede estar en otra fuente o con otro nombre.",
                   "failed": "No pude armar un análisis confiable para esta pregunta con los datos disponibles. Probá reformularla indicando la variable, el sitio o la fuente que querés usar.",
                   "blocked": "No pude completar el análisis con los datos disponibles.",
                   "generation_failed": "No pude redactar la respuesta completa. Estos son los resultados verificados:"},
            "en": {"validated": "",
                   "partial": "I could only answer part of the question; the rest could not be calculated in a verifiable way.",
                   "needs_clarification": "I need you to clarify something to continue.",
                   "insufficient_coverage": "I did not find enough data to answer this question in the sources I reviewed. This does not mean the information does not exist: it may be in another source or under another name.",
                   "failed": "I could not build a reliable analysis for this question with the available data. Try rephrasing it with the variable, site or source you want to use.",
                   "blocked": "I could not complete the analysis with the available data.",
                   "generation_failed": "I could not write the complete answer. These are the verified results:"},
        }
        return messages[language].get(status, messages[language]["failed"])

    @staticmethod
    def _finish(response, started):
        response.timings_ms["response"] = round((perf_counter() - started) * 1000, 3)
        return response
