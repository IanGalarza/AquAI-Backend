"""Respuesta fundamentada: el LLM organiza enunciados; Python conserva los hechos."""

from time import perf_counter

from aquai.application.answering.context import build_response_context, response_language
from aquai.application.contracts.answering import (
    GenerationRecord, QueryResponse, QueryResponseState, ResponseOperationReference, ResponseResultReference,
)
from aquai.application.contracts.execution import ExecutionError
from aquai.application.errors import LLMProviderError
from aquai.application.execution import evidence_fingerprint
from aquai.application.interfaces.answering import ResponseProvider


class GenerateResponseHandler:
    def __init__(self, provider: ResponseProvider, *, max_preview_rows=10):
        self._provider, self._max_preview_rows = provider, max_preview_rows

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
            response.answer = self._message("failed", language)
            return self._finish(response, started)
        if validation.status not in {"validated", "partial"} or not validation.usable_evidence:
            response.answer = self._message(validation.status, language)
            if any(item.code == "incomplete_materialization" for item in validation.issues):
                response.answer += (" Hay una entrada truncada o incompleta; no se utilizó ese prefijo para calcular una respuesta."
                                    if language == "es" else " An input is truncated or incomplete; that prefix was not used to calculate an answer.")
            if validation.clarification_question:
                response.answer += " " + validation.clarification_question
            elif validation.missing_information:
                response.answer += (" Información faltante: " if language == "es" else " Missing information: ") + "; ".join(validation.missing_information)
            return self._finish(response, started)

        refs = {item.evidence_id: item for item in validation.usable_evidence}
        steps = {step.id: step for step in plan.operations}
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
        catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
        if validation.status == "partial":
            response.answer = self._message("partial", language) + "\n\n" + "\n\n".join(
                item["direct"] + " " + " ".join(f"[{key}]" for key in item["evidence_ids"]) for item in catalog.values())
            self._add_preview_note(response, context, language)
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
            paragraphs = []
            for sentence in reply.draft.sentences:
                entry = catalog[sentence.sentence_id]
                if (sentence.evidence_ids != entry["evidence_ids"]
                        or any(key not in refs for key in sentence.evidence_ids)):
                    raise LLMProviderError("La redacción cita evidencia no autorizada.")
                paragraphs.append(entry[sentence.variant] + " " + " ".join(f"[{key}]" for key in entry["evidence_ids"]))
            response.answer = self._message(validation.status, language) + "\n\n" + "\n\n".join(paragraphs)
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
            response.answer = self._message("generation_failed", language) + "\n\n" + "\n\n".join(
                item["direct"] + " " + " ".join(f"[{key}]" for key in item["evidence_ids"]) for item in catalog.values())
        response.generation.duration_ms = round((perf_counter() - model_started) * 1000, 3)
        self._add_preview_note(response, context, language)
        return self._finish(response, started)

    @staticmethod
    def _add_preview_note(response, context, language):
        if context.omitted_rows:
            note = ("El texto muestra una vista abreviada; los resultados completos permanecen en las referencias estructuradas."
                    if language == "es" else "The text is an abbreviated view; complete results remain available through the structured references.")
            response.answer += "\n\n" + note
            response.limitations.append(note)

    @staticmethod
    def _message(status, language):
        messages = {
            "es": {"validated": "Estos son los resultados obtenidos.",
                   "partial": "Pude resolver sólo una parte de la consulta. Los resultados siguientes están verificados; quedan operaciones pendientes o con errores.",
                   "needs_clarification": "Necesito una aclaración para continuar.",
                   "insufficient_coverage": "No encontré evidencia suficiente entre los candidatos recuperados para responder esta pregunta; esto no demuestra que la información no exista en todo el catálogo.",
                   "failed": "No dispongo de resultados completos y verificables para responder de forma confiable.",
                   "generation_failed": "No pude generar una explicación verificable con el modelo. Estos son los resultados disponibles, presentados de forma determinística."},
            "en": {"validated": "Here are the results.",
                   "partial": "I could resolve only part of the question. The following results are verified; some operations remain pending or failed.",
                   "needs_clarification": "I need a clarification to continue.",
                   "insufficient_coverage": "There is insufficient evidence among the retrieved candidates to answer this question; this does not establish that the information is absent from the entire catalog.",
                   "failed": "I do not have complete, verifiable results to answer reliably.",
                   "generation_failed": "The model did not produce a verifiable explanation. The available results are presented deterministically below."},
        }
        return messages[language].get(status, messages[language]["failed"])

    @staticmethod
    def _finish(response, started):
        response.timings_ms["response"] = round((perf_counter() - started) * 1000, 3)
        return response
