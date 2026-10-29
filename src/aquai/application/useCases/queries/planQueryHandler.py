"""Planificación semántica con validaciones determinísticas y catálogo vigente."""

import asyncio
from datetime import datetime, timezone

from aquai.application.contracts.planning import PlanningInput, PlanningResult, ValidationIssue
from aquai.application.contracts.retrieval import DatasetRetrievalResult
from aquai.application.errors import PlannerValidationError
from aquai.application.interfaces.planning import PlannerProvider
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.planning.context import build_planning_context, latest_requested
from aquai.application.planning.filterValues import resolve_filter_values
from aquai.application.planning.validation import validate_operations, validate_selections, validate_temporal_scope


class PlanQueryHandler:
    def __init__(self, provider: PlannerProvider, queries: SourceQueries) -> None:
        self._provider = provider
        self._queries = queries

    async def handle(self, question: str, retrieval: DatasetRetrievalResult, *, original_question: str | None = None,
                     data_mentions=()) -> PlanningResult:
        question = PlanningInput(question=question).question
        if question != retrieval.question.strip():
            raise PlannerValidationError("La pregunta no coincide con la utilizada por el Retriever.")
        versions = {item.dataset_id: item.verified_version for item in retrieval.candidates}
        verified = await asyncio.to_thread(self._queries.verified_datasets, list(versions), versions=versions) if versions else []
        by_id = {item.dataset_id: item for item in verified}
        # Mantiene el orden recuperado sin trasladar scores al LLM.
        catalog = [by_id[item.dataset_id] for item in retrieval.candidates if item.dataset_id in by_id]
        require_latest = latest_requested(original_question or question)
        excluded = []
        if require_latest:
            excluded = [str(item.dataset_id) for item in catalog
                        if not any(column.sql_type in {"DATE", "TIMESTAMP"} for column in item.columns_metadata)]
            catalog = [item for item in catalog if str(item.dataset_id) not in excluded]
        context = build_planning_context(question, catalog, self._provider.context_budget_bytes,
                                         temporal_candidates_excluded=excluded, require_latest=require_latest,
                                         query_mentions=data_mentions)
        common = {"context": context, "verified_at": datetime.now(timezone.utc)}
        if not catalog and excluded:
            return PlanningResult(
                **common, status="needs_clarification", llm_status=None,
                reason="Los candidatos recuperados carecen de fechas DATE/TIMESTAMP verificadas; no se interpreta TEXT como fecha ni se elige un registro arbitrario.",
                clarification_question="¿Qué columna y formato de fecha están documentados para estos registros?",
                missing_information=["Una fecha real DATE/TIMESTAMP para determinar la medición más reciente."],
                warnings=list(retrieval.warnings),
            )
        if not catalog:
            return PlanningResult(
                **common, status="insufficient_coverage", llm_status=None,
                reason="No hay candidatos disponibles para fundamentar una planificación.",
                missing_information=["Datasets recuperados disponibles y vigentes."],
                warnings=[*retrieval.warnings, "El alcance es el conjunto recuperado; no se evaluó todo el catálogo."],
            )
        if context.size_bytes > context.budget_bytes:
            return PlanningResult(
                **common, status="needs_clarification", llm_status=None,
                reason="El catálogo completo de columnas supera el presupuesto del modelo; no se descartaron columnas para forzar un plan.",
                clarification_question="¿Podés acotar las variables o las fuentes necesarias para esta consulta?",
                missing_information=["Un alcance que permita enviar todos sus metadatos al Planner."],
            )
        reply = await self._provider.propose(context)
        context.sent_to_model = True
        decision = reply.decision
        selected = validate_selections(decision, catalog)
        operations, issues = validate_operations(decision, selected)
        resolutions = []
        if decision.status == "plannable" and not any(issue.severity == "clarification" for issue in issues):
            decision, resolutions, value_issues = await resolve_filter_values(
                decision, selected, operations, data_mentions, self._queries)
            if resolutions:
                operations, issues = validate_operations(decision, selected)
            issues.extend(value_issues)
        if decision.status == "plannable":
            issues.extend(validate_temporal_scope(operations, decision.output_steps, require_latest=require_latest,
                requested_values=[mention.text for mention in data_mentions if mention.kind == "value"]))
        if selected:
            current = await asyncio.to_thread(
                self._queries.verified_datasets, [item.dataset_id for item in selected],
                versions={item.dataset_id: item.verified_version for item in selected},
            )
            current_by_id = {item.dataset_id: item for item in current}
            old_by_id = {item.dataset_id: item for item in catalog}
            for item in selected:
                latest = current_by_id.get(item.dataset_id)
                if latest is None or latest.model_dump() != old_by_id[item.dataset_id].model_dump():
                    issues.append(ValidationIssue(code="catalog_changed", severity="clarification",
                        message="El catálogo elegido cambió durante la inferencia; hay que recuperar y planificar de nuevo."))
        blocked = [issue.message for issue in issues if issue.severity == "clarification"]
        status = "needs_clarification" if blocked else decision.status
        warnings = list(dict.fromkeys([
            "La cobertura se evaluó sólo sobre los candidatos recuperados, no sobre todo el catálogo.",
            *([f"No se evaluó recencia en {len(excluded)} candidatos sin fechas DATE/TIMESTAMP verificadas; no se interpretaron fechas TEXT ni encabezados como mediciones temporales."] if excluded else []),
            *decision.warnings, *(warning for item in selected for warning in item.warnings),
            *(issue.message for issue in issues if issue.severity == "warning"),
        ]))
        if blocked:
            reason = "La propuesta requiere resolver verificaciones determinísticas pendientes."
        elif status == "plannable":
            reason = f"Propuesta de {len(selected)} datasets y {len(operations)} operaciones contrastada con el catálogo; la relevancia fue evaluada por el modelo."
        elif status == "needs_clarification":
            reason = "El modelo requiere aclaración para completar el plan."
        else:
            reason = "El modelo no encontró cobertura suficiente entre los candidatos recuperados."
        return PlanningResult(
            status=status, llm_status=decision.status,
            reason=reason, llm_reason=decision.reason,
            selected_datasets=selected, operations=operations if status == "plannable" else [],
            output_steps=decision.output_steps if status == "plannable" else [],
            required_tools=sorted({step.tool for step in operations}) if status == "plannable" else [],
            missing_information=[*decision.missing_information, *blocked],
            clarification_question=("¿Cómo querés resolver estas limitaciones: " + " ".join(blocked) + "?") if blocked else decision.clarification_question,
            warnings=warnings, validation_issues=issues, context=context,
            filter_value_resolutions=resolutions,
            model=reply.model, input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
            verified_at=datetime.now(timezone.utc),
        )
