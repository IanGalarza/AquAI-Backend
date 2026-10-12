"""Procesa una fuente reclamada y persiste cada etapa del pipeline."""

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID

from aquai.application.contracts.tabular.results import TabularEvaluation
from aquai.application.errors import SourceConflictError
from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.services.sourceReviews import SourceReviewService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.application.useCases.sources.constants import DATASET_STORAGE


_FINAL_STAGES = {
    "ready": "prepared",
    "ready_with_warnings": "prepared",
    "review_required": "awaiting_review",
    "failed": "failed",
}


@dataclass
class _ProcessingState:
    recipe_data: dict
    report: dict
    rules: dict

    @classmethod
    def from_source(cls, source: dict) -> "_ProcessingState":
        return cls(
            recipe_data=deepcopy(source["recipe_data"]),
            report=deepcopy(source.get("validation_report") or {}),
            rules=deepcopy(source["rules_snapshot"]),
        )

    @property
    def workflow(self) -> dict:
        return self.recipe_data["_workflow"]

    def saved_evaluation(self) -> TabularEvaluation | None:
        if self.workflow.get("resume_stage") and self.report.get("evaluation"):
            return TabularEvaluation.model_validate(self.report["evaluation"])
        return None

    def checkpoint(
        self, stage: str, evaluation: TabularEvaluation | None, rules: dict | None,
    ) -> None:
        self.workflow["stage"] = stage
        if rules is not None:
            self.rules = deepcopy(rules)
        if evaluation is not None:
            self.report["evaluation"] = evaluation.model_dump(mode="json")
            if evaluation.recipe is not None:
                effective = evaluation.recipe.model_dump(mode="json")
                if self.recipe_data.get("proposal") is None:
                    self.recipe_data["proposal"] = deepcopy(effective)
                self.recipe_data["effective"] = effective
        self.report["dataset_storage"] = DATASET_STORAGE


class ProcessSourceHandler:
    def __init__(
        self,
        commands: SourceCommands,
        queries: SourceQueries,
        evaluator: TabularEvaluationService,
        *,
        reviews: SourceReviewService | None = None,
    ) -> None:
        self._commands = commands
        self._queries = queries
        self._evaluator = evaluator
        self._reviews = reviews or SourceReviewService()

    async def handle(self, source_id: UUID) -> None:
        source = await asyncio.to_thread(self._queries.get, source_id, include_content=True)
        state = _ProcessingState.from_source(source)
        saved = state.saved_evaluation()

        async def checkpoint(
            stage: str, evaluation: TabularEvaluation | None, rules: dict | None,
        ) -> None:
            state.checkpoint(stage, evaluation, rules)
            await self._save(source_id, state)

        result = await self._evaluator.process_source(
            source["original_filename"],
            source["file_content"],
            checkpoint=checkpoint,
            saved_evaluation=saved,
            saved_rules=state.rules if saved is not None else None,
            resume_stage=state.workflow.get("resume_stage", "execution"),
        )
        await self._complete(source_id, state, result)

    async def _save(
        self,
        source_id: UUID,
        state: _ProcessingState,
        *,
        processing_status: str | None = None,
        validation_status: str | None = None,
    ) -> None:
        statuses = {}
        if processing_status is not None:
            statuses["processing_status"] = processing_status
        if validation_status is not None:
            statuses["validation_status"] = validation_status
        saved = await asyncio.to_thread(
            self._commands.save,
            source_id,
            recipe_data=state.recipe_data,
            rules_snapshot=state.rules,
            validation_report=state.report,
            expected_processing_status="processing",
            **statuses,
        )
        if not saved:
            message = (
                "La fuente dejó de pertenecer al procesamiento actual."
                if processing_status is None
                else "No se pudo guardar el resultado en el estado actual."
            )
            raise SourceConflictError(message)

    async def _complete(
        self, source_id: UUID, state: _ProcessingState, result: TabularEvaluation,
    ) -> None:
        validation_status = result.validation.status.value
        reviews = self._reviews.build_reviews(
            source_id, result, state.rules, revision=state.workflow.get("revision", 0),
        )
        # Una corrección pendiente puede conservar el estado de validación failed.
        processing_status = "review_required" if reviews else validation_status
        state.workflow.pop("resume_stage", None)
        try:
            state.workflow["stage"] = _FINAL_STAGES[processing_status]
        except KeyError as exc:
            raise SourceConflictError("Estado de procesamiento final desconocido.") from exc
        state.report.update(
            evaluation=result.model_dump(mode="json"),
            reviews=reviews,
            technical_errors=self._technical_errors(result, reviews),
        )
        await self._save(
            source_id, state,
            processing_status=processing_status,
            validation_status=validation_status,
        )

    def _technical_errors(self, result: TabularEvaluation, reviews: list[dict]) -> list[dict]:
        corrective_errors = {
            (f"Hoja '{review['sheet_name']}': " if review.get("sheet_name") else "")
            + review["message"]
            for review in reviews
            if review.get("options") and review["finding"].get("severity") == "error"
        }
        errors = [
            {"code": "validation_error", "message": error}
            for error in result.validation.errors
            if error not in corrective_errors
        ]
        errors.extend(
            {
                "code": finding.code,
                "message": finding.message,
                "sheet_name": finding.sheet_name,
                "cell": finding.cell,
                "range": finding.range,
            }
            for finding in result.validation.findings
            if self._reviews.is_technical_finding(finding)
        )
        return errors
