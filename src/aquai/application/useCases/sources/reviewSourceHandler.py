"""Aplica una decisión de revisión y encola su revalidación atómicamente."""

import asyncio
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
from uuid import UUID

from aquai.application.contracts.tabular.results import TabularEvaluation
from aquai.application.dtos.requests.reviewSource import SourceDecisionRequest
from aquai.application.dtos.responses.reviewSource import SourceDecisionResponse
from aquai.application.errors import SourceConflictError, SourceDecisionError
from aquai.application.interfaces.sources import SourceCommands
from aquai.application.services.sourceReviews import SourceReviewService


class ReviewSourceHandler:
    def __init__(
        self,
        commands: SourceCommands,
        *,
        reviews: SourceReviewService | None = None,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._reviews = reviews or SourceReviewService()
        self._notify = notify

    async def handle(
        self, source_id: UUID, review_id: str, decision: SourceDecisionRequest,
    ) -> SourceDecisionResponse:
        update = partial(
            self._apply_decision,
            source_id=source_id,
            review_id=review_id,
            action=decision.action,
            values=decision.values,
            decision_key=self._decision_key(decision.action, decision.values),
        )
        response = await asyncio.to_thread(self._commands.mutate, source_id, update)
        if self._notify is not None:
            self._notify()
        return SourceDecisionResponse.model_validate(response)

    def _apply_decision(
        self,
        source: dict,
        *,
        source_id: UUID,
        review_id: str,
        action: str,
        values: dict,
        decision_key: str,
    ) -> tuple[dict, dict]:
        report = deepcopy(source.get("validation_report") or {})
        duplicate = self._duplicate_response(source, report, review_id, decision_key)
        if duplicate is not None:
            return {}, duplicate

        review = self._current_review(source, report, review_id)
        evaluation = TabularEvaluation.model_validate(report["evaluation"])
        corrected, rules, resume_stage = self._reviews.apply(
            review, action, values, evaluation, deepcopy(source["rules_snapshot"])
        )
        data = deepcopy(source["recipe_data"])
        revision = self._queue_resume(data, report, corrected, resume_stage)
        response = {
            "id": str(source_id),
            "review_id": review_id,
            "processing_status": "uploaded",
            "validation_status": "pending",
            "revision": revision,
            "duplicate": False,
        }
        report.setdefault("decisions", []).append({
            "review_id": review_id,
            "key": decision_key,
            "action": action,
            "values": deepcopy(values),
            "scope": {
                "source_id": str(source_id),
                **{key: review.get(key) for key in ("sheet_name", "block_id", "range", "cell")},
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": response,
        })
        changes = {
            "recipe_data": data,
            "rules_snapshot": rules,
            "validation_report": report,
            "processing_status": "uploaded",
            "validation_status": "pending",
        }
        return changes, response

    @staticmethod
    def _decision_key(action: str, values: dict) -> str:
        try:
            serialized = json.dumps(
                {"action": action, "values": values},
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (ValueError, TypeError) as exc:
            raise SourceDecisionError("Los valores deben formar un objeto JSON válido.") from exc
        return sha256(serialized.encode()).hexdigest()

    @staticmethod
    def _duplicate_response(
        source: dict, report: dict, review_id: str, decision_key: str
    ) -> dict | None:
        decision = next(
            (item for item in report.get("decisions", []) if item["review_id"] == review_id),
            None,
        )
        if decision is None:
            return None
        if decision["key"] != decision_key:
            raise SourceConflictError("La revisión ya recibió otra decisión.")
        return dict(
            decision["response"],
            duplicate=True,
            processing_status=source["processing_status"],
            validation_status=source["validation_status"],
        )

    @staticmethod
    def _current_review(source: dict, report: dict, review_id: str) -> dict:
        if (
            source["processing_status"] != "review_required"
            or source["validation_status"] not in {"review_required", "failed"}
        ):
            raise SourceConflictError("La fuente no se encuentra esperando una revisión vigente.")
        review = next(
            (item for item in report.get("reviews", []) if item["id"] == review_id),
            None,
        )
        if review is None:
            raise SourceConflictError("La revisión no está vigente para esta fuente.")
        return review

    @staticmethod
    def _queue_resume(
        data: dict, report: dict, corrected: TabularEvaluation, resume_stage: str
    ) -> int:
        workflow = data["_workflow"]
        workflow["revision"] = workflow.get("revision", 0) + 1
        workflow.update(stage="queued_resume", resume_stage=resume_stage)
        data["effective"] = corrected.recipe.model_dump(mode="json") if corrected.recipe else None
        report["evaluation"] = corrected.model_dump(mode="json")
        report["reviews"] = []
        return workflow["revision"]
