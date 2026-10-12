"""Coordinate persisted reviews and decisions; a review never waives validation."""

from copy import deepcopy
from hashlib import sha256
import json
from uuid import UUID

from pydantic import ValidationError

from aquai.application.contracts.tabular.results import TabularEvaluation, TabularFinding
from aquai.application.errors import SourceDecisionError
from aquai.application.services.sourceReviewActions import apply_recipe_correction, clarify_record_identity
from aquai.application.services.sourceReviewOptions import build_review_options, is_technical_finding
from aquai.application.utils.tabularCoordinates import cell_coordinate


class SourceReviewService:
    def build_reviews(
        self, source_id: UUID, evaluation: TabularEvaluation, rules: dict | None,
        *, revision: int = 0,
    ) -> list[dict]:
        reviews = []
        seen = set()
        for finding in evaluation.validation.findings:
            if not finding.requires_review:
                continue
            options = build_review_options(finding, evaluation)
            if not options and self.is_technical_finding(finding):
                continue
            identity = json.dumps({
                "source_id": str(source_id), "revision": revision,
                "finding": finding.model_dump(mode="json"),
            }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            review_id = sha256(identity.encode()).hexdigest()[:32]
            if review_id in seen:
                continue
            seen.add(review_id)
            reviews.append({
                "id": review_id, "review_id": review_id, "source_id": str(source_id),
                "revision": revision, "code": finding.code, "message": finding.message,
                "kind": "review",
                "sheet_name": finding.sheet_name, "block_id": finding.block_id,
                "cell": finding.cell, "range": self._finding_range(finding, evaluation), "region_id": finding.region_id,
                "options": options, "finding": finding.model_dump(mode="json"),
                "blocked_reason": None if options else "No hay una corrección estructurada admitida para este hallazgo; se conserva pendiente de revisión.",
            })
        return reviews

    @classmethod
    def is_technical_finding(cls, finding: TabularFinding | dict) -> bool:
        return is_technical_finding(finding)

    @staticmethod
    def _finding_range(finding: TabularFinding, evaluation: TabularEvaluation) -> str | None:
        if finding.range or finding.cell:
            return finding.range or finding.cell
        if evaluation.recipe is None:
            return None
        block = next((block for block in evaluation.recipe.sheets
            if block.sheet_name == finding.sheet_name and block.block_id == finding.block_id), None)
        if block is not None:
            start = cell_coordinate(block.data_start_row, block.data_start_column)
            end = cell_coordinate(block.data_end_row, block.data_end_column)
            return f"{start}:{end}"
        sheet = next((sheet for sheet in evaluation.structural_summary.sheets
                      if sheet.name == finding.sheet_name), None)
        if sheet is not None and sheet.rows > 0 and sheet.columns > 0:
            return f"A1:{cell_coordinate(sheet.rows, sheet.columns)}"
        return None

    def apply(
        self, review: dict, action: str, values: dict,
        evaluation: TabularEvaluation, rules: dict | None,
    ) -> tuple[TabularEvaluation, dict, str]:
        try:
            finding = TabularFinding.model_validate(review["finding"])
        except (KeyError, ValidationError) as exc:
            raise SourceDecisionError("La revisión guardada no es válida.") from exc
        if not any(item == finding and item.requires_review for item in evaluation.validation.findings):
            raise SourceDecisionError("La revisión ya no corresponde a un hallazgo vigente.")
        supported = {option["action"] for option in build_review_options(finding, evaluation)}
        if action not in supported:
            raise SourceDecisionError("La acción no está admitida para esta revisión.")
        if not isinstance(values, dict):
            raise SourceDecisionError("Los valores de la decisión deben ser un objeto.")
        result = evaluation.model_copy(deep=True)
        snapshot = deepcopy(rules) if rules is not None else {
            "metadata": {}, "prompt_rules": {}, "findings": [], "review_reasons": [],
            "identity_clarifications": [], "preparation_policies": None,
        }
        try:
            match action:
                case "clarify_record_identity":
                    clarify_record_identity(review, finding, values, result, snapshot)
                    restart = "validation"
                case "correct_transformations" | "replace_block" | "add_block":
                    result.recipe = apply_recipe_correction(
                        action, result.recipe, finding, values, result.structural_summary,
                    )
                    restart = "execution"
                case _:
                    raise SourceDecisionError("La acción no está admitida para esta revisión.")
        except (ValidationError, KeyError, TypeError) as exc:
            raise SourceDecisionError("Los valores de la decisión no cumplen el contrato de la receta.") from exc
        snapshot.setdefault("source_decisions", []).append({
            "review_id": review["id"], "action": action, "values": deepcopy(values),
            "scope": {
                "sheet_name": review.get("sheet_name"),
                "block_id": review.get("block_id"),
                "cell": review.get("cell"),
                "range": review.get("range"),
                "region_id": review.get("region_id"),
            },
        })
        return result, snapshot, restart
