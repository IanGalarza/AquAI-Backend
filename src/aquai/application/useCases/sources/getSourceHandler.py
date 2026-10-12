"""Detalle de una fuente con hallazgos y muestras acotadas."""

import asyncio
import json
from copy import deepcopy
from hashlib import sha256
from uuid import UUID

from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.services.sourceReviews import SourceReviewService
from aquai.application.useCases.sources.constants import DATASET_STORAGE


_METADATA_FIELDS = (
    "id", "name", "original_filename", "file_type", "mime_type", "file_size_bytes",
    "created_at", "updated_at", "is_active", "processing_status", "validation_status",
)


class GetSourceHandler:
    def __init__(
        self, queries: SourceQueries, *, reviews: SourceReviewService | None = None,
    ) -> None:
        self._queries = queries
        self._reviews = reviews or SourceReviewService()

    async def handle(self, source_id: UUID) -> SourceDetailResponse:
        source = await asyncio.to_thread(self._queries.get, source_id)
        report = source.get("validation_report") or {}
        data = source.get("recipe_data") or {}
        workflow = data.get("_workflow", {})
        evaluation = report.get("evaluation") or {}
        validation = evaluation.get("validation") or {}
        reviews = report.get("reviews", []) if source["processing_status"] == "review_required" else []

        return SourceDetailResponse.model_validate({
            **{key: source[key] for key in _METADATA_FIELDS},
            "stage": workflow.get("stage", "uploaded"),
            "revision": workflow.get("revision", 0),
            "blocks": self._blocks(evaluation),
            "preview": self._preview(evaluation),
            "findings": self._findings(source_id, validation.get("findings", []), reviews),
            "reviews": reviews,
            "decisions": report.get("decisions", []),
            "technical_errors": report.get("technical_errors", []),
            "dataset_storage": DATASET_STORAGE,
            "queryable": False,
            "original_proposal": data.get("proposal"),
            "effective_recipe": data.get("effective"),
            "rules": source.get("rules_snapshot") or {},
        })

    def _findings(self, source_id: UUID, findings: list[dict], reviews: list[dict]) -> list[dict]:
        result = []
        for finding in findings:
            finding = deepcopy(finding)
            review = next((item for item in reviews if item.get("finding") == finding), None)
            finding["id"] = review["id"] if review else self._finding_id(source_id, finding)
            if review and not finding.get("range"):
                finding["range"] = review.get("range")
            if review:
                finding["kind"] = "review"
            else:
                finding["kind"] = (
                    "technical" if self._reviews.is_technical_finding(finding) else "validation"
                )
            result.append(finding)
        return result

    @staticmethod
    def _finding_id(source_id: UUID, finding: dict) -> str:
        identity = str(source_id) + json.dumps(finding, sort_keys=True, ensure_ascii=False)
        return sha256(identity.encode()).hexdigest()[:32]

    @staticmethod
    def _blocks(evaluation: dict) -> list[dict]:
        sheets = evaluation.get("structural_summary", {}).get("sheets", [])
        return [
            dict(region, sheet_name=sheet["name"])
            for sheet in sheets
            for region in sheet.get("regions", [])
        ]

    @classmethod
    def _preview(cls, evaluation: dict) -> list[dict]:
        executed = evaluation.get("execution", {}).get("sheets", [])
        sheets = executed or evaluation.get("structural_summary", {}).get("sheets", [])
        name_key = "sheet_name" if executed else "name"
        columns_key = "columns_after" if executed else "initial_columns"
        preview = []
        for sheet in sheets[:30]:
            item = {
                "sheet_name": sheet[name_key],
                "columns": sheet.get(columns_key, [])[:20],
                "rows": cls._bounded_rows(sheet.get("sample", [])),
            }
            if executed:
                item["block_id"] = sheet.get("block_id")
            preview.append(item)
        return preview

    @staticmethod
    def _bounded_rows(rows: list[dict]) -> list[dict]:
        result = []
        for row in rows[:10]:
            values = [
                value[:200] if isinstance(value, str) else value
                for value in row.get("values", [])[:20]
            ]
            result.append(dict(row, values=values))
        return result
