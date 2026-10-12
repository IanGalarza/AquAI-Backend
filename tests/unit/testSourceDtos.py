"""DTOs reales de Application y conservación de los documentos públicos existentes."""

from datetime import datetime, timezone
import unittest
from unittest.mock import Mock
from uuid import uuid4

from pydantic import ValidationError

from aquai.application.dtos.requests.listSources import ListSourcesRequest
from aquai.application.dtos.requests.reviewSource import SourceDecisionRequest
from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.dtos.responses.listSources import SourceListItem, SourceListResponse
from aquai.application.dtos.responses.reviewSource import SourceDecisionResponse
from aquai.application.dtos.responses.sourceDetails import (
    SourceFinding,
    SourcePreview,
    SourceReview,
    SourceReviewOption,
)
from aquai.application.useCases.sources import GetSourceHandler, ListSourcesHandler, ReviewSourceHandler


class SourceDtoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.source_id = uuid4()
        self.metadata = {
            "id": self.source_id, "name": "data.csv", "original_filename": "data.csv",
            "file_type": "csv", "mime_type": "text/csv", "file_size_bytes": 12,
            "created_at": datetime(2026, 10, 12, tzinfo=timezone.utc), "updated_at": None,
            "is_active": False, "processing_status": "review_required", "validation_status": "review_required",
        }

    async def test_list_returns_typed_items_and_filters_persistence_only_fields(self):
        queries = Mock()
        queries.list.return_value = {
            "items": [dict(self.metadata, file_content=b"private")], "total": 1, "page": 2, "page_size": 10,
        }
        request = ListSourcesRequest(page=2, page_size=10, name="data", processing_status="review_required")
        result = await ListSourcesHandler(queries).handle(request)
        self.assertIsInstance(result, SourceListResponse)
        self.assertIsInstance(result.items[0], SourceListItem)
        self.assertNotIn("file_content", result.model_dump()["items"][0])
        queries.list.assert_called_once_with(
            page=2, page_size=10, search="data", processing_status="review_required", validation_status=None,
        )

    async def test_detail_types_stable_objects_and_preserves_dynamic_values_and_absent_fields(self):
        finding = {"code": "conversion_failures", "message": "Corregir columna.", "range": None}
        option = {"action": "correct_transformations", "description": "Corregir operaciones.",
                  "values": {"type": "object", "properties": {"transformations": {"type": "array"}}}}
        review = {"id": "review-1", "code": finding["code"], "message": finding["message"],
                  "options": [option], "finding": finding, "range": "A2:B2"}
        queries = Mock()
        queries.get.return_value = dict(self.metadata, recipe_data={"_workflow": {"stage": "awaiting_review"}},
            validation_report={"reviews": [review], "evaluation": {
                "validation": {"findings": [finding]},
                "execution": {"sheets": [{"sheet_name": "data", "block_id": "records", "columns_after": ["value"],
                                          "sample": [{"row_number": 2, "values": [0, 1.25, None, {"dynamic": True}]}]}]},
            }})
        result = await GetSourceHandler(queries).handle(self.source_id)
        self.assertIsInstance(result, SourceDetailResponse)
        self.assertIsInstance(result.findings[0], SourceFinding)
        self.assertIsInstance(result.preview[0], SourcePreview)
        self.assertIsInstance(result.reviews[0], SourceReview)
        self.assertIsInstance(result.reviews[0].options[0], SourceReviewOption)
        payload = result.model_dump(mode="json", by_alias=True)
        self.assertEqual(payload["reviews"][0], review)
        self.assertEqual(payload["preview"][0]["rows"][0]["values"], [0, 1.25, None, {"dynamic": True}])
        self.assertEqual(payload["findings"][0]["range"], "A2:B2")
        self.assertNotIn("cell", payload["findings"][0])
        self.assertIsNone(payload["reviews"][0]["finding"]["range"])
        self.assertFalse(payload["isActive"])

    async def test_decision_returns_real_response_and_keeps_mutation_inside_commands(self):
        commands = Mock()
        commands.mutate.return_value = {
            "id": str(self.source_id), "processing_status": "uploaded", "validation_status": "pending",
            "review_id": "review-1", "revision": 1, "duplicate": False,
        }
        request = SourceDecisionRequest(action="correct_transformations", values={"transformations": []})
        result = await ReviewSourceHandler(commands).handle(self.source_id, "review-1", request)
        self.assertIsInstance(result, SourceDecisionResponse)
        self.assertEqual(result.id, self.source_id)
        self.assertEqual(commands.mutate.call_args.args[0], self.source_id)
        self.assertTrue(callable(commands.mutate.call_args.args[1]))

    def test_request_constraints_reject_invalid_filters_and_non_json_decision_shapes(self):
        for values in ({"page": 0}, {"page_size": 101}, {"processing_status": "unknown"},
                       {"validation_status": "unknown"}, {"name": "x" * 256}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ListSourcesRequest.model_validate(values)
        for values in ({"action": ""}, {"action": "correct_transformations", "values": []},
                       {"action": "correct_transformations", "values": {}, "extra": True}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SourceDecisionRequest.model_validate(values)


if __name__ == "__main__":
    unittest.main()
