"""Contratos HTTP y workflow real con BD aislada y Ollama simulado."""

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
from fastapi.testclient import TestClient
from openpyxl import Workbook
from sqlalchemy import func, select
from starlette.routing import Match

from aquai.application.services.tabularRules import TabularRuleService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.models import Dataset, Source, SourceRecipe
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.llm.ollama import OllamaProvider
from aquai.infrastructure.providers.sources import SourceRuntime, build_source_runtime
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from aquai.main import create_app
from tests.sourceDatabase import SourceTestDbContext


def preparation_recipe(**overrides) -> dict:
    block = {
        "sheet_name": "data", "block_id": "observations", "role": "observations",
        "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
        "data_start_column": 1, "data_end_column": 2, "transformations": [],
    }
    block.update(overrides)
    return {"status": "ready", "sheets": [block], "warnings": [], "requires_human_review": False}


class TrackingSourceDbContext(SourceTestDbContext):
    def __init__(self) -> None:
        self.open_sessions = 0
        super().__init__()

    @contextmanager
    def session(self):
        self.open_sessions += 1
        try:
            with super().session() as session:
                yield session
        finally:
            self.open_sessions -= 1


class SourceEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrackingSourceDbContext()
        self.addCleanup(self.context.dispose)
        self.settings = TabularSettings(_env_file=None)
        self.requests = []
        self.provider_sessions = []
        self.raw_response = json.dumps(preparation_recipe())
        self.provider_error = None

        def respond(request: httpx.Request) -> httpx.Response:
            self.provider_sessions.append(self.context.open_sessions)
            payload = json.loads(request.content)
            self.requests.append(payload)
            if self.provider_error is not None:
                raise self.provider_error
            return httpx.Response(200, json={
                "model": payload["model"], "done": True,
                "message": {"role": "assistant", "content": self.raw_response},
            })

        self.processor = PandasTabularProcessor()
        self.evaluator = TabularEvaluationService(
            self.processor, OllamaProvider(self.settings, transport=httpx.MockTransport(respond)),
            default_model=self.settings.default_model, max_upload_bytes=self.settings.max_upload_bytes,
            rule_service=TabularRuleService(FileTabularRuleCatalog()),
        )
        self.commands = SqlAlchemySourceCommands(self.context)
        self.queries = SqlAlchemySourceQueries(self.context)
        self.sources = self.new_runtime()
        # Lifespan is deliberately not started: run_once gives deterministic
        # checkpoints between HTTP requests while using the production router.
        self.client = TestClient(create_app(sources=self.sources))
        self.addCleanup(self.client.close)

    def new_runtime(self) -> SourceRuntime:
        return build_source_runtime(
            self.commands,
            self.queries,
            self.evaluator,
            max_upload_bytes=self.settings.max_upload_bytes,
        )

    def upload(self, content=b"name,score\nAna,10\nLuis,20\n", name="data.csv", **kwargs):
        return self.client.post("/sources", files={"file": (name, content)}, **kwargs)

    def upload_id(self, *args, **kwargs) -> UUID:
        response = self.upload(*args, **kwargs)
        self.assertEqual(response.status_code, 202, response.text)
        return UUID(response.json()["id"])

    def process(self) -> None:
        self.assertTrue(asyncio.run(self.sources.worker.run_once()))

    def detail(self, source_id: UUID) -> dict:
        response = self.client.get(f"/sources/{source_id}")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def review_source(self) -> tuple[UUID, dict]:
        self.raw_response = json.dumps(preparation_recipe(transformations=[{
            "operation": "convert_type", "columns": ["score"], "target_type": "integer",
        }]))
        source_id = self.upload_id(b"name,score\nAna,invalid\nLuis,20\n")
        self.process()
        detail = self.detail(source_id)
        self.assertEqual(detail["processing_status"], "review_required")
        review = next(item for item in detail["reviews"] if item["code"] == "conversion_failures")
        return source_id, review

    def decide(self, source_id: UUID, review: dict, payload=None):
        return self.client.post(f"/sources/{source_id}/reviews/{review['id']}/decisions", json=payload or {
            "action": "correct_transformations", "values": {"transformations": []},
        })

    def test_upload_persists_original_and_initial_status_before_any_processing(self) -> None:
        original = b"name,score\nAna,10\nLuis,20\n"
        response = self.upload(original)
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(set(body), {"id", "processing_status", "validation_status"})
        self.assertEqual((body["processing_status"], body["validation_status"]), ("uploaded", "pending"))
        saved = self.queries.get(UUID(body["id"]), include_content=True)
        self.assertEqual(saved["file_content"], original)
        self.assertEqual(saved["recipe_data"]["_workflow"]["stage"], "uploaded")
        self.assertFalse(saved["is_active"])
        self.assertEqual(self.requests, [])
        with self.context.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SourceRecipe)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(Dataset)), 0)

    def test_upload_rejects_extra_fields_query_controls_and_multiple_files(self) -> None:
        for extra in ({"known_rules": "Conservar todo"}, {"rule_profile": "self_sampling"}, {"model": "x"}):
            response = self.upload(data=extra)
            self.assertEqual(response.status_code, 400, response.text)
            response = self.upload(params=extra)
            self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/sources", files=[
            ("file", ("one.csv", b"a\n1")), ("file", ("two.csv", b"a\n2")),
        ])
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.get("/sources").json()["total"], 0)
        self.assertFalse(self.requests)

    def test_upload_validation_and_size_limit_do_not_create_sources(self) -> None:
        for name, content in (("file.pdf", b"pdf"), ("data.csv", b"")):
            response = self.upload(content, name=name)
            self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.client.post("/sources").status_code, 422)
        self.sources.upload.max_upload_bytes = 5
        self.assertEqual(self.upload(b"123456").status_code, 413)
        self.assertEqual(self.client.get("/sources").json()["total"], 0)
        self.assertFalse(self.requests)

    def test_upload_retry_reuses_source_recipe_without_overwriting_original(self) -> None:
        first = self.upload_id()
        second = self.upload_id(name="retried.csv")
        self.assertEqual(first, second)
        self.assertEqual(self.queries.get(first)["original_filename"], "data.csv")
        with self.context.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Source)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(SourceRecipe)), 1)
        self.process()
        self.assertFalse(asyncio.run(self.sources.worker.run_once()))
        self.assertEqual(len(self.requests), 1)

    def test_worker_reads_saved_original_after_client_closes_and_has_no_open_session_for_ollama(self) -> None:
        source_id = self.upload_id()
        self.client.close()
        # A fresh use case can finish without retaining an HTTP upload or worker queue.
        fresh = self.new_runtime()
        self.assertTrue(asyncio.run(fresh.worker.run_once()))
        saved = self.queries.get(source_id, include_content=True)
        self.assertEqual(saved["validation_status"], "ready")
        self.assertEqual(saved["recipe_data"]["proposal"], saved["recipe_data"]["effective"])
        self.assertTrue(saved["rules_snapshot"]["prompt_rules"])
        self.assertEqual(self.provider_sessions, [0])
        self.assertEqual(self.requests[0]["model"], "qwen3.5:9b")
        self.assertEqual(self.requests[0]["options"]["num_ctx"], 16384)
        self.assertIsNone(saved["rules_snapshot"]["metadata"]["profile"])
        self.assertFalse(saved["is_active"])
        with self.context.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Dataset)), 0)

    def test_listing_is_paginated_searchable_filtered_and_excludes_reports_and_bytes(self) -> None:
        self.upload_id(name="Mediciones.csv")
        self.upload_id(b"name,score\nOtro,1\nLuis,2\n", name="Otro.csv")
        self.process()
        response = self.client.get("/sources", params={"page": 1, "page_size": 1})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 2)
        self.assertEqual(len(response.json()["items"]), 1)
        item = response.json()["items"][0]
        self.assertEqual(set(item), {
            "id", "name", "file_type", "file_size_bytes", "created_at", "isActive",
            "processing_status", "validation_status",
        })
        for filters, expected in (({"name": "medic"}, 1), ({"processing_status": "uploaded"}, 1),
                                  ({"validation_status": "ready"}, 1), ({"name": "absent"}, 0)):
            self.assertEqual(self.client.get("/sources", params=filters).json()["total"], expected)
        for invalid in ({"page": 0}, {"page_size": 101}, {"processing_status": "unknown"}):
            self.assertEqual(self.client.get("/sources", params=invalid).status_code, 422)

    def test_detail_has_bounded_preview_and_reports_dataset_storage_limit(self) -> None:
        original = "name,score\n" + "\n".join(f"{'x' * 300},{row}" for row in range(1, 21)) + "\n"
        self.raw_response = json.dumps(preparation_recipe(data_end_row=21))
        source_id = self.upload_id(original.encode())
        self.process()
        detail = self.detail(source_id)
        self.assertIn(detail["validation_status"], {"ready", "ready_with_warnings"})
        self.assertEqual(detail["stage"], "prepared")
        self.assertLessEqual(len(detail["preview"][0]["rows"]), 10)
        self.assertLessEqual(len(detail["preview"][0]["rows"][0]["values"][0]), 200)
        self.assertFalse(detail["queryable"])
        self.assertFalse(detail["isActive"])
        self.assertEqual(detail["dataset_storage"]["status"], "not_implemented")
        self.assertNotIn("file_content", detail)

    def test_detail_and_decision_on_missing_source_return_404(self) -> None:
        source_id = uuid4()
        self.assertEqual(self.client.get(f"/sources/{source_id}").status_code, 404)
        self.assertEqual(self.client.post(f"/sources/{source_id}/reviews/absent/decisions", json={
            "action": "correct_transformations", "values": {"transformations": []},
        }).status_code, 404)

    def test_xlsx_upload_uses_the_persisted_workbook_and_existing_pipeline(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Records"
        sheet.append(["name", "score"])
        sheet.append(["Ana", 10])
        sheet.append(["Luis", 20])
        buffer = BytesIO()
        workbook.save(buffer)
        workbook.close()
        original = buffer.getvalue()
        self.raw_response = json.dumps(preparation_recipe(sheet_name="Records"))
        source_id = self.upload_id(original, name="records.xlsx")
        self.process()
        detail = self.detail(source_id)
        self.assertEqual(detail["file_type"], "xlsx")
        self.assertEqual(detail["validation_status"], "ready")
        self.assertEqual(detail["preview"][0]["sheet_name"], "Records")
        self.assertEqual(self.queries.get(source_id, include_content=True)["file_content"], original)

    def test_identity_clarification_is_source_scoped_and_revalidates_without_execution_or_ollama(self) -> None:
        note = "No sé si hay alguna repetida con otra campaña."
        content = f"name,score,note\nAna,10,{note}\nLuis,20,\n".encode()
        self.raw_response = json.dumps(preparation_recipe(data_end_column=3))
        source_id = self.upload_id(content)
        with patch.object(self.processor, "execute", wraps=self.processor.execute) as execution:
            self.process()
            before = self.detail(source_id)
            review = next(item for item in before["reviews"] if item["code"] == "source_record_identity_uncertainty")
            response = self.decide(source_id, review, {
                "action": "clarify_record_identity", "values": {
                    "identity": "distinct", "statement": "Confirmo que son muestras distintas.",
                },
            })
            self.assertEqual(response.status_code, 202, response.text)
            self.process()
            self.assertEqual(execution.call_count, 1)
        after = self.detail(source_id)
        self.assertIn(after["validation_status"], {"ready", "ready_with_warnings"})
        self.assertEqual(after["reviews"], [])
        self.assertEqual(after["original_proposal"], before["original_proposal"])
        self.assertEqual(len(self.requests), 1)
        clarification = after["rules"]["identity_clarifications"][0]
        self.assertEqual(clarification["sheet_name"], "data")
        self.assertEqual(clarification["original_note"], note)
        self.assertIn(str(source_id), clarification["source_reference"])
        # An unrelated source never inherits the particular clarification.
        other_id = self.upload_id(content.replace(b"Ana", b"Otra"), name="other.csv")
        self.process()
        other = self.detail(other_id)
        self.assertEqual(other["validation_status"], "review_required")
        self.assertEqual(other["rules"]["identity_clarifications"], [])

    def test_review_identifiers_and_scope_are_stable_across_reads(self) -> None:
        source_id, review = self.review_source()
        detail = self.detail(source_id)
        same = next(item for item in detail["reviews"] if item["code"] == "conversion_failures")
        self.assertEqual(same, review)
        self.assertEqual(review["sheet_name"], "data")
        self.assertEqual(review["block_id"], "observations")
        self.assertTrue(review["message"])
        self.assertTrue(review["range"])
        self.assertIn("correct_transformations", [option["action"] for option in review["options"]])
        self.assertFalse(detail["queryable"])
        self.assertFalse(detail["isActive"])

    def test_decision_preserves_proposal_reexecutes_and_revalidates_without_calling_ollama(self) -> None:
        source_id, review = self.review_source()
        original = deepcopy(self.detail(source_id)["original_proposal"])
        response = self.decide(source_id, review)
        self.assertEqual(response.status_code, 202, response.text)
        queued = self.detail(source_id)
        self.assertEqual(queued["processing_status"], "uploaded")
        self.assertEqual(queued["validation_status"], "pending")
        self.assertEqual(queued["stage"], "queued_resume")
        self.assertEqual(queued["original_proposal"], original)
        self.assertEqual(queued["effective_recipe"]["sheets"][0]["transformations"], [])
        self.process()
        final = self.detail(source_id)
        self.assertEqual(final["validation_status"], "ready")
        self.assertEqual(final["reviews"], [])
        self.assertEqual(final["original_proposal"], original)
        self.assertEqual(len(final["decisions"]), 1)
        self.assertEqual(final["decisions"][0]["scope"]["source_id"], str(source_id))
        self.assertEqual(len(self.requests), 1)
        saved = self.queries.get(source_id, include_content=True)
        self.assertIn(b"invalid", saved["file_content"])
        self.assertEqual(saved["validation_report"]["evaluation"]["llm_response"]["content"], self.raw_response)

    def test_repeated_decision_is_idempotent_and_different_or_stale_decisions_conflict(self) -> None:
        source_id, review = self.review_source()
        first = self.decide(source_id, review)
        duplicate = self.decide(source_id, review)
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(duplicate.status_code, 202, duplicate.text)
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(duplicate.json()["revision"], first.json()["revision"])
        conflicting = self.decide(source_id, review, {"action": "correct_transformations", "values": {"transformations": [
            {"operation": "strip_strings"},
        ]}})
        self.assertEqual(conflicting.status_code, 409, conflicting.text)
        self.process()
        repeat_after_processing = self.decide(source_id, review)
        self.assertEqual(repeat_after_processing.status_code, 202, repeat_after_processing.text)
        self.assertTrue(repeat_after_processing.json()["duplicate"])
        self.assertEqual(repeat_after_processing.json()["validation_status"], "ready")
        self.assertEqual(len(self.detail(source_id)["decisions"]), 1)
        stale = self.client.post(f"/sources/{source_id}/reviews/unrelated/decisions", json={
            "action": "correct_transformations", "values": {"transformations": []},
        })
        self.assertEqual(stale.status_code, 409, stale.text)

    def test_invalid_decision_actions_values_and_executable_operations_are_rejected(self) -> None:
        source_id, review = self.review_source()
        before = self.detail(source_id)
        for payload in (
            {"action": "accept_anyway", "values": {}},
            {"action": "correct_transformations", "values": {"other": []}},
            {"action": "correct_transformations", "values": {"transformations": "anything"}},
            {"action": "correct_transformations", "values": {"transformations": [{"operation": "run_python", "code": "pass"}]}},
        ):
            response = self.decide(source_id, review, payload)
            self.assertEqual(response.status_code, 400, response.text)
        response = self.decide(source_id, review, {"action": "correct_transformations", "values": {}, "model": "other"})
        self.assertEqual(response.status_code, 422, response.text)
        after = self.detail(source_id)
        self.assertEqual(after["revision"], before["revision"])
        self.assertEqual(after["decisions"], [])
        self.assertEqual(after["processing_status"], "review_required")

    def test_technical_provider_failure_is_persisted_without_user_review(self) -> None:
        self.provider_error = httpx.ReadTimeout("Ollama simulado sin respuesta")
        source_id = self.upload_id()
        self.process()
        detail = self.detail(source_id)
        self.assertEqual((detail["processing_status"], detail["validation_status"]), ("failed", "failed"))
        self.assertTrue(detail["technical_errors"])
        self.assertEqual(detail["reviews"], [])
        self.assertFalse(detail["queryable"])

    def test_abandoned_processing_is_registered_by_next_worker(self) -> None:
        source_id = self.upload_id()
        self.assertEqual(self.commands.claim_next(), source_id)
        fresh = self.new_runtime()
        self.assertFalse(asyncio.run(fresh.worker.run_once()))
        detail = self.detail(source_id)
        self.assertEqual(detail["stage"], "interrupted")
        self.assertEqual(detail["processing_status"], "failed")
        self.assertEqual(detail["validation_status"], "failed")
        self.assertTrue(any("interrupted" in item["code"] for item in detail["technical_errors"]))
        self.assertFalse(self.requests)

    def test_openapi_exposes_only_four_source_operations_and_file_only_upload(self) -> None:
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(set(schema["paths"]), {"/sources", "/sources/{id}", "/sources/{id}/reviews/{review_id}/decisions"})
        self.assertNotIn("TabularEvaluationResponse", schema["components"]["schemas"])
        scope = {"type": "http", "path": "/experimental/tabular/evaluate", "root_path": "", "method": "POST"}
        for route in self.client.app.routes:
            self.assertEqual(route.matches(scope)[0], Match.NONE)
        self.assertEqual(self.client.post("/experimental/tabular/evaluate").status_code, 404)
        operation = schema["paths"]["/sources"]["post"]
        self.assertNotIn("parameters", operation)
        body_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
        upload_schema = schema["components"]["schemas"][body_schema["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(set(upload_schema["properties"]), {"file"})
        self.assertIn("202", operation["responses"])


if __name__ == "__main__":
    unittest.main()
