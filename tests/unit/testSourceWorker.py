"""Worker durable con originales en SQLite y proveedor simulado."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from aquai.application.contracts.tabular.results import LLMResponse, TabularEvaluation, ValidationResult
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.dtos.requests.listSources import ListSourcesRequest
from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.dtos.responses.listSources import SourceListResponse
from aquai.application.dtos.responses.sourceDetails import SourceBlock, SourcePreview
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.application.services.tabularRules import TabularRuleService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.application.useCases.sources import ProcessSourceHandler
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog
from aquai.infrastructure.providers.sources import build_source_runtime
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from tests.sourceDatabase import SourceTestDbContext


class GatedProvider:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate_recipe(self, summary, model, known_rules):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            await self.release.wait()
            return LLMResponse(model=model, content=json.dumps({
                "status": "ready", "requires_human_review": False, "warnings": [],
                "sheets": [{"sheet_name": summary.sheets[0].name, "block_id": "records",
                            "role": "observations", "header_rows": [1], "data_start_row": 2,
                            "data_end_row": summary.sheets[0].rows, "data_start_column": 1,
                            "data_end_column": summary.sheets[0].columns, "transformations": []}],
            }))
        finally:
            self.active -= 1


class SourceWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = SourceTestDbContext()
        self.queries = SqlAlchemySourceQueries(self.context)
        self.provider = GatedProvider()
        self.evaluator = TabularEvaluationService(
            PandasTabularProcessor(), self.provider, default_model="qwen3.5:9b", max_upload_bytes=1024,
            rule_service=TabularRuleService(FileTabularRuleCatalog()),
        )
        self.runtimes = []

    def make_runtime(self):
        runtime = build_source_runtime(
            SqlAlchemySourceCommands(self.context), self.queries, self.evaluator,
            max_upload_bytes=1024, poll_seconds=0.01,
        )
        self.runtimes.append(runtime)
        return runtime

    async def asyncTearDown(self):
        for runtime in self.runtimes:
            await runtime.worker.stop()
        self.context.dispose()

    async def await_state(self, runtime, source_id, state):
        async def poll():
            while True:
                source = await runtime.get.handle(source_id)
                self.assertIsInstance(source, SourceDetailResponse)
                if source.processing_status == state:
                    return source
                await asyncio.sleep(0.01)
        return await asyncio.wait_for(poll(), 5)

    async def test_background_workers_serialize_sources_and_keep_requests_independent(self):
        first_runtime, second_runtime = self.make_runtime(), self.make_runtime()
        first = await first_runtime.upload.handle(UploadSourceRequest("first.csv", b"site,value\nA,1\n"))
        second = await first_runtime.upload.handle(UploadSourceRequest("second.csv", b"site,value\nB,2\n"))
        self.assertIsInstance(first, SourceAcceptedResponse)
        self.assertIsInstance(second, SourceAcceptedResponse)
        await first_runtime.worker.start()
        await second_runtime.worker.start()
        await asyncio.wait_for(self.provider.entered.wait(), 2)
        first_snapshot = self.queries.get(first.id, include_content=True)
        self.assertEqual(first_snapshot["processing_status"], "processing")
        self.assertEqual(first_snapshot["recipe_data"]["_workflow"]["stage"], "generating_recipe")
        self.assertEqual(first_snapshot["file_content"], b"site,value\nA,1\n")
        self.assertEqual(self.queries.get(second.id)["processing_status"], "uploaded")
        # GET y otra carga terminan mientras el proveedor está detenido.
        listing = await asyncio.wait_for(second_runtime.list.handle(ListSourcesRequest()), 2)
        self.assertIsInstance(listing, SourceListResponse)
        self.assertEqual(listing.total, 2)
        third = await asyncio.wait_for(second_runtime.upload.handle(
            UploadSourceRequest("third.csv", b"site,value\nC,3\n"),
        ), 2)
        self.assertEqual(third.processing_status, "uploaded")
        self.assertEqual(self.provider.calls, 1)
        self.provider.release.set()
        for accepted in (first, second, third):
            detail = await self.await_state(first_runtime, accepted.id, "ready")
            self.assertFalse(detail.queryable)
            self.assertFalse(detail.is_active)
            self.assertIsInstance(detail.blocks[0], SourceBlock)
            self.assertIsInstance(detail.preview[0], SourcePreview)
        self.assertEqual(self.provider.calls, 3)
        self.assertEqual(self.provider.max_active, 1)

    async def test_shutdown_records_interruption_and_preserves_inspection_rules_and_original(self):
        runtime = self.make_runtime()
        original = b"site,value\nA,1\n"
        accepted = await runtime.upload.handle(UploadSourceRequest("first.csv", original))
        await runtime.worker.start()
        await asyncio.wait_for(self.provider.entered.wait(), 2)
        await asyncio.wait_for(runtime.worker.stop(), 2)
        detail = await runtime.get.handle(accepted.id)
        self.assertEqual(detail.processing_status, "failed")
        self.assertEqual(detail.validation_status, "failed")
        self.assertEqual(detail.stage, "interrupted")
        self.assertEqual(detail.technical_errors[0].code, "interrupted")
        self.assertEqual(detail.reviews, [])
        saved = self.queries.get(accepted.id, include_content=True)
        self.assertEqual(saved["file_content"], original)
        self.assertTrue(saved["validation_report"]["evaluation"]["structural_summary"]["sheets"])
        self.assertTrue(saved["rules_snapshot"]["metadata"]["common_rules"])
        # Otra instancia ve el fallo persistido y no repite la LLM al arrancar.
        replacement = self.make_runtime()
        self.provider.release.set()
        self.assertFalse(await replacement.worker.run_once())
        self.assertEqual(self.provider.calls, 1)

    async def test_existing_worker_recovers_failure_left_by_another_process(self):
        runtime = self.make_runtime()
        self.assertFalse(await runtime.worker.run_once())
        accepted = await runtime.upload.handle(UploadSourceRequest("first.csv", b"site,value\nA,1\n"))
        abandoned = SqlAlchemySourceCommands(self.context)
        with abandoned.worker_lock():
            self.assertEqual(abandoned.claim_next(), accepted.id)
        # También se recupera después de la primera adquisición de este worker.
        self.assertFalse(await runtime.worker.run_once())
        detail = await runtime.get.handle(accepted.id)
        self.assertEqual(detail.stage, "interrupted")
        self.assertEqual(detail.processing_status, "failed")
        self.assertEqual(self.provider.calls, 0)

    async def test_review_required_without_actionable_reviews_keeps_awaiting_review_stage(self):
        source_id = uuid4()
        commands = Mock()
        commands.save.return_value = True
        queries = Mock()
        queries.get.return_value = {
            "original_filename": "data.csv", "file_content": b"site,value\nA,1\n",
            "recipe_data": {"proposal": None, "effective": None, "_workflow": {"stage": "uploaded"}},
            "validation_report": {}, "rules_snapshot": {},
        }
        evaluation = TabularEvaluation(
            model="offline", structural_summary=StructuralSummary(
                file_name="data.csv", file_type="csv", file_size_bytes=15, sheets=[],
            ), validation=ValidationResult(status=EvaluationStatus.REVIEW_REQUIRED),
        )
        evaluator = Mock(process_source=AsyncMock(return_value=evaluation))
        reviews = Mock()
        reviews.build_reviews.return_value = []
        await ProcessSourceHandler(commands, queries, evaluator, reviews=reviews).handle(source_id)
        saved = commands.save.call_args.kwargs
        self.assertEqual(saved["processing_status"], "review_required")
        self.assertEqual(saved["validation_status"], "review_required")
        self.assertEqual(saved["recipe_data"]["_workflow"]["stage"], "awaiting_review")
        self.assertEqual(saved["validation_report"]["reviews"], [])


if __name__ == "__main__":
    unittest.main()
