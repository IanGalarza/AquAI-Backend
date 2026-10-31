"""Flujo HTTP y fallos externos sin servicios ni archivos privados."""

import asyncio
import unittest
from datetime import date
from io import BytesIO
from uuid import UUID

from openpyxl import Workbook
from sqlalchemy import MetaData, Table, event, inspect, select

from aquai.application.contracts.ingestion import (
    SemanticDescription,
    StructureSuggestion,
)
from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.persistence.mapping import metadata
from aquai.infrastructure.processing.tabular.processor import SimpleTabularProcessor
from aquai.infrastructure.providers.sources import build_source_runtime
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from tests.sourceDatabase import SourceTestDbContext


class DescriptionDouble:
    def __init__(self, fail=False):
        self.fail = fail

    async def describe(self, table):
        if self.fail:
            raise RuntimeError("offline")
        return SemanticDescription(
            description=table.description,
            columns=table.columns,
        )

    async def suggest_structure(self, candidates):
        if self.fail:
            raise RuntimeError("offline")
        return []


class IndexDouble:
    def __init__(self, fail=False):
        self.fail = fail
        self.sources = []

    async def index_source(self, source):
        if self.fail:
            raise RuntimeError("offline")
        self.sources.append(source)

    async def set_source_active(self, source_id, active):
        pass


class SourceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = SourceTestDbContext()
        self.addCleanup(self.context.dispose)
        self.commands = SqlAlchemySourceCommands(self.context)
        self.queries = SqlAlchemySourceQueries(self.context)
        self.settings = TabularSettings(_env_file=None)
        self.reader = SimpleTabularProcessor(self.settings)
        self.index = IndexDouble()
        self.runtime = self.build()

    def build(self, semantics=None):
        return build_source_runtime(
            self.commands,
            self.queries,
            self.reader,
            semantics or DescriptionDouble(),
            self.index,
            max_upload_bytes=self.settings.max_upload_bytes,
        )

    async def upload(self):
        return await self.runtime.upload.handle(
            UploadSourceRequest(
                filename="test.csv",
                content=b"code,value\n001,10\n002,20\n002,20\n",
            )
        )

    def physical_rows(self, dataset):
        table = Table(
            dataset["physical_table_name"],
            MetaData(),
            autoload_with=self.context.engine,
        )
        with self.context.engine.connect() as connection:
            return list(
                connection.execute(
                    select(table).order_by(table.c._aquai_row),
                ).mappings()
            )

    async def test_full_rows_duplicate_upload_and_idempotent_reindex(self):
        accepted = await self.upload()
        duplicate = await self.upload()
        self.assertEqual(accepted.id, duplicate.id)
        self.assertTrue(duplicate.duplicate)
        self.assertTrue(await self.runtime.worker.run_once())
        self.assertFalse(await self.runtime.worker.run_once())
        state = self.queries.get(accepted.id)
        self.assertEqual(state["processing_status"], "ready")
        self.assertEqual(state["semantic_index_status"], "ready")
        self.assertEqual(state["datasets"][0]["row_count"], 3)
        rows = self.physical_rows(state["datasets"][0])
        self.assertEqual([row["code"] for row in rows], ["001", "002", "002"])
        dataset_id = state["datasets"][0]["id"]
        self.commands.request_reindex(accepted.id)
        await self.runtime.worker.run_once()
        self.assertEqual(self.queries.get(accepted.id)["datasets"][0]["id"], dataset_id)
        self.assertEqual(len(self.physical_rows(state["datasets"][0])), 3)

    async def test_qwen_failure_keeps_confirmed_sql_and_indexes_basic_description(self):
        self.runtime = self.build(DescriptionDouble(fail=True))
        accepted = await self.upload()
        await self.runtime.worker.run_once()
        state = self.queries.get(accepted.id)
        self.assertEqual(state["processing_status"], "ready_with_warnings")
        self.assertEqual(state["semantic_index_status"], "ready")
        self.assertEqual(len(self.physical_rows(state["datasets"][0])), 3)
        self.assertTrue(state["processing_report"]["warnings"])

    async def test_index_failure_visible_and_sql_preserved_then_retry_without_reading(self):
        self.index.fail = True
        accepted = await self.upload()
        await self.runtime.worker.run_once()
        state = self.queries.get(accepted.id)
        self.assertEqual(state["semantic_index_status"], "failed")
        self.assertIn(state["processing_status"], {"ready", "ready_with_warnings"})
        self.assertEqual(len(self.physical_rows(state["datasets"][0])), 3)
        self.index.fail = False
        self.reader.prepare = lambda *_: self.fail("Reindex must not read CSV")
        self.commands.request_reindex(accepted.id)
        await self.runtime.worker.run_once()
        self.assertEqual(self.queries.get(accepted.id)["semantic_index_status"], "ready")

    async def test_structural_assistance_is_bounded_and_fallback_keeps_normal_sql(self):
        def content(value):
            workbook = Workbook()
            workbook.active.append(["codigo", "valor"])
            workbook.active.append(["001", value])
            matrix = workbook.create_sheet("matriz")
            matrix.append(["variable", date(2024, 1, 1), date(2024, 1, 2)])
            matrix.append(["sitio", "A", "B"])
            matrix.append(["masa", 10, "nd"])
            stream = BytesIO()
            workbook.save(stream)
            workbook.close()
            return stream.getvalue()

        class AssistedDescription(DescriptionDouble):
            calls = 0

            async def suggest_structure(self, candidates):
                self.calls += 1
                hint = candidates[0]
                return [StructureSuggestion(hint["region_id"], "columns", 1, 1, 2)]

        semantics = AssistedDescription()
        self.runtime = self.build(semantics)
        original = content(7)
        accepted = await self.runtime.upload.handle(
            UploadSourceRequest(filename="multi.xlsx", content=original)
        )
        await self.runtime.worker.run_once()
        state = self.queries.get(accepted.id, include_content=True)
        self.assertEqual(semantics.calls, 1)
        self.assertEqual(state["processing_report"]["structure_assistance"]["accepted"], 1)
        self.assertEqual(sorted(dataset["row_count"] for dataset in state["datasets"]), [1, 2])
        self.assertEqual(state["file_content"], original)
        matrix = next(
            dataset
            for dataset in state["datasets"]
            if dataset["source_locator"].get("orientation") == "columns"
        )
        self.assertEqual(
            [row["_aquai_source_column"] for row in self.physical_rows(matrix)], [2, 3]
        )

        self.runtime = self.build(DescriptionDouble(fail=True))
        accepted = await self.runtime.upload.handle(
            UploadSourceRequest(filename="fallback.xlsx", content=content(8))
        )
        await self.runtime.worker.run_once()
        fallback = self.queries.get(accepted.id)
        self.assertEqual([dataset["row_count"] for dataset in fallback["datasets"]], [1])
        self.assertEqual(fallback["semantic_index_status"], "ready")
        self.assertTrue(
            any(
                "estructural no disponible" in warning
                for warning in fallback["processing_report"]["warnings"]
            )
        )

    async def test_reprocessing_keeps_ids_and_rolls_back_tables_on_failure(self):
        metadata.create_all(self.context.engine)
        accepted = await self.upload()
        await self.runtime.worker.run_once()
        before = self.queries.get(accepted.id)["datasets"][0]
        await self.runtime.worker.reprocess(accepted.id)
        after = self.queries.get(accepted.id)["datasets"][0]
        self.assertEqual(after["id"], before["id"])
        self.assertEqual(after["semantic_point_id"], before["semantic_point_id"])
        self.assertEqual(after["version"], 2)
        self.assertFalse(self.queries.usable_dataset_ids([after["id"]], versions={after["id"]: 1}))
        self.assertEqual(
            self.queries.usable_dataset_ids([after["id"]], versions={after["id"]: 2}), {after["id"]}
        )
        self.assertEqual(len(self.physical_rows(after)), 3)
        self.assertNotIn(
            before["physical_table_name"], inspect(self.context.engine).get_table_names()
        )

        def reject_drop(connection, cursor, statement, parameters, context, executemany):
            if statement.lstrip().startswith("DROP TABLE") and "dataset_" in statement:
                raise RuntimeError("Fallo después del reemplazo")

        event.listen(self.context.engine, "after_cursor_execute", reject_drop)
        try:
            with self.assertRaisesRegex(RuntimeError, "reemplazo"):
                await self.runtime.worker.reprocess(accepted.id)
        finally:
            event.remove(self.context.engine, "after_cursor_execute", reject_drop)
        unchanged = self.queries.get(accepted.id)["datasets"][0]
        self.assertEqual(unchanged["physical_table_name"], after["physical_table_name"])
        self.assertEqual(unchanged["version"], 2)
        self.assertEqual(len(self.physical_rows(unchanged)), 3)
        self.assertEqual(
            sum(
                name.startswith("dataset_")
                for name in inspect(self.context.engine).get_table_names()
            ),
            1,
        )

    async def test_activation_never_enables_unprepared_or_inactive_datasets(self):
        accepted = await self.upload()
        detail = await self.runtime.activate.handle(accepted.id, True)
        self.assertFalse(detail.queryable)
        await self.runtime.worker.run_once()
        dataset_id = self.queries.get(accepted.id)["datasets"][0]["id"]
        self.assertEqual(self.queries.usable_dataset_ids([dataset_id]), {dataset_id})
        detail = await self.runtime.activate.handle(accepted.id, False)
        self.assertFalse(detail.queryable)
        self.assertFalse(self.queries.usable_dataset_ids([dataset_id]))
        public_detail = await self.runtime.get.handle(accepted.id)
        self.assertFalse(public_detail.datasets[0].queryable)

    async def test_cancelled_semantics_keeps_sql_and_recovers_pending_index(self):
        entered = asyncio.Event()

        class WaitingDescription:
            async def describe(self, table):
                entered.set()
                await asyncio.Event().wait()

        self.runtime = self.build(WaitingDescription())
        accepted = await self.upload()
        task = asyncio.create_task(self.runtime.worker.run_once())
        await asyncio.wait_for(entered.wait(), timeout=3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        state = self.queries.get(accepted.id)
        self.assertIn(state["processing_status"], {"ready", "ready_with_warnings"})
        self.assertEqual(len(self.physical_rows(state["datasets"][0])), 3)
        self.runtime = self.build()
        await self.runtime.worker.run_once()
        self.assertEqual(self.queries.get(accepted.id)["semantic_index_status"], "ready")

        # El cierre debe terminar incluso si una operación consume su cancelación.
        worker = self.runtime.worker
        entered.clear()

        async def consume_cancel():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return False

        worker.run_once = consume_cancel
        await worker.start()
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.wait_for(worker.stop(), timeout=2)

    def test_four_http_routes_contract_and_invalid_uploads(self):
        from fastapi.testclient import TestClient

        from aquai.main import create_app

        app = create_app(sources=self.runtime)
        with TestClient(app) as client:
            response = client.post("/sources", files={"file": ("safe.csv", b"id,value\n001,3\n")})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(set(response.json()), {"id", "duplicate"})
            source_id = response.json()["id"]
            listing = client.get("/sources")
            self.assertEqual(listing.status_code, 200)
            self.assertNotIn("duplicate", listing.json()["items"][0])
            detail = client.get("/sources/" + source_id).json()
            self.assertNotIn("file_content", detail)
            self.assertNotIn("reviews", detail)
            self.assertTrue({"mime_type", "updated_at", "duplicate", "name"}.isdisjoint(detail))
            activation = client.patch("/sources/" + source_id, json={"isActive": False})
            self.assertEqual(activation.status_code, 200)
            self.assertEqual(
                set(activation.json()),
                {"id", "isActive", "queryable", "processing_status", "semantic_index_status"},
            )
            self.assertFalse(activation.json()["isActive"])
            self.assertEqual(
                client.patch("/sources/" + source_id, json={"isActive": "yes"}).status_code, 422
            )
            self.assertEqual(
                client.post("/sources", files={"file": ("x.pdf", b"%PDF")}).status_code, 400
            )
            self.assertEqual(
                client.post("/sources", files={"file": ("x.csv", b"id\x00")}).status_code, 400
            )
            self.runtime.upload.max_upload_bytes = 32
            self.assertEqual(
                client.post("/sources", files={"file": ("x.csv", b"a" * 40)}).status_code, 413
            )
            self.assertEqual(
                client.post("/sources", files={"file": ("x.csv", b"a" * 70000)}).status_code, 413
            )
            self.assertEqual(client.get("/sources/" + str(UUID(int=0))).status_code, 404)
            paths = app.openapi()["paths"]
            self.assertEqual(set(paths), {"/sources", "/sources/{id}"})
            self.assertEqual(set(paths["/sources/{id}"]), {"get", "patch"})
