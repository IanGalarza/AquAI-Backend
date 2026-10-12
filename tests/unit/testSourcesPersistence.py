"""Gestión de fuentes sobre una BD aislada, sin Ollama ni PostgreSQL locales."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import unittest
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy import event, func, select
from sqlalchemy.dialects import postgresql

from aquai.application.errors import SourceConflictError, SourceNotFoundError
from aquai.domain.models import Dataset, Source, SourceRecipe
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands, managed_workflow_predicate
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from tests.sourceDatabase import SourceTestDbContext


class SourcesPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = SourceTestDbContext()
        self.addCleanup(self.context.dispose)
        self.commands = SqlAlchemySourceCommands(self.context)
        self.queries = SqlAlchemySourceQueries(self.context)

    def upload(self, name: str = "Water.csv", content: bytes = b"station,pH\nA,7.2\n"):
        source = Source(
            name=name, original_filename=name, file_type="CSV", mime_type="text/csv",
            file_size_bytes=len(content), file_content=content, is_active=False, processing_status_id=1,
        )
        document = {
            "proposal": None, "effective": None,
            "_workflow": {"stage": "uploaded", "revision": 0, "content_sha256": sha256(content).hexdigest()},
        }
        return self.commands.create_upload(source, document)

    def test_upload_commits_original_initial_state_and_one_recipe_without_datasets(self) -> None:
        source_id, created = self.upload()
        self.assertTrue(created)
        snapshot = self.queries.get(source_id, include_content=True)
        self.assertEqual(snapshot["file_content"], b"station,pH\nA,7.2\n")
        self.assertEqual(snapshot["processing_status"], "uploaded")
        self.assertEqual(snapshot["validation_status"], "pending")
        self.assertFalse(snapshot["is_active"])
        with self.context.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(Source)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(SourceRecipe)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(Dataset)), 0)

    def test_concurrent_upload_retries_return_the_original_source_and_recipe(self) -> None:
        with ThreadPoolExecutor(max_workers=4) as executor:
            uploads = list(executor.map(lambda _: self.upload(), range(8)))
        self.assertEqual(len({result[0] for result in uploads}), 1)
        self.assertEqual(sum(result[1] for result in uploads), 1)
        self.assertEqual(self.queries.list()["total"], 1)
        with self.context.session() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(SourceRecipe)), 1)

    def test_invalid_fingerprint_does_not_persist_anything(self) -> None:
        source = Source(
            name="x", original_filename="x.csv", file_type="CSV", mime_type="text/csv",
            file_size_bytes=1, file_content=b"a", is_active=False, processing_status_id=1,
        )
        with self.assertRaises(ValueError):
            self.commands.create_upload(source, {"_workflow": {"content_sha256": "wrong"}})
        self.assertEqual(self.queries.list()["total"], 0)

    def test_listing_selects_metadata_only_and_searches_literal_wildcards(self) -> None:
        self.upload("Water_100%.csv", b"a,b\n1,2\n")
        self.upload("waterZ100X.csv", b"a,b\n3,4\n")
        queries = []

        def record(connection, cursor, statement, parameters, context, executemany):
            queries.append(statement)

        event.listen(self.context.engine, "before_cursor_execute", record)
        self.addCleanup(event.remove, self.context.engine, "before_cursor_execute", record)
        listing = self.queries.list(search="water_100%", page_size=1)
        self.assertEqual(listing["total"], 1)
        self.assertEqual(listing["items"][0]["name"], "Water_100%.csv")
        for statement in queries:
            for column in ("fileContent", "recipeData", "rulesSnapshot", "validationReport"):
                self.assertNotIn(column, statement)
        self.assertNotIn("file_content", listing["items"][0])
        self.assertNotIn("validation_report", listing["items"][0])

    def test_pagination_combines_processing_and_validation_filters(self) -> None:
        first, _ = self.upload("first.csv", b"a,b\n1,2\n")
        self.upload("second.csv", b"a,b\n3,4\n")
        self.commands.save(first, processing_status="review_required", validation_status="review_required")
        result = self.queries.list(processing_status="review_required", validation_status="review_required")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["id"], first)
        self.assertEqual(self.queries.list(page=2, page_size=1)["total"], 2)
        self.assertEqual(len(self.queries.list(page=2, page_size=1)["items"]), 1)

    def test_detail_documents_are_detached_and_original_is_loaded_only_explicitly(self) -> None:
        source_id, _ = self.upload()
        snapshot = self.queries.get(source_id)
        self.assertNotIn("file_content", snapshot)
        snapshot["recipe_data"]["_workflow"]["stage"] = "changed outside the database"
        self.assertEqual(self.queries.get(source_id)["recipe_data"]["_workflow"]["stage"], "uploaded")
        with self.assertRaises(SourceNotFoundError):
            self.queries.get(uuid4())

    def test_worker_lock_is_shared_between_instances_and_released(self) -> None:
        second = SqlAlchemySourceCommands(self.context)
        with self.commands.worker_lock() as first_acquired:
            self.assertTrue(first_acquired)
            with second.worker_lock() as second_acquired:
                self.assertFalse(second_acquired)
        with second.worker_lock() as second_acquired:
            self.assertTrue(second_acquired)

    def test_historical_sources_do_not_block_or_get_changed_by_the_managed_queue(self) -> None:
        historical = []
        with self.context.session() as session:
            for status in (1, 2):
                source = Source(
                    name=f"legacy-{status}.csv", original_filename=f"legacy-{status}.csv",
                    file_type="csv", mime_type="text/csv", file_size_bytes=4,
                    file_content=b"x\n1\n", is_active=True, processing_status_id=status,
                )
                session.add(source)
                historical.append(source.id)
            session.commit()
        source_id, _ = self.upload("managed.csv", b"x\n2\n")
        with self.commands.worker_lock():
            self.assertEqual(self.commands.recover_interrupted(), 0)
            self.assertEqual(self.commands.claim_next(), source_id)
        self.assertEqual(self.queries.get(historical[0])["processing_status"], "uploaded")
        self.assertEqual(self.queries.get(historical[1])["processing_status"], "processing")
        self.assertTrue(self.queries.get(historical[1])["is_active"])
        self.assertIsNone(self.queries.get(historical[1])["recipe_id"])

    def test_historical_recipe_without_managed_workflow_is_not_claimed_or_recovered(self) -> None:
        source = Source(
            name="historical.csv", original_filename="historical.csv", file_type="csv",
            mime_type="text/csv", file_size_bytes=4, file_content=b"x\n1\n",
            is_active=True, processing_status_id=2,
        )
        with self.context.session() as session:
            session.add(source)
            session.add(SourceRecipe(source_id=source.id, recipe_data={"sheets": []}, rules_snapshot={}))
            session.commit()
        with self.commands.worker_lock():
            self.assertEqual(self.commands.recover_interrupted(), 0)
            self.assertIsNone(self.commands.claim_next())
        self.assertEqual(self.queries.get(source.id)["processing_status"], "processing")

    def test_malformed_historical_workflow_versions_do_not_block_the_managed_queue(self) -> None:
        historical_ids = []
        with self.context.session() as session:
            for version in ("unexpected", "1", True, None):
                source = Source(
                    name="legacy.csv", original_filename="legacy.csv", file_type="csv",
                    mime_type="text/csv", file_size_bytes=4, file_content=b"x\n1\n",
                    is_active=True, processing_status_id=1,
                )
                session.add(source)
                session.add(SourceRecipe(
                    source_id=source.id, recipe_data={"_workflow": {"version": version}}, rules_snapshot={},
                ))
                historical_ids.append(source.id)
            session.commit()
        source_id, _ = self.upload("managed.csv", b"x\n2\n")
        with self.commands.worker_lock():
            self.assertEqual(self.commands.claim_next(), source_id)
            self.assertIsNone(self.commands.claim_next())
        for historical_id in historical_ids:
            self.assertEqual(self.queries.get(historical_id)["processing_status"], "uploaded")

    def test_postgresql_managed_queue_uses_jsonb_containment_without_integer_cast(self) -> None:
        statement = select(Source.id).join(SourceRecipe, SourceRecipe.source_id == Source.id).where(
            managed_workflow_predicate("postgresql"),
        )
        compiled = statement.compile(dialect=postgresql.dialect())
        self.assertIn("@>", str(compiled))
        self.assertIn("::JSONB", str(compiled))
        self.assertNotIn("CAST", str(compiled))
        self.assertIn({"_workflow": {"version": 1}}, compiled.params.values())

    def test_claim_and_checkpoint_are_committed_and_preserve_original(self) -> None:
        source_id, _ = self.upload()
        with self.commands.worker_lock() as acquired:
            self.assertTrue(acquired)
            self.assertEqual(self.commands.claim_next(), source_id)
            self.assertIsNone(self.commands.claim_next())
            snapshot = self.queries.get(source_id)
            self.assertEqual(snapshot["processing_status"], "processing")
            self.assertEqual(snapshot["recipe_data"]["_workflow"]["stage"], "inspection")
            document = snapshot["recipe_data"]
            document["proposal"] = {"sheet": "CSV", "range": "A1:B2"}
            document["effective"] = deepcopy(document["proposal"])
            self.commands.save(
                source_id, recipe_data=document, rules_snapshot={"scope": "source"},
                validation_report={"reviews": [{"id": "stable-review"}]},
                processing_status="review_required", validation_status="review_required",
                expected_processing_status="processing",
            )
        snapshot = self.queries.get(source_id, include_content=True)
        self.assertEqual(snapshot["file_content"], b"station,pH\nA,7.2\n")
        self.assertEqual(snapshot["recipe_data"]["proposal"], {"sheet": "CSV", "range": "A1:B2"})
        self.assertEqual(snapshot["validation_status"], "review_required")
        self.assertFalse(snapshot["is_active"])

    def test_save_rejects_stale_state_and_rolls_back_invalid_updates(self) -> None:
        source_id, _ = self.upload()
        self.assertFalse(self.commands.save(
            source_id, processing_status="ready", expected_processing_status="processing",
        ))
        with self.assertRaises(ValueError):
            self.commands.save(source_id, processing_status="processing", validation_status="unknown")
        self.assertEqual(self.queries.get(source_id)["processing_status"], "uploaded")
        with self.assertRaises(SourceNotFoundError):
            self.commands.save(uuid4(), processing_status="processing")

    def test_save_distinguishes_omitted_documents_from_explicit_null(self) -> None:
        source_id, _ = self.upload()
        report = {"reviews": [{"id": "review-1"}], "optional": None}
        self.commands.save(source_id, validation_report=report)
        report["reviews"].append({"id": "changed outside the database"})
        self.commands.save(source_id, processing_status="processing")
        snapshot = self.queries.get(source_id)
        self.assertEqual(snapshot["validation_report"], {
            "reviews": [{"id": "review-1"}], "optional": None,
        })
        original_recipe = snapshot["recipe_data"]
        self.commands.save(source_id, validation_report=None)
        cleared = self.queries.get(source_id)
        self.assertIn("validation_report", cleared)
        self.assertIsNone(cleared["validation_report"])
        self.assertEqual(cleared["recipe_data"], original_recipe)
        for field in ("recipe_data", "rules_snapshot"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.commands.save(source_id, **{field: None})
        self.assertEqual(self.queries.get(source_id)["recipe_data"], original_recipe)

    def test_atomic_decision_callback_replays_without_duplicate_decisions(self) -> None:
        source_id, _ = self.upload()
        report = {"reviews": [{"id": "review-1"}], "decisions": []}
        self.commands.save(
            source_id, validation_report=report,
            processing_status="review_required", validation_status="review_required",
        )

        def decide(snapshot):
            current = snapshot["validation_report"]
            existing = next((decision for decision in current["decisions"] if decision["review_id"] == "review-1"), None)
            if existing:
                return {}, {"decision": existing, "replayed": True}
            if snapshot["processing_status"] != "review_required":
                raise SourceConflictError("La revisión no está vigente.")
            decision = {"review_id": "review-1", "scope": "source", "action": "confirm"}
            current["decisions"].append(decision)
            return {"validation_report": current, "processing_status": "uploaded"}, {"decision": decision, "replayed": False}

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: self.commands.mutate(source_id, decide), range(8)))
        self.assertEqual(sum(not result["replayed"] for result in results), 1)
        self.assertEqual(len(self.queries.get(source_id)["validation_report"]["decisions"]), 1)
        self.assertEqual(report["decisions"], [])

    def test_failed_decision_callback_does_not_write_and_cannot_overwrite_original(self) -> None:
        source_id, _ = self.upload()

        def invalid(snapshot):
            snapshot["recipe_data"]["proposal"] = {"changed": True}
            raise SourceConflictError("La revisión no está vigente.")

        with self.assertRaises(SourceConflictError):
            self.commands.mutate(source_id, invalid)
        with self.assertRaises(ValueError):
            self.commands.mutate(source_id, lambda snapshot: ({"file_content": b"changed"}, {}))
        self.assertIsNone(self.queries.get(source_id)["recipe_data"]["proposal"])
        self.assertEqual(self.queries.get(source_id, include_content=True)["file_content"], b"station,pH\nA,7.2\n")

    def test_interruption_is_recorded_as_technical_failure_with_evidence_retained(self) -> None:
        source_id, _ = self.upload()
        self.commands.save(source_id, validation_report={"reviews": [{"id": "old-review"}], "decisions": []})
        with self.commands.worker_lock():
            self.commands.claim_next()
            self.assertEqual(self.commands.recover_interrupted(), 1)
            self.assertEqual(self.commands.recover_interrupted(), 0)
        snapshot = self.queries.get(source_id, include_content=True)
        self.assertEqual(snapshot["processing_status"], "failed")
        self.assertEqual(snapshot["validation_status"], "failed")
        self.assertEqual(snapshot["recipe_data"]["_workflow"]["stage"], "interrupted")
        self.assertEqual(snapshot["validation_report"]["reviews"], [{"id": "old-review"}])
        self.assertEqual(snapshot["validation_report"]["technical_errors"][0]["code"], "processing_interrupted")
        self.assertFalse(snapshot["is_active"])


class PostgreSQLWorkerLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = MagicMock()
        self.context.engine.dialect.name = "postgresql"
        self.connection = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.context.engine.connect.return_value.execution_options.return_value = self.connection
        self.commands = SqlAlchemySourceCommands(self.context)

    def test_lock_connection_uses_autocommit_and_finishes_before_processing(self) -> None:
        self.connection.scalar.return_value = True
        with self.commands.worker_lock() as acquired:
            self.assertTrue(acquired)
            self.context.engine.connect.return_value.execution_options.assert_called_once_with(
                isolation_level="AUTOCOMMIT",
            )
            self.connection.commit.assert_called_once_with()
            self.connection.execute.assert_not_called()
        self.assertIn("pg_try_advisory_lock", str(self.connection.scalar.call_args.args[0]))
        self.assertIn("pg_advisory_unlock", str(self.connection.execute.call_args.args[0]))
        self.assertEqual(self.connection.commit.call_count, 2)

    def test_unavailable_lock_is_not_unlocked_by_the_losing_worker(self) -> None:
        self.connection.scalar.return_value = False
        with self.commands.worker_lock() as acquired:
            self.assertFalse(acquired)
        self.connection.execute.assert_not_called()

    def test_lock_is_released_if_pipeline_raises(self) -> None:
        self.connection.scalar.return_value = True
        with self.assertRaises(RuntimeError):
            with self.commands.worker_lock():
                raise RuntimeError("pipeline error")
        self.connection.execute.assert_called_once()
        self.connection.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
