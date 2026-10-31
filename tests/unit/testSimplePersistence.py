"""Persistencia completa, recuperación y atomicidad sin servicios externos."""

import re
from importlib import import_module
from io import StringIO
from unittest import TestCase

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, Table, event, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from aquai.application.contracts.ingestion import PreparedTable
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.domain.models import (
    Conversation,
    Message,
    MessageSource,
    MessageStatus,
    Source,
)
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.persistence.mapping import metadata
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from tests.sourceDatabase import SourceTestDbContext


def _upload(content: bytes = b"codigo,valor\n001,nd\n") -> Source:
    return Source(
        name="Observaciones",
        original_filename="observaciones.csv",
        file_type="csv",
        mime_type="text/csv",
        file_size_bytes=len(content),
        file_content=content,
    )


def _prepared(count: int = 2) -> PreparedTable:
    return PreparedTable(
        name="Observaciones",
        source_locator={"file_type": "csv", "header_row": 1},
        columns=[
            {
                "original_name": "Código",
                "sql_name": "codigo",
                "sql_type": "TEXT",
                "meaning": None,
                "unit": None,
                "numeric_eligible": False,
            },
            {
                "original_name": "Valor",
                "sql_name": "valor",
                "sql_type": "TEXT",
                "meaning": None,
                "unit": None,
                "numeric_eligible": False,
            },
        ],
        rows=[
            {
                "_aquai_row": index,
                "_aquai_source_row": index + 1,
                "codigo": "001",
                "valor": "nd" if index % 2 else "NA",
            }
            for index in range(1, count + 1)
        ],
    )


def _ddl_lines(body: str) -> list[str]:
    return sorted(" ".join(line.strip().rstrip(",").split()) for line in body.strip().splitlines())


class SimplePersistenceTests(TestCase):
    def setUp(self):
        self.context = SourceTestDbContext()
        self.commands = SqlAlchemySourceCommands(self.context)
        self.queries = SqlAlchemySourceQueries(self.context)
        self.addCleanup(self.context.dispose)

    def _rows(self, dataset):
        with self.context.engine.connect() as connection:
            table = Table(dataset["physical_table_name"], MetaData(), autoload_with=connection)
            return connection.execute(select(table).order_by(table.c._aquai_row)).mappings().all()

    def test_all_rows_persist_once_and_sha_deduplicates_originals(self):
        source = _upload()
        self.assertEqual(self.commands.create_upload(source), (source.id, True))
        self.assertEqual(self.commands.create_upload(_upload()), (source.id, False))
        self.assertEqual(self.commands.claim_next(), source.id)
        prepared = _prepared(1002)
        self.commands.save_prepared(source.id, [prepared], {"warnings": []})
        self.commands.save_prepared(source.id, [prepared], {})

        detail = self.queries.get(source.id)
        self.assertEqual(detail["processing_status"], "ready")
        self.assertNotIn("file_content", detail)
        self.assertEqual(
            self.queries.get(source.id, include_content=True)["file_content"], source.file_content
        )
        self.assertEqual(len(detail["datasets"]), 1)
        dataset = detail["datasets"][0]
        rows = self._rows(dataset)
        self.assertEqual(dataset["row_count"], 1002)
        self.assertEqual(dataset["semantic_point_id"], dataset["id"])
        self.assertEqual(len(rows), 1002)
        self.assertEqual(rows[-1]["_aquai_source_row"], 1003)
        self.assertEqual(rows[0]["codigo"], "001")
        self.assertEqual(rows[0]["valor"], "nd")
        self.assertEqual(rows[1]["valor"], "NA")
        self.assertEqual(self.queries.list(search="Observ", processing_status="ready")["total"], 1)

    def test_interruption_and_provider_failure_preserve_rows_and_activation_gates_queries(self):
        source = _upload()
        self.commands.create_upload(source)
        with self.commands.worker_lock() as acquired:
            self.assertTrue(acquired)
            with self.commands.worker_lock() as second:
                self.assertFalse(second)
            self.commands.claim_next()
            self.commands.record_failure(source.id, "interrupted", "Cancelación al preparar.")
            self.assertEqual(self.queries.get(source.id)["processing_status"], "uploaded")
            self.commands.claim_next()
            self.assertEqual(self.commands.recover_interrupted(), 1)
            self.assertEqual(self.commands.claim_next(), source.id)
        self.commands.save_prepared(source.id, [_prepared()], {})
        dataset = self.queries.get(source.id)["datasets"][0]
        invented_units = [dict(column, unit="mg/L") for column in dataset["columns_metadata"]]
        with self.assertRaisesRegex(ValueError, "metadatos verificados"):
            self.commands.save_dataset_semantics(dataset["id"], "Unidad sugerida.", invented_units)
        self.commands.save_dataset_semantics(
            dataset["id"],
            "Datos cargados con descripción básica.",
            [],
            warnings=["Qwen no está disponible."],
        )
        self.assertEqual(self.commands.claim_index_next(), source.id)
        self.assertEqual(self.commands.recover_interrupted(), 1)
        self.commands.set_index_status(source.id, "failed", "Qdrant no está disponible.")
        detail = self.queries.get(source.id)
        self.assertEqual(detail["processing_status"], "ready_with_warnings")
        self.assertEqual(detail["semantic_index_status"], "failed")
        self.assertEqual(len(self._rows(dataset)), 2)
        self.assertEqual(self.queries.usable_dataset_ids([dataset["id"]]), {dataset["id"]})
        self.commands.request_reindex(source.id)
        self.commands.set_active(source.id, False)
        self.commands.set_index_status(source.id, "ready", expected_active=True)
        self.assertEqual(self.queries.get(source.id)["semantic_index_status"], "pending")
        self.assertEqual(self.queries.usable_dataset_ids([dataset["id"]]), set())
        self.commands.set_active(source.id, True)
        self.assertEqual(self.queries.usable_dataset_ids([dataset["id"]]), {dataset["id"]})

        empty = _upload(b"region sin estructura")
        self.commands.create_upload(empty)
        self.commands.save_prepared(empty.id, [], {"warnings": ["Región no tabular."]})
        self.commands.set_active(empty.id, True)
        self.assertEqual(self.queries.get(empty.id)["processing_status"], "failed")
        self.assertEqual(self.queries.get(empty.id)["datasets"], [])

    def test_reprocess_rejects_lost_regions_and_sources_used_by_messages(self):
        metadata.create_all(self.context.engine)
        source = _upload()
        self.commands.create_upload(source)
        self.commands.save_prepared(source.id, [_prepared()], {})
        dataset = self.queries.get(source.id)["datasets"][0]
        with self.assertRaisesRegex(ValueError, "regiones existentes"):
            self.commands.save_prepared(source.id, [], {}, replace_existing=True)
        with self.assertRaisesRegex(ValueError, "reduciría registros"):
            self.commands.save_prepared(source.id, [_prepared(1)], {}, replace_existing=True)
        with self.context.session() as session:
            conversation = Conversation(title="Consulta existente")
            session.add_all([conversation, MessageStatus(id=1, name="done")])
            session.flush()
            message = Message(
                conversation_id=conversation.id,
                sequence_number=1,
                role="user",
                content="Datos",
                status_id=1,
            )
            session.add(message)
            session.flush()
            session.add(
                MessageSource(
                    message_id=message.id,
                    source_id=source.id,
                    dataset_id=dataset["id"],
                    reference_order=1,
                    details={},
                )
            )
            session.commit()
        with self.assertRaisesRegex(ValueError, "referenciada"):
            self.commands.save_prepared(source.id, [_prepared()], {}, replace_existing=True)
        self.assertEqual(
            self.queries.get(source.id)["datasets"][0]["physical_table_name"],
            dataset["physical_table_name"],
        )
        self.assertEqual(len(self._rows(dataset)), 2)

    def test_failed_batch_rolls_back_catalog_and_physical_tables_and_schema_rejects_bad_state(self):
        source = _upload()
        self.commands.create_upload(source)
        with self.assertRaisesRegex(ValueError, "idéntica procedencia"):
            self.commands.save_prepared(source.id, [_prepared(), _prepared()], {})
        inserts = 0

        def reject_second_table(connection, cursor, statement, parameters, context, executemany):
            nonlocal inserts
            if statement.startswith('INSERT INTO "dataset_'):
                inserts += 1
                if inserts == 2:
                    raise RuntimeError("Fallo simulado del segundo lote.")

        event.listen(self.context.engine, "before_cursor_execute", reject_second_table)
        second_table = _prepared()
        second_table.source_locator["header_row"] = 5
        try:
            with self.assertRaisesRegex(RuntimeError, "Fallo simulado"):
                self.commands.save_prepared(source.id, [_prepared(), second_table], {})
        finally:
            event.remove(self.context.engine, "before_cursor_execute", reject_second_table)
        self.assertEqual(self.queries.get(source.id)["datasets"], [])
        self.assertFalse(
            any(
                name.startswith("dataset_")
                for name in inspect(self.context.engine).get_table_names()
            )
        )
        self.assertEqual(
            self.queries.get(source.id)["processing_status"], ProcessingStatus.UPLOADED
        )
        self.assertEqual(
            set(metadata.tables),
            {
                "Source",
                "Dataset",
                "Conversation",
                "Message",
                "MessageSource",
                "MessageStatus",
            },
        )
        with self.assertRaises(IntegrityError):
            with self.context.engine.begin() as connection:
                connection.execute(
                    metadata.tables["Source"].update().values(processing_status="invalid")
                )

        output = StringIO()
        migration_context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": output},
        )
        migration = import_module("aquai.infrastructure.migrations.versions.initialCreate")
        with Operations.context(migration_context):
            migration.upgrade()
        statements = dict(re.findall(r'CREATE TABLE "(\w+)" \((.*?)\);', output.getvalue(), re.S))
        self.assertEqual(set(statements), set(metadata.tables))
        for name, table in metadata.tables.items():
            with self.subTest(table=name):
                compiled = str(CreateTable(table).compile(dialect=postgresql.dialect()))
                body = compiled[compiled.index("(") + 1 : compiled.rindex(")")]
                self.assertEqual(_ddl_lines(statements[name]), _ddl_lines(body))
