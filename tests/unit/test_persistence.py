"""Pruebas del contrato de persistencia sin conectarse a una base de datos."""

import unittest
import warnings
from uuid import uuid4

from sqlalchemy import ForeignKeyConstraint, UniqueConstraint, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.exc import SAWarning
from sqlalchemy.orm import undefer
from sqlalchemy.orm.attributes import NO_VALUE
from sqlalchemy.schema import CreateTable

from aquai.domain.models import Dataset, Message, MessageSource, Source
from aquai.infrastructure.persistence.mapping import configure_mappings, mapper_registry, metadata


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SAWarning)
            configure_mappings()

    def test_mapping_registration_is_idempotent_and_schema_compiles_offline(self) -> None:
        configure_mappings()
        self.assertEqual(len(list(mapper_registry.mappers)), 7)
        self.assertEqual(
            set(metadata.tables),
            {"Source", "ProcessingStatus", "Dataset", "Conversation", "Message", "MessageStatus", "MessageSource"},
        )
        ddl = "\n".join(
            str(CreateTable(table).compile(dialect=postgresql.dialect()))
            for table in metadata.sorted_tables
        )
        for expected in (
            '"originalFilename" VARCHAR NOT NULL',
            '"fileContent" BYTEA NOT NULL',
            '"fileSizeBytes" BIGINT NOT NULL',
            '"sourceLocator" JSONB NOT NULL',
            '"CreatedAt" TIMESTAMP WITH TIME ZONE NOT NULL',
            '"UpdatedAt" TIMESTAMP WITH TIME ZONE',
            '"ReferenceOrder" INTEGER NOT NULL',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, ddl)
        self.assertNotIn("SERIAL", ddl)

    def test_message_sequence_is_unique_only_within_its_conversation(self) -> None:
        unique_keys = {
            tuple(column.name for column in constraint.columns)
            for constraint in metadata.tables["Message"].constraints
            if isinstance(constraint, UniqueConstraint)
        }
        self.assertEqual(unique_keys, {("conversationId", "sequenceNumber")})

    def test_optional_dataset_reference_enforces_same_source(self) -> None:
        table = metadata.tables["MessageSource"]
        composite_keys = [
            constraint
            for constraint in table.constraints
            if isinstance(constraint, ForeignKeyConstraint) and len(constraint.elements) == 2
        ]
        self.assertEqual(len(composite_keys), 1)
        constraint = composite_keys[0]
        self.assertEqual(
            [(element.parent.name, element.column.table.name, element.column.name) for element in constraint.elements],
            [("datasetId", "Dataset", "id"), ("sourceId", "Dataset", "sourceId")],
        )
        self.assertTrue(table.c.dataset_id.nullable)
        self.assertFalse(table.c.source_id.nullable)
        dataset_keys = {
            tuple(column.name for column in key.columns)
            for key in metadata.tables["Dataset"].constraints
            if isinstance(key, UniqueConstraint)
        }
        self.assertEqual(dataset_keys, {("id", "sourceId")})
        self.assertFalse(metadata.tables["Dataset"].c.physical_table_name.unique)

    def test_dataset_navigation_does_not_overwrite_source_id(self) -> None:
        for model, navigation in ((MessageSource, "dataset"), (Dataset, "message_sources")):
            pairs = inspect(model).relationships[navigation].synchronize_pairs
            with self.subTest(model=model.__name__):
                self.assertEqual(
                    [(left.name, right.name) for left, right in pairs],
                    [("id", "datasetId")],
                )
        source_pairs = inspect(MessageSource).relationships.source.synchronize_pairs
        self.assertEqual([(left.name, right.name) for left, right in source_pairs], [("id", "sourceId")])

    def test_source_listing_defers_file_content_and_can_load_it_explicitly(self) -> None:
        dialect = postgresql.dialect()
        listing_sql = str(select(Source).compile(dialect=dialect))
        self.assertNotIn('"fileContent"', listing_sql)
        self.assertIn('"originalFilename"', listing_sql)

        file_sql = str(
            select(Source).options(undefer(inspect(Source).attrs.file_content))
            .compile(dialect=dialect)
        )
        self.assertIn('"fileContent"', file_sql)

    def test_message_can_start_with_an_independent_empty_json_document(self) -> None:
        values = {
            "conversation_id": uuid4(), "sequence_number": 1,
            "role": "user", "content": "consulta", "status_id": 1,
        }
        first = Message(**values)
        second = Message(**values)
        self.assertEqual(first.content_data, {})
        self.assertEqual(second.content_data, {})
        self.assertIsNot(first.content_data, second.content_data)
        first.content_data["result"] = "respuesta"
        self.assertEqual(second.content_data, {})

        custom = Message(**values, content_data={"table": []})
        self.assertEqual(custom.content_data, {"table": []})

    def test_fk_only_constructors_leave_scalar_navigations_unassigned(self) -> None:
        source_id, conversation_id, dataset_id, message_id = (uuid4() for _ in range(4))
        source = Source(
            name="water", original_filename="water.csv", file_type="CSV", mime_type="text/csv",
            file_size_bytes=3, is_active=True, processing_status_id=1, file_content=b"a,b",
        )
        dataset = Dataset(source_id=source_id, name="measurements", source_locator={}, version=1, is_active=True)
        message = Message(
            conversation_id=conversation_id, sequence_number=1, role="assistant", content="",
            content_data={}, status_id=1,
        )
        citation = MessageSource(
            message_id=message_id, source_id=source_id, dataset_id=dataset_id, reference_order=1, details={},
        )
        for model, navigations in (
            (source, ("processing_status",)), (dataset, ("source",)),
            (message, ("conversation", "status")), (citation, ("message", "source", "dataset")),
        ):
            for navigation in navigations:
                with self.subTest(model=type(model).__name__, navigation=navigation):
                    self.assertIs(inspect(model).attrs[navigation].loaded_value, NO_VALUE)
        self.assertEqual(citation.source_id, source_id)
        self.assertEqual(citation.dataset_id, dataset_id)
        self.assertEqual(source.created_at.utcoffset().total_seconds(), 0)

    def test_required_json_maps_none_to_sql_null_and_preserves_nested_null(self) -> None:
        dialect = PGDialect_psycopg()
        for table_name, column_key in (
            ("Dataset", "source_locator"), ("Message", "content_data"), ("MessageSource", "details"),
        ):
            column = metadata.tables[table_name].c[column_key]
            with self.subTest(table=table_name):
                self.assertFalse(column.nullable)
                bind = column.type.dialect_impl(dialect).bind_processor(dialect)
                self.assertIsNone(bind(None))
                self.assertEqual(bind({"optional": None}).obj, {"optional": None})

    def test_deletion_preserves_history_instead_of_cascading(self) -> None:
        for table in metadata.tables.values():
            for constraint in table.foreign_key_constraints:
                with self.subTest(table=table.name):
                    self.assertEqual(constraint.ondelete, "RESTRICT")
        for mapper in mapper_registry.mappers:
            for navigation in mapper.relationships:
                with self.subTest(model=mapper.class_.__name__, navigation=navigation.key):
                    self.assertNotIn("delete", navigation.cascade)
                    if navigation.uselist:
                        self.assertEqual(navigation.passive_deletes, "all")


if __name__ == "__main__":
    unittest.main()
