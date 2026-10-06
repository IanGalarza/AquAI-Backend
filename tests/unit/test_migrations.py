"""Verifica Alembic y la migración inicial generando SQL en memoria."""

import ast
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from alembic import util
from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.environment import EnvironmentContext
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from mako.template import Template
from sqlalchemy import ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.configurations.status_seeds import (
    message_status_seed_rows,
    processing_status_seed_rows,
)
from aquai.infrastructure.persistence.mapping import configure_mappings, metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_mappings()
        cls.config = Config(str(PROJECT_ROOT / "alembic.ini"))
        cls.scripts = ScriptDirectory.from_config(cls.config)
        cls.revision = cls.scripts.get_revision("fefbd016121a")
        cls.created_tables = {}
        cls.created_indexes = {}
        cls.seed_rows = {}
        buffer = StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": buffer},
        )
        original_create = Operations.create_table
        original_insert = Operations.bulk_insert
        original_index = Operations.create_index

        def capture_table(operations, table_name, *columns, **options):
            table = original_create(operations, table_name, *columns, **options)
            cls.created_tables[table_name] = table
            return table

        def capture_seeds(operations, table, rows, **options):
            cls.seed_rows[table.name] = rows
            original_insert(operations, table, rows, **options)

        def capture_index(operations, name, table_name, columns, **options):
            cls.created_indexes.setdefault(table_name, set()).add(
                (name, tuple(columns), options.get("unique", False))
            )
            original_index(operations, name, table_name, columns, **options)

        with Operations.context(context), patch.object(
            Operations, "create_table", capture_table
        ), patch.object(Operations, "bulk_insert", capture_seeds), patch.object(
            Operations, "create_index", capture_index
        ):
            cls.revision.module.upgrade()
        cls.upgrade_sql = buffer.getvalue()

        buffer = StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": buffer},
        )
        with Operations.context(context):
            cls.revision.module.downgrade()
        cls.downgrade_sql = buffer.getvalue()

    def test_initial_revision_creates_only_the_current_model_tables(self) -> None:
        self.assertIsNone(self.revision.down_revision)
        self.assertEqual(self.revision.doc, "InitialCreate")
        self.assertEqual(set(self.created_tables), set(metadata.tables))
        self.assertEqual(len(self.created_tables), 7)
        self.assertNotIn("SERIAL", self.upgrade_sql)

    def test_generated_columns_preserve_names_types_nullability_and_keys(self) -> None:
        dialect = postgresql.dialect()
        for name, expected in metadata.tables.items():
            generated = self.created_tables[name]
            with self.subTest(table=name):
                self.assertEqual(
                    [column.name for column in generated.columns],
                    [column.name for column in expected.columns],
                )
                self.assertEqual(
                    [column.name for column in generated.primary_key],
                    [column.name for column in expected.primary_key],
                )
            for actual_column, expected_column in zip(generated.columns, expected.columns):
                with self.subTest(table=name, column=expected_column.name):
                    self.assertEqual(actual_column.nullable, expected_column.nullable)
                    self.assertEqual(
                        actual_column.type.compile(dialect=dialect),
                        expected_column.type.compile(dialect=dialect),
                    )

    def test_generated_constraints_and_indexes_match_the_metadata(self) -> None:
        def foreign_keys(table):
            return {
                (tuple((element.parent.name, element.column.table.name, element.column.name)
                       for element in constraint.elements), constraint.ondelete)
                for constraint in table.constraints
                if isinstance(constraint, ForeignKeyConstraint)
            }

        def unique_keys(table):
            return {
                tuple(column.name for column in constraint.columns)
                for constraint in table.constraints
                if isinstance(constraint, UniqueConstraint)
            }

        def indexes(table):
            return {(index.name, tuple(column.name for column in index.columns), index.unique)
                    for index in table.indexes}

        for name, expected in metadata.tables.items():
            generated = self.created_tables[name]
            with self.subTest(table=name):
                self.assertEqual(foreign_keys(generated), foreign_keys(expected))
                self.assertEqual(unique_keys(generated), unique_keys(expected))
                self.assertEqual(self.created_indexes.get(name, set()), indexes(expected))

    def test_seed_values_are_frozen_in_the_revision_and_match_the_catalogs(self) -> None:
        self.assertEqual(self.seed_rows, {
            "ProcessingStatus": processing_status_seed_rows(),
            "MessageStatus": message_status_seed_rows(),
        })
        self.assertEqual(self.upgrade_sql.count("INSERT INTO"), 11)
        source = Path(self.revision.path).read_text(encoding="utf-8")
        self.assertNotIn("from aquai", source)
        self.assertNotIn("create_all", source)

    def test_downgrade_removes_tables_in_reverse_dependency_order(self) -> None:
        positions = [self.downgrade_sql.index(f'DROP TABLE "{name}"')
                     for name in reversed(list(self.created_tables))]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.downgrade_sql.count("DROP TABLE"), 7)
        self.assertEqual(self.upgrade_sql.count("CREATE INDEX"), 6)
        self.assertEqual(self.downgrade_sql.count("DROP INDEX"), 6)

    def test_offline_environment_uses_existing_settings_without_connecting(self) -> None:
        config = Config(str(PROJECT_ROOT / "alembic.ini"), output_buffer=StringIO())
        settings = DatabaseSettings(
            _env_file=None,
            database_url="postgresql+psycopg://user:clave%25%40@localhost/aquai",
        )
        captured = {}

        def inspect_context(revision, context):
            captured["metadata"] = context.opts["target_metadata"]
            captured["dialect"] = context.dialect.name
            return ()

        with patch(
            "aquai.infrastructure.config.database.DatabaseSettings", return_value=settings
        ) as settings_factory, patch("sqlalchemy.create_engine") as create_engine, EnvironmentContext(
            config, self.scripts, fn=inspect_context, as_sql=True
        ):
            self.scripts.run_env()
        settings_factory.assert_called_once_with()
        create_engine.assert_not_called()
        self.assertIs(captured["metadata"], metadata)
        self.assertEqual(captured["dialect"], "postgresql")
        self.assertNotIn("sqlalchemy.url", config.get_section("alembic"))

    def test_new_revision_template_has_spanish_comments_and_valid_python(self) -> None:
        template = Template(filename=str(Path(self.scripts.dir) / "script.py.mako"))
        operations = (
            "# ### commands auto generated by Alembic - please adjust! ###\n"
            "pass\n# ### end Alembic commands ###"
        )
        result = template.render(
            message="Ejemplo", up_revision="example", down_revision=self.revision.revision,
            branch_labels=None, depends_on=None, create_date="2026-10-06",
            imports="", upgrades=operations, downgrades=operations, comma=util.format_as_comma,
        )
        ast.parse(result)
        self.assertIn("revisalas antes de aplicar", result)
        self.assertNotIn("commands auto generated", result)


if __name__ == "__main__":
    unittest.main()
