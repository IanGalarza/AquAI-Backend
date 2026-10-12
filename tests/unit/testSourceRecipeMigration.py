"""Verifica la revisión de recetas generando SQL sólo en memoria, sin conexiones."""

import ast
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint
from sqlalchemy.dialects import postgresql

from aquai.infrastructure.persistence.configurations.statusSeeds import validation_status_seed_rows
from aquai.infrastructure.persistence.mapping import configure_mappings, metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INITIAL_REVISION_ID = "fefbd016121a"
SOURCE_RECIPE_REVISION_ID = "5ef1c8d4e697"
SOURCE_RECIPE_REVISION_TABLES = {"SourceRecipe", "ValidationStatus"}
EXPECTED_SEEDS = [
    {"id": 1, "name": "pending"},
    {"id": 2, "name": "ready"},
    {"id": 3, "name": "ready_with_warnings"},
    {"id": 4, "name": "review_required"},
    {"id": 5, "name": "failed"},
]


class SourceRecipeMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        configure_mappings()
        cls.scripts = ScriptDirectory.from_config(Config(str(PROJECT_ROOT / "alembic.ini")))
        cls.revision = cls.scripts.get_revision(SOURCE_RECIPE_REVISION_ID)
        cls.created_tables = {}
        cls.created_indexes = {}
        cls.seed_rows = {}
        buffer = StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer},
        )
        original_create = Operations.create_table
        original_index = Operations.create_index
        original_insert = Operations.bulk_insert

        def capture_table(operations, table_name, *columns, **options):
            table = original_create(operations, table_name, *columns, **options)
            cls.created_tables[table_name] = table
            return table

        def capture_index(operations, name, table_name, columns, **options):
            cls.created_indexes.setdefault(table_name, set()).add(
                (name, tuple(columns), options.get("unique", False))
            )
            return original_index(operations, name, table_name, columns, **options)

        def capture_seeds(operations, table, rows, **options):
            cls.seed_rows[table.name] = rows
            return original_insert(operations, table, rows, **options)

        with Operations.context(context), patch.object(
            Operations, "create_table", capture_table
        ), patch.object(Operations, "create_index", capture_index), patch.object(
            Operations, "bulk_insert", capture_seeds
        ), patch("sqlalchemy.create_engine") as create_engine:
            cls.revision.module.upgrade()
        create_engine.assert_not_called()
        cls.upgrade_sql = buffer.getvalue()

        buffer = StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer},
        )
        with Operations.context(context), patch("sqlalchemy.create_engine") as create_engine:
            cls.revision.module.downgrade()
        create_engine.assert_not_called()
        cls.downgrade_sql = buffer.getvalue()

    def test_revision_chain_extends_the_initial_revision_without_branches(self) -> None:
        self.assertEqual(self.revision.revision, SOURCE_RECIPE_REVISION_ID)
        self.assertEqual(self.revision.down_revision, INITIAL_REVISION_ID)
        self.assertIsNone(self.revision.module.branch_labels)
        self.assertIsNone(self.revision.module.depends_on)
        self.assertEqual(self.scripts.get_heads(), [self.revision.revision])
        self.assertEqual(
            [revision.revision for revision in self.scripts.walk_revisions()],
            [self.revision.revision, INITIAL_REVISION_ID],
        )

    def test_upgrade_creates_only_the_two_new_tables_in_dependency_order(self) -> None:
        self.assertEqual(set(self.created_tables), SOURCE_RECIPE_REVISION_TABLES)
        self.assertEqual(list(self.created_tables), ["ValidationStatus", "SourceRecipe"])
        self.assertEqual(self.upgrade_sql.count("CREATE TABLE"), 2)
        self.assertEqual(self.upgrade_sql.count("CREATE INDEX"), 1)
        self.assertNotIn("ALTER TABLE", self.upgrade_sql)
        self.assertNotIn("DROP TABLE", self.upgrade_sql)
        self.assertNotIn("SERIAL", self.upgrade_sql)
        self.assertNotIn("IDENTITY", self.upgrade_sql)

    def test_columns_preserve_sql_names_types_nullability_and_primary_keys(self) -> None:
        dialect = postgresql.dialect()
        for table_name in sorted(SOURCE_RECIPE_REVISION_TABLES):
            expected = metadata.tables[table_name]
            generated = self.created_tables[table_name]
            with self.subTest(table=table_name):
                self.assertEqual(generated.schema, expected.schema)
                self.assertEqual(
                    [column.name for column in generated.columns],
                    [column.name for column in expected.columns],
                )
                self.assertEqual(
                    [column.name for column in generated.primary_key],
                    [column.name for column in expected.primary_key],
                )
                self.assertEqual(generated.primary_key.name, expected.primary_key.name)
            for actual, wanted in zip(generated.columns, expected.columns):
                with self.subTest(table=table_name, column=wanted.name):
                    self.assertEqual(actual.nullable, wanted.nullable)
                    self.assertEqual(
                        actual.type.compile(dialect=dialect), wanted.type.compile(dialect=dialect)
                    )
                    self.assertIsNone(actual.default)
                    self.assertIsNone(actual.server_default)
                    self.assertIsNone(actual.onupdate)
                    self.assertIsNone(actual.server_onupdate)
                    self.assertIsNone(actual.identity)
                    if isinstance(wanted.type, postgresql.JSONB):
                        self.assertTrue(actual.type.none_as_null)
        self.assertFalse(self.created_tables["ValidationStatus"].c.id.autoincrement)

    def test_constraints_and_indexes_match_the_current_metadata(self) -> None:
        def foreign_keys(table):
            return {
                (constraint.name,
                 tuple((element.parent.name, element.target_fullname)
                       for element in constraint.elements),
                 constraint.ondelete, constraint.onupdate)
                for constraint in table.constraints if isinstance(constraint, ForeignKeyConstraint)
            }

        def unique_keys(table):
            return {
                (constraint.name, tuple(column.name for column in constraint.columns))
                for constraint in table.constraints if isinstance(constraint, UniqueConstraint)
            }

        def checks(table):
            return {
                (constraint.name, " ".join(str(constraint.sqltext).split()))
                for constraint in table.constraints if isinstance(constraint, CheckConstraint)
            }

        for table_name in sorted(SOURCE_RECIPE_REVISION_TABLES):
            expected = metadata.tables[table_name]
            generated = self.created_tables[table_name]
            with self.subTest(table=table_name):
                self.assertEqual(foreign_keys(generated), foreign_keys(expected))
                self.assertEqual(unique_keys(generated), unique_keys(expected))
                self.assertEqual(checks(generated), checks(expected))
                self.assertEqual(
                    self.created_indexes.get(table_name, set()),
                    {(index.name, tuple(column.name for column in index.columns), index.unique)
                     for index in expected.indexes},
                )
        self.assertEqual(len(checks(self.created_tables["SourceRecipe"])), 3)

    def test_the_five_validation_seeds_are_frozen_in_the_revision(self) -> None:
        self.assertEqual(self.seed_rows, {"ValidationStatus": EXPECTED_SEEDS})
        self.assertEqual(validation_status_seed_rows(), EXPECTED_SEEDS)
        self.assertEqual(self.upgrade_sql.count('INSERT INTO "ValidationStatus"'), 5)
        self.assertEqual(self.upgrade_sql.count("INSERT INTO"), 5)
        self.assertNotIn("UPDATE ", self.upgrade_sql)
        self.assertNotIn("DELETE FROM", self.upgrade_sql)
        source = Path(self.revision.path).read_text(encoding="utf-8")
        imports = [node for node in ast.walk(ast.parse(source))
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        for node in imports:
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [
                alias.name for alias in node.names
            ]
            self.assertFalse(any(module == "aquai" or module.startswith("aquai.")
                                 for module in modules))
        self.assertNotIn("create_all", source)

    def test_downgrade_drops_only_the_new_index_and_tables_in_dependency_order(self) -> None:
        expected = [
            'DROP INDEX "ix_SourceRecipe_validationStatusId"',
            'DROP TABLE "SourceRecipe"',
            'DROP TABLE "ValidationStatus"',
        ]
        positions = [self.downgrade_sql.index(statement) for statement in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.downgrade_sql.count("DROP INDEX"), 1)
        self.assertEqual(self.downgrade_sql.count("DROP TABLE"), 2)
        self.assertNotIn("ALTER TABLE", self.downgrade_sql)
        self.assertNotIn("CREATE TABLE", self.downgrade_sql)


if __name__ == "__main__":
    unittest.main()
