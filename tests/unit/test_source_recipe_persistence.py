"""Contrato de SourceRecipe y ValidationStatus, verificado sin acceder a una BD."""

from copy import deepcopy
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime, timezone
import unittest
import warnings
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, JSON, UniqueConstraint, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.exc import SAWarning
from sqlalchemy.orm.attributes import NO_VALUE
from sqlalchemy.schema import CreateTable

from aquai.domain.models import Source, SourceRecipe, ValidationStatus
from aquai.infrastructure.persistence.configurations.status_seeds import (
    VALIDATION_STATUS_SEEDS,
    validation_status_seed_rows,
)
from aquai.infrastructure.persistence.mapping import (
    configure_mappings,
    mapper_registry,
    metadata,
)


class SourceRecipePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SAWarning)
            configure_mappings()

    def make_recipe(self, **overrides) -> SourceRecipe:
        values = {
            "source_id": uuid4(),
            "recipe_data": {"sheets": [{"name": "measurements", "blocks": []}]},
            "rules_snapshot": {"rules": [], "provenance": "project"},
        }
        values.update(overrides)
        return SourceRecipe(**values)

    def make_source(self) -> Source:
        return Source(
            name="water", original_filename="water.csv", file_type="CSV",
            mime_type="text/csv", file_size_bytes=3, is_active=True,
            processing_status_id=1, file_content=b"a,b",
        )

    def test_domain_attributes_and_required_constructor_values(self) -> None:
        self.assertTrue(is_dataclass(SourceRecipe))
        self.assertTrue(is_dataclass(ValidationStatus))
        recipe_fields = {attribute.name: attribute for attribute in fields(SourceRecipe)}
        self.assertEqual(set(recipe_fields), {
            "id", "source_id", "recipe_data", "rules_snapshot", "validation_status_id",
            "validation_report", "created_at", "updated_at", "source", "validation_status",
        })
        status_fields = {attribute.name: attribute for attribute in fields(ValidationStatus)}
        self.assertEqual(set(status_fields), {"id", "name", "source_recipes"})
        for attribute in ("source_id", "recipe_data", "rules_snapshot"):
            with self.subTest(attribute=attribute):
                self.assertIs(recipe_fields[attribute].default, MISSING)
                self.assertIs(recipe_fields[attribute].default_factory, MISSING)
                self.assertTrue(recipe_fields[attribute].init)
        for attribute in ("id", "name"):
            with self.subTest(status_attribute=attribute):
                self.assertIs(status_fields[attribute].default, MISSING)
                self.assertIs(status_fields[attribute].default_factory, MISSING)
        for attribute in ("source", "validation_status"):
            self.assertFalse(recipe_fields[attribute].init)
        self.assertFalse(status_fields["source_recipes"].init)

    def test_application_defaults_generate_uuid4_and_utc_creation_time(self) -> None:
        before = datetime.now(timezone.utc)
        first = self.make_recipe()
        second = self.make_recipe()
        after = datetime.now(timezone.utc)
        for recipe in (first, second):
            with self.subTest(recipe_id=recipe.id):
                self.assertIsInstance(recipe.id, UUID)
                self.assertEqual(recipe.id.version, 4)
                self.assertEqual(recipe.created_at.utcoffset().total_seconds(), 0)
                self.assertLessEqual(before, recipe.created_at)
                self.assertLessEqual(recipe.created_at, after)
                self.assertIsNone(recipe.updated_at)
                self.assertIsNone(recipe.validation_report)
                self.assertEqual(recipe.validation_status_id, 1)
        self.assertNotEqual(first.id, second.id)
        self.assertIsNot(first.recipe_data, second.recipe_data)
        self.assertIsNot(first.rules_snapshot, second.rules_snapshot)

    def test_explicit_ids_timestamps_and_system_status_are_preserved(self) -> None:
        recipe_id, source_id = uuid4(), uuid4()
        created_at = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
        updated_at = datetime(2026, 2, 3, 4, 5, tzinfo=timezone.utc)
        recipe = self.make_recipe(
            id=recipe_id, source_id=source_id, created_at=created_at, updated_at=updated_at,
            recipe_data={"validationStatus": "failed"}, validation_status_id=2,
            validation_report={"checks": [{"result": "ready"}]},
        )
        self.assertEqual(recipe.id, recipe_id)
        self.assertEqual(recipe.source_id, source_id)
        self.assertEqual(recipe.created_at, created_at)
        self.assertEqual(recipe.updated_at, updated_at)
        self.assertEqual(recipe.validation_status_id, 2)
        self.assertEqual(recipe.recipe_data, {"validationStatus": "failed"})
        self.assertEqual(recipe.validation_report, {"checks": [{"result": "ready"}]})

    def test_json_roots_are_objects_and_report_also_accepts_none(self) -> None:
        for attribute in ("recipe_data", "rules_snapshot", "validation_report"):
            for invalid in ([], "text", 1, False):
                with self.subTest(attribute=attribute, value=invalid):
                    with self.assertRaises(ValueError):
                        self.make_recipe(**{attribute: invalid})
        for attribute in ("recipe_data", "rules_snapshot"):
            with self.subTest(attribute=attribute, value=None):
                with self.assertRaises(ValueError):
                    self.make_recipe(**{attribute: None})
        document = {"optional": None, "blocks": [{"header": None}]}
        recipe = self.make_recipe(
            recipe_data=deepcopy(document), rules_snapshot=deepcopy(document),
            validation_report=deepcopy(document),
        )
        self.assertEqual(recipe.recipe_data, document)
        self.assertEqual(recipe.rules_snapshot, document)
        self.assertEqual(recipe.validation_report, document)
        self.assertIsNone(self.make_recipe(validation_report=None).validation_report)

    def test_document_reassignment_and_updated_at_remain_explicit(self) -> None:
        recipe = self.make_recipe()
        document = {"sheets": [{"name": "revised"}]}
        recipe.recipe_data = document
        self.assertIs(recipe.recipe_data, document)
        self.assertIsNone(recipe.updated_at)
        table = metadata.tables["SourceRecipe"]
        self.assertIsNone(table.c.updated_at.onupdate)
        self.assertIsNone(table.c.updated_at.server_onupdate)

    def test_sql_columns_match_the_contract_in_the_existing_schema(self) -> None:
        dialect = postgresql.dialect()
        recipe_table = metadata.tables["SourceRecipe"]
        status_table = metadata.tables["ValidationStatus"]
        expected = {
            "id": ("id", "UUID", False),
            "source_id": ("sourceId", "UUID", False),
            "recipe_data": ("recipeData", "JSONB", False),
            "rules_snapshot": ("rulesSnapshot", "JSONB", False),
            "validation_status_id": ("validationStatusId", "INTEGER", False),
            "validation_report": ("validationReport", "JSONB", True),
            "created_at": ("CreatedAt", "TIMESTAMP WITH TIME ZONE", False),
            "updated_at": ("UpdatedAt", "TIMESTAMP WITH TIME ZONE", True),
        }
        self.assertEqual(set(recipe_table.c.keys()), set(expected))
        for key, (name, sql_type, nullable) in expected.items():
            column = recipe_table.c[key]
            with self.subTest(column=name):
                self.assertEqual(column.name, name)
                self.assertEqual(column.type.compile(dialect=dialect), sql_type)
                self.assertEqual(column.nullable, nullable)
        self.assertEqual(
            [(column.name, column.type.compile(dialect=dialect), column.nullable)
             for column in status_table.c],
            [("id", "INTEGER", False), ("name", "VARCHAR", False)],
        )
        self.assertEqual(recipe_table.schema, metadata.tables["Source"].schema)
        self.assertEqual(status_table.schema, metadata.tables["Source"].schema)
        self.assertEqual([column.name for column in recipe_table.primary_key], ["id"])
        self.assertEqual([column.name for column in status_table.primary_key], ["id"])
        self.assertFalse(status_table.c.id.autoincrement)
        self.assertIsNone(status_table.c.id.identity)
        self.assertIsNone(status_table.c.id.default)
        self.assertIsNone(status_table.c.id.server_default)
        ddl = "\n".join(
            str(CreateTable(table).compile(dialect=dialect))
            for table in (recipe_table, status_table)
        )
        self.assertNotIn("SERIAL", ddl)
        self.assertNotIn("IDENTITY", ddl)

    def test_foreign_keys_uniqueness_and_status_index(self) -> None:
        recipe_table = metadata.tables["SourceRecipe"]
        status_table = metadata.tables["ValidationStatus"]
        self.assertEqual(
            {(element.parent.name, element.column.table.name, element.column.name,
              constraint.ondelete)
             for constraint in recipe_table.foreign_key_constraints
             for element in constraint.elements},
            {("sourceId", "Source", "id", "RESTRICT"),
             ("validationStatusId", "ValidationStatus", "id", "RESTRICT")},
        )
        for table, expected in ((recipe_table, {("sourceId",)}), (status_table, {("name",)})):
            with self.subTest(table=table.name):
                self.assertEqual(
                    {tuple(column.name for column in constraint.columns)
                     for constraint in table.constraints
                     if isinstance(constraint, UniqueConstraint)},
                    expected,
                )
        self.assertEqual(
            {(tuple(column.name for column in index.columns), index.unique)
             for index in recipe_table.indexes},
            {(("validationStatusId",), False)},
        )

    def test_jsonb_binding_distinguishes_sql_null_and_json_null(self) -> None:
        dialect = PGDialect_psycopg()
        table = metadata.tables["SourceRecipe"]
        for key in ("recipe_data", "rules_snapshot", "validation_report"):
            column = table.c[key]
            with self.subTest(column=column.name):
                self.assertTrue(column.type.none_as_null)
                bind = column.type.dialect_impl(dialect).bind_processor(dialect)
                self.assertIsNone(bind(None))
                self.assertIsNone(bind(JSON.NULL).obj)
                self.assertEqual(bind({"optional": None}).obj, {"optional": None})
        self.assertFalse(table.c.recipe_data.nullable)
        self.assertFalse(table.c.rules_snapshot.nullable)
        self.assertTrue(table.c.validation_report.nullable)

    def test_sql_checks_enforce_object_roots_for_all_json_documents(self) -> None:
        table = metadata.tables["SourceRecipe"]
        checks = [str(constraint.sqltext.compile(dialect=postgresql.dialect()))
                  for constraint in table.constraints if isinstance(constraint, CheckConstraint)]
        for column_name in ("recipeData", "rulesSnapshot", "validationReport"):
            with self.subTest(column=column_name):
                matching = [check for check in checks if column_name in check]
                self.assertTrue(matching)
                self.assertTrue(any("jsonb_typeof" in check and "'object'" in check
                                    for check in matching))

    def test_navigation_cardinality_and_delete_behavior(self) -> None:
        relationships = (
            (Source, "source_recipe", SourceRecipe, False, "source"),
            (SourceRecipe, "source", Source, False, "source_recipe"),
            (SourceRecipe, "validation_status", ValidationStatus, False, "source_recipes"),
            (ValidationStatus, "source_recipes", SourceRecipe, True, "validation_status"),
        )
        for model, navigation, target, uselist, back_populates in relationships:
            relation = inspect(model).relationships[navigation]
            with self.subTest(model=model.__name__, navigation=navigation):
                self.assertIs(relation.mapper.class_, target)
                self.assertEqual(relation.uselist, uselist)
                self.assertEqual(relation.back_populates, back_populates)
                self.assertNotIn("delete", relation.cascade)
                self.assertNotIn("delete-orphan", relation.cascade)
        self.assertEqual(inspect(Source).relationships.source_recipe.passive_deletes, "all")
        self.assertEqual(
            inspect(ValidationStatus).relationships.source_recipes.passive_deletes, "all"
        )
        source = self.make_source()
        self.assertIsNone(source.source_recipe)
        recipe = self.make_recipe(source_id=source.id)
        source.source_recipe = recipe
        self.assertIs(recipe.source, source)
        self.assertIs(source.source_recipe, recipe)
        status = ValidationStatus(id=1, name="pending")
        recipe.validation_status = status
        self.assertEqual(status.source_recipes, [recipe])

    def test_fk_only_constructor_leaves_navigations_unassigned(self) -> None:
        recipe = self.make_recipe()
        for navigation in ("source", "validation_status"):
            with self.subTest(navigation=navigation):
                self.assertIs(inspect(recipe).attrs[navigation].loaded_value, NO_VALUE)

    def test_identical_documents_are_allowed_for_independent_sources(self) -> None:
        recipe_data = {"sheets": [{"name": "measurements", "blocks": []}]}
        rules_snapshot = {"rules": ["normalize_headers"]}
        first = self.make_recipe(
            recipe_data=deepcopy(recipe_data), rules_snapshot=deepcopy(rules_snapshot),
        )
        second = self.make_recipe(
            recipe_data=deepcopy(recipe_data), rules_snapshot=deepcopy(rules_snapshot),
        )
        self.assertNotEqual(first.source_id, second.source_id)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.recipe_data, second.recipe_data)
        self.assertEqual(first.rules_snapshot, second.rules_snapshot)
        first.recipe_data["sheets"][0]["name"] = "changed"
        self.assertEqual(second.recipe_data, recipe_data)
        table = metadata.tables["SourceRecipe"]
        for constraint in table.constraints:
            if isinstance(constraint, UniqueConstraint):
                self.assertEqual([column.name for column in constraint.columns], ["sourceId"])
        self.assertFalse(any(index.unique for index in table.indexes))

    def test_metadata_registration_is_idempotent_and_keeps_source_columns(self) -> None:
        tables_before = dict(metadata.tables)
        mappers_before = {mapper.class_: mapper for mapper in mapper_registry.mappers}
        configure_mappings()
        configure_mappings()
        self.assertEqual(set(metadata.tables), {
            "Source", "ProcessingStatus", "Dataset", "Conversation", "Message",
            "MessageStatus", "MessageSource", "SourceRecipe", "ValidationStatus",
        })
        self.assertEqual(len(list(mapper_registry.mappers)), 9)
        for name, table in tables_before.items():
            self.assertIs(metadata.tables[name], table)
        for model, mapper in mappers_before.items():
            self.assertIs(inspect(model), mapper)
        self.assertEqual([column.name for column in metadata.tables["Source"].c], [
            "id", "name", "originalFilename", "fileType", "mimeType", "fileSizeBytes",
            "isActive", "description", "processingStatusId", "fileContent", "CreatedAt",
            "UpdatedAt",
        ])


class ValidationStatusSeedTests(unittest.TestCase):
    def test_catalog_has_exactly_the_five_agreed_statuses(self) -> None:
        self.assertEqual(validation_status_seed_rows(), [
            {"id": 1, "name": "pending"},
            {"id": 2, "name": "ready"},
            {"id": 3, "name": "ready_with_warnings"},
            {"id": 4, "name": "review_required"},
            {"id": 5, "name": "failed"},
        ])
        self.assertEqual(len(VALIDATION_STATUS_SEEDS), 5)
        self.assertEqual(len({seed.id for seed in VALIDATION_STATUS_SEEDS}), 5)
        self.assertEqual(len({seed.name for seed in VALIDATION_STATUS_SEEDS}), 5)

    def test_seed_definitions_are_immutable_and_rows_are_fresh(self) -> None:
        original = validation_status_seed_rows()
        modified = validation_status_seed_rows()
        modified[0]["name"] = "changed"
        modified.append({"id": 99, "name": "extra"})
        self.assertEqual(validation_status_seed_rows(), original)
        with self.assertRaises(AttributeError):
            setattr(VALIDATION_STATUS_SEEDS[0], "name", "changed")


if __name__ == "__main__":
    unittest.main()
