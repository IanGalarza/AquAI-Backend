import json
import unittest

from pydantic import ValidationError

from aquai.application.dtos.tabular import PreparationRecipe
from aquai.application.services.tabular_recipe import (
    RecipeParseError,
    parse_preparation_recipe,
    parse_preparation_recipe_with_metrics,
)


def recipe_document() -> dict:
    return {
        "status": "ready",
        "sheets": [{
            "sheet_name": "data",
            "header_rows": [1],
            "data_start_row": 2,
            "transformations": [],
        }],
        "warnings": [],
        "requires_human_review": False,
    }


class TabularRecipeTests(unittest.TestCase):
    def assert_parse_failure(self, content: str) -> RecipeParseError:
        with self.assertRaises(RecipeParseError) as captured:
            parse_preparation_recipe_with_metrics(content)
        return captured.exception

    def test_generated_schema_requires_structural_fields_and_forbids_extras(self) -> None:
        schema = PreparationRecipe.model_json_schema()
        self.assertEqual(set(schema["required"]), {"status", "sheets", "warnings", "requires_human_review"})
        sheet_schema = schema["$defs"]["SheetRecipe"]
        self.assertEqual(set(sheet_schema["required"]), {
            "sheet_name", "header_rows", "data_start_row", "transformations",
        })
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(sheet_schema["additionalProperties"])

    def test_missing_required_fields_are_rejected_by_schema_and_pydantic(self) -> None:
        for field in ("status", "sheets", "warnings", "requires_human_review"):
            with self.subTest(field=field):
                document = recipe_document()
                del document[field]
                error = self.assert_parse_failure(json.dumps(document))
                self.assertTrue(error.response_format.json_valid)
                self.assertFalse(error.response_format.schema_valid)
                self.assertIsNone(error.response_format.recipe_valid)
                self.assertIn(field, str(error))
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(document)
        for field in ("sheet_name", "header_rows", "data_start_row", "transformations"):
            with self.subTest(sheet_field=field):
                document = recipe_document()
                del document["sheets"][0][field]
                self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)

    def test_direct_json_has_no_cleanup_and_compatibility_parser_returns_recipe(self) -> None:
        content = " \n" + json.dumps(recipe_document()) + "\n "
        parsed = parse_preparation_recipe_with_metrics(content)
        self.assertEqual(parsed.response_format.model_dump(), {
            "proposed_status": "ready",
            "json_valid": True,
            "strict_json": True,
            "cleanup_required": False,
            "schema_valid": True,
            "recipe_valid": True,
            "recipe_executable": False,
        })
        self.assertEqual(parse_preparation_recipe(content), parsed.recipe)

    def test_markdown_and_prose_are_distinguished_from_direct_json(self) -> None:
        for prefix, suffix in (("```json\n", "\n```"), ('Nota {"note":"datos"}. Receta: ', " Fin.")):
            with self.subTest(prefix=prefix):
                parsed = parse_preparation_recipe_with_metrics(prefix + json.dumps(recipe_document()) + suffix)
                self.assertTrue(parsed.response_format.json_valid)
                self.assertTrue(parsed.response_format.cleanup_required)
                self.assertFalse(parsed.response_format.strict_json)
                self.assertTrue(parsed.response_format.schema_valid)
                self.assertTrue(parsed.response_format.recipe_valid)

    def test_json_validity_and_schema_validity_are_separate(self) -> None:
        for content in ('{"explanation":"sin receta"}', '[{"status":"ready"}]', 'null', '"respuesta"'):
            with self.subTest(content=content):
                metrics = self.assert_parse_failure(content).response_format
                self.assertTrue(metrics.json_valid)
                self.assertTrue(metrics.strict_json)
                self.assertFalse(metrics.cleanup_required)
                self.assertFalse(metrics.schema_valid)
                self.assertIsNone(metrics.recipe_valid)

    def test_invalid_json_preserves_not_attempted_schema_and_contract_metrics(self) -> None:
        for content in ("No puedo decidir.", '{"status":', '{"status":"failed","status":"ready"}',
                        '{"status":NaN}', '{"status":Infinity}'):
            with self.subTest(content=content):
                metrics = self.assert_parse_failure(content).response_format
                self.assertFalse(metrics.json_valid)
                self.assertFalse(metrics.strict_json)
                self.assertIsNone(metrics.schema_valid)
                self.assertIsNone(metrics.recipe_valid)
                self.assertFalse(metrics.recipe_executable)

    def test_unknown_operations_and_extra_fields_fail_schema(self) -> None:
        documents = []
        extra_root = recipe_document()
        extra_root["code"] = "print(1)"
        documents.append(extra_root)
        extra_sheet = recipe_document()
        extra_sheet["sheets"][0]["extra"] = True
        documents.append(extra_sheet)
        unknown_operation = recipe_document()
        unknown_operation["sheets"][0]["transformations"] = [{"operation": "run_python", "code": "print(1)"}]
        documents.append(unknown_operation)
        for document in documents:
            with self.subTest(document=document):
                metrics = self.assert_parse_failure(json.dumps(document)).response_format
                self.assertTrue(metrics.json_valid)
                self.assertFalse(metrics.schema_valid)
                self.assertIsNone(metrics.recipe_valid)

    def test_schema_success_does_not_imply_business_contract_success(self) -> None:
        documents = []
        empty_ready = recipe_document()
        empty_ready["sheets"] = []
        documents.append(empty_ready)
        review_without_flag = recipe_document()
        review_without_flag["status"] = "review_required"
        documents.append(review_without_flag)
        reversed_headers = recipe_document()
        reversed_headers["sheets"][0].update(header_rows=[2, 1], data_start_row=3)
        documents.append(reversed_headers)
        invalid_range = recipe_document()
        invalid_range["sheets"][0]["data_end_row"] = 1
        documents.append(invalid_range)
        duplicate_sheets = recipe_document()
        duplicate_sheets["sheets"] *= 2
        documents.append(duplicate_sheets)
        strict_integer = recipe_document()
        strict_integer["sheets"][0]["data_start_row"] = 2.0
        documents.append(strict_integer)
        for document in documents:
            with self.subTest(document=document):
                error = self.assert_parse_failure(json.dumps(document))
                self.assertTrue(error.response_format.schema_valid)
                self.assertFalse(error.response_format.recipe_valid)
                self.assertEqual(error.response_format.proposed_status, document["status"])
                self.assertIn("Pydantic", str(error))

    def test_proposed_status_survives_schema_failure_without_valid_recipe(self) -> None:
        metrics = self.assert_parse_failure('{"status":"ready"}').response_format
        self.assertEqual(metrics.proposed_status, "ready")
        self.assertFalse(metrics.schema_valid)
        self.assertIsNone(metrics.recipe_valid)

    def test_review_can_be_valid_without_executable_sheets(self) -> None:
        document = recipe_document()
        document.update(status="review_required", sheets=[], requires_human_review=True)
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.recipe_valid)
        self.assertFalse(parsed.response_format.recipe_executable)

    def test_recipe_warning_count_is_bounded_by_schema_and_pydantic(self) -> None:
        warning_schema = PreparationRecipe.model_json_schema()["properties"]["warnings"]
        self.assertEqual(warning_schema["maxItems"], 20)
        document = recipe_document()
        document["warnings"] = ["Advertencia de prueba."] * 20
        self.assertEqual(len(PreparationRecipe.model_validate(document).warnings), 20)
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.schema_valid)
        document["warnings"].append("Advertencia adicional.")
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)
        self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)

    def test_recipe_warning_length_is_bounded_by_schema_and_pydantic(self) -> None:
        warning_schema = PreparationRecipe.model_json_schema()["properties"]["warnings"]
        self.assertEqual(warning_schema["items"]["maxLength"], 512)
        document = recipe_document()
        document["warnings"] = ["a" * 512] * 20
        self.assertEqual(len(PreparationRecipe.model_validate(document).warnings[0]), 512)
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.schema_valid)
        document["warnings"][0] += "a"
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)
        self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)


if __name__ == "__main__":
    unittest.main()
