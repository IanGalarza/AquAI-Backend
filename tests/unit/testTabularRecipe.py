import json
import unittest

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from aquai.application.dtos.tabular import PreparationRecipe, SheetRecipe
from aquai.application.services.tabularRecipe import (
    RecipeParseError,
    parse_preparation_recipe,
    parse_preparation_recipe_with_metrics,
)


def recipe_document() -> dict:
    return {
        "status": "ready",
        "sheets": [{
            "sheet_name": "data",
            "block_id": "observations",
            "role": "observations",
            "header_rows": [1],
            "data_start_row": 2,
            "data_end_row": 2,
            "data_start_column": 1,
            "data_end_column": 1,
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
        self.assertNotIn("properties", schema)
        self.assertNotIn("properties", schema["$defs"]["SheetRecipe"])
        self.assertNotIn("allOf", schema)
        for state_schema in schema["oneOf"]:
            self.assertEqual(set(state_schema["required"]), {"status", "sheets", "warnings", "requires_human_review"})
            self.assertFalse(state_schema["additionalProperties"])
        for name in ("SheetRecipeHeader", "SheetRecipeUnknown", "SheetRecipeContent"):
            sheet_schema = schema["$defs"][name]
            self.assertEqual(set(sheet_schema["required"]), {
                "sheet_name", "block_id", "role", "header_rows", "data_start_row",
                "data_end_row", "data_start_column", "data_end_column", "transformations",
            })
            self.assertFalse(sheet_schema["additionalProperties"])
            for field in sheet_schema["required"]:
                self.assertNotIn("default", sheet_schema["properties"][field])
            for field in ("data_start_row", "data_end_row", "data_start_column", "data_end_column"):
                self.assertEqual(sheet_schema["properties"][field]["type"], "integer")
                self.assertEqual(sheet_schema["properties"][field]["minimum"], 1)
        self.assertEqual(schema["$defs"]["SheetRecipeHeader"]["properties"]["header_rows"]["maxItems"], 0)
        self.assertEqual(schema["$defs"]["SheetRecipeUnknown"]["properties"]["transformations"]["maxItems"], 0)

    def test_role_branches_agree_with_pydantic_for_every_permitted_role(self) -> None:
        schema = SheetRecipe.model_json_schema()
        validator = Draft202012Validator(schema)
        branches = [Draft202012Validator({**branch, "$defs": schema["$defs"]}) for branch in schema["oneOf"]]
        for role in ("observations", "total", "legend", "derived", "summary", "header", "unknown"):
            for headers in ([], [1]):
                for transformations in ([], [{"operation": "strip_strings"}]):
                    with self.subTest(role=role, headers=headers, transformations=transformations):
                        block = recipe_document()["sheets"][0]
                        block.update(role=role, header_rows=headers, transformations=transformations)
                        expected = (role != "header" or not headers) and (role != "unknown" or not transformations)
                        self.assertEqual(validator.is_valid(block), expected)
                        self.assertEqual(sum(branch.is_valid(block) for branch in branches), int(expected))
                        if expected:
                            SheetRecipe.model_validate(block)
                        else:
                            with self.assertRaises(ValidationError):
                                SheetRecipe.model_validate(block)

    def test_state_branches_agree_with_pydantic_for_roles_flags_and_warnings(self) -> None:
        schema = PreparationRecipe.model_json_schema()
        validator = Draft202012Validator(schema)
        for status in ("ready", "review_required", "failed"):
            for role in ("observations", "header", "unknown"):
                for requires_review in (False, True):
                    for warnings in ([], ["Motivo verificable de revisión o impedimento."]):
                        with self.subTest(status=status, role=role, requires_review=requires_review, warnings=warnings):
                            document = recipe_document()
                            document.update(status=status, requires_human_review=requires_review, warnings=warnings)
                            document["sheets"][0].update(role=role, header_rows=[])
                            valid_pydantic = True
                            try:
                                PreparationRecipe.model_validate(document)
                            except ValidationError:
                                valid_pydantic = False
                            self.assertEqual(validator.is_valid(document), valid_pydantic)

    def test_header_self_reference_fails_schema_without_repairs(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(
            block_id="initial-header", role="header", data_start_row=1, data_end_row=1,
        )
        document["sheets"].append({
            **recipe_document()["sheets"][0], "header_rows": [], "data_start_row": 2, "data_end_row": 3,
        })
        error = self.assert_parse_failure(json.dumps(document))
        self.assertFalse(error.response_format.schema_valid)
        self.assertIsNone(error.response_format.recipe_valid)
        self.assertFalse(error.response_format.recipe_executable)
        self.assertIn("sheets.0.header_rows", str(error))
        self.assertNotIn("requires_human_review", str(error))
        self.assertNotIn("status:", str(error))

    def test_nonblank_patterns_preserve_existing_string_semantics_and_legal_newlines(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(sheet_name="first\nsecond", block_id="block\nname")
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.schema_valid)
        properties = PreparationRecipe.model_json_schema()["$defs"]["SheetRecipeContent"]["properties"]
        for field in ("sheet_name", "block_id"):
            self.assertEqual(properties[field]["pattern"], r"\S")

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
        for field in ("sheet_name", "block_id", "role", "header_rows", "data_start_row",
                      "data_end_row", "data_start_column", "data_end_column", "transformations"):
            with self.subTest(sheet_field=field):
                document = recipe_document()
                del document["sheets"][0][field]
                self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(document)

    def test_previous_recipe_without_final_bounds_and_identifier_is_rejected(self) -> None:
        document = recipe_document()
        for field in ("block_id", "role", "data_end_row", "data_start_column", "data_end_column"):
            del document["sheets"][0][field]
        error = self.assert_parse_failure(json.dumps(document))
        self.assertFalse(error.response_format.schema_valid)
        self.assertIsNone(error.response_format.recipe_valid)
        self.assertFalse(error.response_format.recipe_executable)
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)

    def test_limits_must_be_explicit_positive_nonnull_integers(self) -> None:
        for field in ("data_start_row", "data_end_row", "data_start_column", "data_end_column"):
            for value in (None, 0, -1, True, "2", 2.0):
                with self.subTest(field=field, value=value):
                    document = recipe_document()
                    document["sheets"][0][field] = value
                    with self.assertRaises(ValidationError):
                        PreparationRecipe.model_validate(document)
                    metrics = self.assert_parse_failure(json.dumps(document)).response_format
                    self.assertEqual(metrics.schema_valid, type(value) is float)

    def test_block_identifier_cannot_be_null_blank_or_whitespace(self) -> None:
        for value in (None, "", "  ", "\n\t"):
            with self.subTest(value=value):
                document = recipe_document()
                document["sheets"][0]["block_id"] = value
                self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(document)

    def test_header_and_transformation_lists_can_be_empty(self) -> None:
        document = recipe_document()
        document["sheets"][0]["header_rows"] = []
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertEqual(parsed.recipe.sheets[0].header_rows, [])
        self.assertEqual(parsed.recipe.sheets[0].transformations, [])

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
        reversed_headers = recipe_document()
        reversed_headers["sheets"][0].update(header_rows=[2, 1], data_start_row=3, data_end_row=3)
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
        document.update(status="review_required", sheets=[], requires_human_review=True,
                        warnings=["No se pudieron justificar los límites de la región."])
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.recipe_valid)
        self.assertFalse(parsed.response_format.recipe_executable)

    def test_recipe_warning_count_is_bounded_by_schema_and_pydantic(self) -> None:
        warning_schema = next(
            state["properties"]["warnings"] for state in PreparationRecipe.model_json_schema()["oneOf"]
            if state["properties"]["status"]["const"] == "review_required"
        )
        self.assertEqual(warning_schema["maxItems"], 20)
        document = recipe_document()
        document.update(status="review_required", requires_human_review=True)
        document["warnings"] = ["Advertencia de prueba."] * 20
        self.assertEqual(len(PreparationRecipe.model_validate(document).warnings), 20)
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.schema_valid)
        document["warnings"].append("Advertencia adicional.")
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)
        self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)

    def test_recipe_warning_length_is_bounded_by_schema_and_pydantic(self) -> None:
        warning_schema = next(
            state["properties"]["warnings"] for state in PreparationRecipe.model_json_schema()["oneOf"]
            if state["properties"]["status"]["const"] == "review_required"
        )
        self.assertEqual(warning_schema["items"]["maxLength"], 512)
        document = recipe_document()
        document.update(status="review_required", requires_human_review=True)
        document["warnings"] = ["a" * 512] * 20
        self.assertEqual(len(PreparationRecipe.model_validate(document).warnings[0]), 512)
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.schema_valid)
        document["warnings"][0] += "a"
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)
        self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)

    def test_explicit_blocks_can_share_a_sheet_and_header_without_overlapping_data(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(
            block_id="observations", data_end_row=4, data_start_column=1, data_end_column=2,
        )
        document["sheets"].append({
            "sheet_name": "data", "block_id": "derived", "role": "derived", "header_rows": [1],
            "data_start_row": 2, "data_end_row": 4, "data_start_column": 3, "data_end_column": 4,
            "transformations": [],
        })
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.recipe_valid)
        self.assertEqual([block.block_id for block in parsed.recipe.sheets], ["observations", "derived"])

    def test_block_identifiers_must_be_unique_even_across_different_sheets(self) -> None:
        for sheet_name in ("data", "another-sheet"):
            with self.subTest(sheet_name=sheet_name):
                document = recipe_document()
                document["sheets"].append({**document["sheets"][0], "sheet_name": sheet_name})
                metrics = self.assert_parse_failure(json.dumps(document)).response_format
                self.assertTrue(metrics.schema_valid)
                self.assertFalse(metrics.recipe_valid)
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(document)

    def test_status_coherence_is_enforced_by_schema_and_pydantic(self) -> None:
        documents = []
        empty_ready = recipe_document()
        empty_ready["sheets"] = []
        documents.append(empty_ready)
        ready_with_review = recipe_document()
        ready_with_review["requires_human_review"] = True
        documents.append(ready_with_review)
        ready_with_warning = recipe_document()
        ready_with_warning["warnings"] = ["No se conoce el significado de esta región."]
        documents.append(ready_with_warning)
        review_without_flag = recipe_document()
        review_without_flag.update(status="review_required", warnings=["Falta confirmar la región."])
        documents.append(review_without_flag)
        review_without_explanation = recipe_document()
        review_without_explanation.update(status="review_required", requires_human_review=True)
        documents.append(review_without_explanation)
        empty_review_without_explanation = {**review_without_explanation, "sheets": []}
        documents.append(empty_review_without_explanation)
        review_with_blank_explanation = {**review_without_explanation, "warnings": ["   "]}
        documents.append(review_with_blank_explanation)
        failed_without_explanation = {**recipe_document(), "status": "failed"}
        documents.append(failed_without_explanation)
        documents.append({**failed_without_explanation, "sheets": []})
        for document in documents:
            with self.subTest(document=document):
                metrics = self.assert_parse_failure(json.dumps(document)).response_format
                self.assertFalse(metrics.schema_valid)
                self.assertIsNone(metrics.recipe_valid)
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(document)

    def test_failed_recipe_retains_identified_impediment_and_is_not_executable(self) -> None:
        document = recipe_document()
        document.update(status="failed", sheets=[], warnings=["No hay evidencia suficiente para una propuesta utilizable."])
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.recipe_valid)
        self.assertFalse(parsed.response_format.recipe_executable)

    def test_header_block_cannot_name_another_header_or_copy_observations(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(
            block_id="header-copy", role="header", data_end_row=4, data_end_column=2,
        )
        document["sheets"].append({
            **document["sheets"][0], "block_id": "observations", "role": "observations", "header_rows": [],
        })
        error = self.assert_parse_failure(json.dumps(document))
        self.assertFalse(error.response_format.schema_valid)
        with self.assertRaises(ValidationError):
            PreparationRecipe.model_validate(document)
        document["sheets"][0]["header_rows"] = []
        error = self.assert_parse_failure(json.dumps(document))
        self.assertTrue(error.response_format.schema_valid)
        self.assertFalse(error.response_format.recipe_valid)
        self.assertIn("header-copy", str(error))
        self.assertIn("observations", str(error))
        self.assertIn("data", str(error))

    def test_header_references_cannot_reuse_other_nonheader_data(self) -> None:
        for role in ("observations", "total", "legend", "derived", "summary", "unknown"):
            with self.subTest(role=role):
                document = recipe_document()
                document["sheets"][0].update(role=role, data_end_row=3, data_end_column=2)
                if role == "unknown":
                    document.update(status="review_required", requires_human_review=True,
                                    warnings=["No se conoce el rol de data!A2:B3."])
                document["sheets"].append({
                    "sheet_name": "data", "block_id": "later", "role": "observations", "header_rows": [2],
                    "data_start_row": 4, "data_end_row": 4, "data_start_column": 2,
                    "data_end_column": 3, "transformations": [],
                })
                error = self.assert_parse_failure(json.dumps(document))
                self.assertTrue(error.response_format.schema_valid)
                self.assertFalse(error.response_format.recipe_valid)
                self.assertIn("later", str(error))
                self.assertIn("observations", str(error))
                self.assertIn("filas [2]", str(error))

    def test_header_references_in_disjoint_columns_do_not_conflict(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(data_end_row=3, data_end_column=2)
        document["sheets"].append({
            "sheet_name": "data", "block_id": "later", "role": "observations", "header_rows": [2],
            "data_start_row": 4, "data_end_row": 4, "data_start_column": 3, "data_end_column": 4,
            "transformations": [],
        })
        self.assertTrue(parse_preparation_recipe_with_metrics(json.dumps(document)).response_format.recipe_valid)

    def test_explicit_header_content_and_shared_references_are_valid(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(data_end_row=3, data_end_column=2)
        document["sheets"].extend([
            {
                "sheet_name": "data", "block_id": "stored-header", "role": "header", "header_rows": [],
                "data_start_row": 1, "data_end_row": 1, "data_start_column": 1, "data_end_column": 2,
                "transformations": [],
            },
            {
                "sheet_name": "data", "block_id": "later", "role": "observations", "header_rows": [1],
                "data_start_row": 4, "data_end_row": 5, "data_start_column": 1, "data_end_column": 2,
                "transformations": [],
            },
        ])
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.schema_valid)
        self.assertTrue(parsed.response_format.recipe_valid)

    def test_unknown_block_is_valid_only_untouched_and_with_explained_review(self) -> None:
        document = recipe_document()
        document.update(status="review_required", requires_human_review=True,
                        warnings=["data!A2: el rol de la región no puede determinarse."])
        document["sheets"][0]["role"] = "unknown"
        parsed = parse_preparation_recipe_with_metrics(json.dumps(document))
        self.assertTrue(parsed.response_format.recipe_valid)
        self.assertEqual(parsed.recipe.sheets[0].transformations, [])
        transformed_unknown = json.loads(json.dumps(document))
        transformed_unknown["sheets"][0]["transformations"] = [{"operation": "strip_strings"}]
        ready_unknown = {**document, "status": "ready", "warnings": [], "requires_human_review": False}
        failed_unknown = {**document, "status": "failed"}
        for invalid in (transformed_unknown, ready_unknown, failed_unknown):
            with self.subTest(document=invalid):
                self.assertFalse(self.assert_parse_failure(json.dumps(invalid)).response_format.schema_valid)
                with self.assertRaises(ValidationError):
                    PreparationRecipe.model_validate(invalid)

    def test_overlapping_explicit_data_rectangles_are_rejected(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(block_id="first", data_end_row=4, data_end_column=2)
        document["sheets"].append({**document["sheets"][0], "block_id": "second", "data_start_row": 4})
        error = self.assert_parse_failure(json.dumps(document))
        self.assertTrue(error.response_format.schema_valid)
        self.assertFalse(error.response_format.recipe_valid)
        self.assertIn("solaparse", str(error))

    def test_reversed_column_range_is_rejected_by_the_business_contract(self) -> None:
        document = recipe_document()
        document["sheets"][0].update(data_start_column=3, data_end_column=2)
        error = self.assert_parse_failure(json.dumps(document))
        self.assertTrue(error.response_format.schema_valid)
        self.assertFalse(error.response_format.recipe_valid)

    def test_unknown_block_roles_fail_the_json_schema(self) -> None:
        document = recipe_document()
        document["sheets"][0]["role"] = "toxic"
        self.assertFalse(self.assert_parse_failure(json.dumps(document)).response_format.schema_valid)


if __name__ == "__main__":
    unittest.main()
