import json
import unittest
from unittest.mock import AsyncMock, Mock

from aquai.application.dtos.tabular import (
    ExecutionResult,
    LLMResponse,
    SheetExecution,
    SheetSummary,
    StructuralSummary,
)
from aquai.application.errors import LLMProviderError
from aquai.application.use_cases.evaluate_tabular_file import EvaluateTabularFile
from aquai.domain.enums.tabular import EvaluationStatus
from tests.unit.test_tabular_recipe import recipe_document


class EvaluateTabularFileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.content = b"a\n1\n"
        self.summary = StructuralSummary(
            file_name="data.csv", file_type="csv", file_size_bytes=len(self.content),
            sheets=[SheetSummary(
                name="data", rows=2, columns=1, initial_columns=["a"],
                inferred_types=["string"], null_counts=[0], sample=[],
            )],
        )
        self.execution = ExecutionResult(sheets=[SheetExecution(
            sheet_name="data", block_id="observations", source_rows=2, source_columns=1,
            data_start_row=2, data_end_row=2, data_start_column=1, data_end_column=1,
            rows_before=1, rows_after=1, columns_before=["a"], columns_after=["a"],
        )])
        self.processor = Mock()
        self.processor.inspect.return_value = self.summary
        self.processor.execute.return_value = self.execution
        self.provider = Mock()
        self.provider.generate_recipe = AsyncMock(return_value=LLMResponse(
            model="test-model", content=json.dumps(recipe_document()),
        ))
        self.evaluator = EvaluateTabularFile(
            self.processor, self.provider, default_model="test-model", max_upload_bytes=100,
        )

    async def evaluate(self, **kwargs):
        return await self.evaluator.execute("data.csv", self.content, **kwargs)

    async def test_normal_request_inspects_and_preserves_response_metrics(self) -> None:
        result = await self.evaluate()
        self.processor.inspect.assert_called_once_with("data.csv", self.content)
        self.assertGreater(result.metrics.inspection_duration_ms, 0)
        self.assertTrue(result.response_format.strict_json)
        self.assertTrue(result.response_format.recipe_valid)
        self.assertTrue(result.response_format.recipe_executable)
        self.assertEqual(result.validation.status, EvaluationStatus.READY)

    async def test_no_response_does_not_claim_attempted_validation(self) -> None:
        self.provider.generate_recipe.side_effect = LLMProviderError("No se pudo contactar Ollama.")
        result = await self.evaluate()
        self.assertIsNone(result.llm_response)
        self.assertFalse(result.response_format.json_valid)
        self.assertIsNone(result.response_format.schema_valid)
        self.assertIsNone(result.response_format.recipe_valid)
        self.assertIsNone(result.response_format.recipe_executable)
        self.processor.execute.assert_not_called()

    async def test_schema_failure_and_contract_failure_keep_distinct_metrics(self) -> None:
        duplicate_blocks = recipe_document()
        duplicate_blocks["sheets"] *= 2
        for document, expected_schema in (({"status": "ready"}, False),
                                          (duplicate_blocks, True)):
            with self.subTest(document=document):
                self.provider.generate_recipe.return_value = LLMResponse(
                    model="test-model", content=json.dumps(document),
                )
                result = await self.evaluate()
                self.assertIsNone(result.recipe)
                self.assertTrue(result.response_format.json_valid)
                self.assertEqual(result.response_format.schema_valid, expected_schema)
                self.assertEqual(result.response_format.recipe_valid, False if expected_schema else None)
                self.assertEqual(result.response_format.proposed_status, "ready")
                self.assertFalse(result.response_format.recipe_executable)
                self.assertEqual(result.validation.status, EvaluationStatus.FAILED)
        self.processor.execute.assert_not_called()

    async def test_review_and_failed_empty_recipes_are_not_executable(self) -> None:
        for status in ("review_required", "failed"):
            with self.subTest(status=status):
                document = recipe_document()
                document.update(status=status, sheets=[], requires_human_review=status == "review_required")
                if status == "review_required":
                    document["warnings"] = ["No se pudieron justificar los límites del archivo."]
                else:
                    document["warnings"] = ["No se encontró evidencia suficiente para una propuesta utilizable."]
                self.provider.generate_recipe.return_value = LLMResponse(
                    model="test-model", content=json.dumps(document),
                )
                self.processor.execute.return_value = ExecutionResult()
                result = await self.evaluate()
                self.assertTrue(result.response_format.recipe_valid)
                self.assertFalse(result.response_format.recipe_executable)
                self.assertEqual(result.validation.status.value, status)

    async def test_executable_recipe_can_require_review_for_data_quality(self) -> None:
        self.execution.sheets[0].conversion_failures = {"a": 1}
        result = await self.evaluate()
        self.assertEqual(result.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(result.response_format.recipe_executable)

    async def test_incomplete_empty_or_failed_execution_is_not_executable(self) -> None:
        for execution in (
            ExecutionResult(),
            ExecutionResult(sheets=[self.execution.sheets[0].model_copy(update={"rows_after": 0})]),
            ExecutionResult(sheets=[self.execution.sheets[0].model_copy(update={"columns_after": []})]),
            ExecutionResult(sheets=[self.execution.sheets[0].model_copy(update={"errors": ["No se ejecutó."]})]),
        ):
            with self.subTest(execution=execution):
                self.processor.execute.return_value = execution
                result = await self.evaluate()
                self.assertTrue(result.response_format.recipe_valid)
                self.assertFalse(result.response_format.recipe_executable)
                self.assertEqual(result.validation.status, EvaluationStatus.FAILED)

    async def test_execution_must_match_block_ids_even_when_sheet_names_match(self) -> None:
        document = recipe_document()
        document["sheets"][0]["block_id"] = "observations"
        self.provider.generate_recipe.return_value = LLMResponse(
            model="test-model", content=json.dumps(document),
        )
        self.execution.sheets[0].block_id = "another-block"
        result = await self.evaluate()
        self.assertFalse(result.response_format.recipe_executable)


if __name__ == "__main__":
    unittest.main()
