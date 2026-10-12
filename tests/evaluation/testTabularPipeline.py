import json
import unittest
from io import BytesIO
from unittest.mock import patch

import httpx
from openpyxl import Workbook

from aquai.application.dtos.tabular import PreparationRecipe
from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.llm.ollama import OllamaProvider


def recipe(sheet_name: str = "data", **sheet_overrides: object) -> dict:
    sheet = {
        "sheet_name": sheet_name,
        "block_id": "observations",
        "role": "observations",
        "header_rows": [1],
        "data_start_row": 2,
        "data_end_row": 3,
        "data_start_column": 1,
        "data_end_column": 2,
        "transformations": [],
    }
    sheet.update(sheet_overrides)
    return {"status": "ready", "sheets": [sheet], "warnings": [], "requires_human_review": False}


class TabularPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.settings = TabularSettings(_env_file=None)
        self.raw_content = json.dumps(recipe())
        self.requests: list[dict] = []
        self.provider_error: Exception | None = None
        self.evaluator = self._create_evaluator(self.settings)

    def _create_evaluator(self, settings: TabularSettings) -> TabularEvaluationService:
        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            if self.provider_error is not None:
                raise self.provider_error
            payload = self.requests[-1]
            return httpx.Response(200, json={
                "model": payload["model"],
                "done": True,
                "message": {"role": "assistant", "content": self.raw_content},
                "eval_count": 37,
            })

        return TabularEvaluationService(
            PandasTabularProcessor(),
            OllamaProvider(settings, transport=httpx.MockTransport(respond)),
            default_model=settings.default_model,
            max_upload_bytes=settings.max_upload_bytes,
        )

    async def evaluate(
        self, content: bytes = b"name,score\n Ana ,10\nLuis,20\n",
        *, file_name: str | None = "data.csv", **kwargs,
    ) -> dict:
        result = await self.evaluator.execute(file_name, content, **kwargs)
        return result.model_dump(mode="json")

    async def test_csv_full_pipeline_returns_counts_samples_raw_response_and_metrics(self) -> None:
        self.raw_content = json.dumps(recipe(transformations=[
            {"operation": "strip_strings"},
            {"operation": "convert_type", "columns": ["score"], "target_type": "integer"},
        ]))
        with patch("aquai.infrastructure.persistence.dbContext.create_engine") as create_engine:
            body = await self.evaluate(model="test-model:small")
            create_engine.assert_not_called()
        self.assertEqual(body["model"], "test-model:small")
        self.assertEqual(body["validation"]["status"], "ready")
        self.assertEqual(body["structural_summary"]["sheets"][0]["rows"], 3)
        sheet = body["execution"]["sheets"][0]
        self.assertEqual((sheet["rows_before"], sheet["rows_after"]), (2, 2))
        self.assertEqual(sheet["columns_after"], ["name", "score"])
        self.assertEqual(sheet["sample"][0]["values"], ["Ana", 10])
        self.assertEqual(body["llm_response"]["content"], self.raw_content)
        self.assertEqual(body["llm_response"]["metadata"]["eval_count"], 37)
        self.assertGreater(body["metrics"]["total_duration_ms"], 0)
        self.assertEqual(self.requests[0]["model"], "test-model:small")
        self.assertFalse(self.requests[0]["stream"])
        self.assertEqual(body["response_format"], {
            "proposed_status": "ready",
            "json_valid": True,
            "strict_json": True,
            "cleanup_required": False,
            "schema_valid": True,
            "recipe_valid": True,
            "recipe_executable": True,
        })

    async def test_known_rules_and_default_model_reach_provider(self) -> None:
        result = await self.evaluate(known_rules="Conservar todos los registros.")
        self.assertEqual(result["model"], self.settings.default_model)
        user_message = json.loads(self.requests[0]["messages"][1]["content"])
        self.assertEqual(user_message["known_rules"], "Conservar todos los registros.")
        self.assertEqual(self.requests[0]["model"], self.settings.default_model)

    async def test_unit_headers_and_exact_views_keep_periods_exclusives_and_source_uncertainty(self) -> None:
        workbook = Workbook()
        original = workbook.active
        original.title = "Original"
        sorted_sheet = workbook.create_sheet("Sorted")
        empty_sheet = workbook.create_sheet("Empty")
        empty_sheet["A40"].number_format = "0.00"
        headers = ["site", "period", "count", "oxygen", "note"]
        units = [None, None, "cells/mL", "%", None]
        first = ["001", "3-4/2006", 0, 85, "No sé si hay alguna repetida con otra campaña."]
        second = ["002", "12/2009-01/2010", "<0,001", 90, None]
        exclusive = ["003", "2021", 10, None, None]
        for sheet, rows in ((original, [first, second, exclusive]), (sorted_sheet, [second, first])):
            sheet.append(headers)
            sheet.append(units)
            for row in rows:
                sheet.append(row)
        content = BytesIO()
        workbook.save(content)
        workbook.close()
        blocks = [recipe(
            sheet_name=name, block_id=name.lower(), header_rows=[1, 2],
            data_start_row=3, data_end_row=end, data_end_column=5,
        )["sheets"][0] for name, end in (("Original", 5), ("Sorted", 4))]
        self.raw_content = json.dumps({
            "status": "ready", "sheets": blocks, "warnings": [], "requires_human_review": False,
        })
        result = await self.evaluate(content.getvalue(), file_name="measurements.xlsx")
        self.assertEqual(result["validation"]["status"], "review_required")
        self.assertEqual(result["validation"]["errors"], [])
        self.assertTrue(any(item["code"] == "source_record_identity_uncertainty"
                            for item in result["validation"]["findings"]))
        self.assertFalse(any(item["code"] == "structural_role_conflict"
                             for item in result["validation"]["findings"]))
        self.assertEqual(len(result["execution"]["sheets"]), 2)
        self.assertEqual([sheet["rows_after"] for sheet in result["execution"]["sheets"]], [3, 2])
        self.assertIn("oxygen %", result["execution"]["sheets"][0]["columns_after"])
        self.assertEqual(result["execution"]["sheets"][0]["sample"][0]["values"], first)
        self.assertEqual(result["execution"]["sheets"][0]["sample"][1]["values"], second)
        overlap = result["execution"]["record_overlaps"][0]
        self.assertEqual(overlap["matches"], [{"left_row": 3, "right_row": 4}, {"left_row": 4, "right_row": 3}])
        self.assertEqual(overlap["left_only_rows"], [5])
        self.assertEqual(overlap["right_only_rows"], [])
        self.assertEqual(overlap["policy"], "preserve_sources_do_not_add_matches")
        self.assertEqual(len(self.requests), 1)

    async def test_multiple_blocks_keep_totals_and_legend_without_reporting_data_loss(self) -> None:
        observations = recipe(
            block_id="observations", role="observations", data_end_row=3,
            data_end_column=2,
            transformations=[{"operation": "convert_type", "columns": ["count"], "target_type": "integer"}],
        )["sheets"][0]
        total = recipe(
            block_id="total", role="total", data_start_row=4, data_end_row=4,
            data_end_column=2,
        )["sheets"][0]
        legend = recipe(
            block_id="legend", role="legend", header_rows=[], data_start_row=6,
            data_end_row=6, data_end_column=2,
        )["sheets"][0]
        self.raw_content = json.dumps({
            "status": "ready", "sheets": [observations, total, legend],
            "warnings": [], "requires_human_review": False,
        })
        result = await self.evaluate(b"taxon,count\nA,0\nB,2\nTotal,2\n\nnote,nd\n")
        self.assertIn(result["validation"]["status"], {"ready", "ready_with_warnings"})
        self.assertTrue(result["response_format"]["recipe_executable"])
        self.assertEqual(result["execution"]["coverage"][0]["uncovered_nonempty_cells"], 0)
        self.assertEqual(result["execution"]["sheets"][0]["sample"][0]["values"], ["A", 0])
        self.assertEqual(result["execution"]["sheets"][2]["sample"][0]["values"], ["note", "nd"])
        self.assertEqual(len(self.requests), 1)

    async def test_text_around_recipe_is_tolerated_and_preserved(self) -> None:
        self.raw_content = "Propuesta:\n```json\n" + json.dumps(recipe()) + "\n```\nFin."
        result = await self.evaluate()
        self.assertEqual(result["validation"]["status"], "ready")
        self.assertEqual(result["llm_response"]["content"], self.raw_content)
        self.assertTrue(result["response_format"]["cleanup_required"])
        self.assertFalse(result["response_format"]["strict_json"])

    async def test_invalid_or_executable_recipes_fail_without_executing(self) -> None:
        invalid_responses = [
            "No puedo decidir.",
            json.dumps(recipe(transformations=[{"operation": "run_python", "code": "raise Exception()"}])),
            json.dumps(recipe(data_start_row="2")),
            json.dumps({**recipe(), "code": "print('unexpected')"}),
            '{"nested":' + '[' * 2000 + '0' + ']' * 2000 + '}',
            '{"status":"failed","status":"ready","sheets":[{"sheet_name":"data","data_start_row":1}]}',
        ]
        for content in invalid_responses:
            with self.subTest(content=content):
                self.raw_content = content
                result = await self.evaluate()
                body = result
                self.assertEqual(body["validation"]["status"], "failed")
                self.assertIsNone(body["recipe"])
                self.assertEqual(body["execution"]["sheets"], [])
                self.assertEqual(body["llm_response"]["content"], content)
                self.assertTrue(body["validation"]["errors"])

    async def test_prose_with_json_metadata_before_recipe_is_tolerated(self) -> None:
        self.raw_content = 'Observación {"note":"muestra parcial"}. Receta: ' + json.dumps(recipe())
        result = await self.evaluate()
        self.assertEqual(result["validation"]["status"], "ready")

    async def test_failed_conversion_requires_review(self) -> None:
        self.raw_content = json.dumps(recipe(transformations=[
            {"operation": "convert_type", "columns": ["score"], "target_type": "integer"},
        ]))
        result = await self.evaluate(b"name,score\nAna,invalid\nLuis,20\n")
        self.assertEqual(result["validation"]["status"], "review_required")
        self.assertEqual(result["execution"]["sheets"][0]["conversion_failures"], {"score": 1})

    async def test_nonempty_rows_excluded_by_recipe_require_review(self) -> None:
        self.raw_content = json.dumps(recipe(data_start_row=3))
        result = await self.evaluate()
        self.assertEqual(result["validation"]["status"], "review_required")
        self.assertEqual(result["execution"]["sheets"][0]["excluded_nonempty_rows"], 1)

    async def test_excel_multiple_sheets_and_unresolved_formula_require_review(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Records"
        sheet.append(["name", "score"])
        sheet.append(["Ana", "=1+1"])
        workbook.create_sheet("Other").append(["Un registro que no debe desaparecer"])
        content = BytesIO()
        workbook.save(content)
        workbook.close()
        self.raw_content = json.dumps(recipe(sheet_name="Records", data_end_row=2))
        body = await self.evaluate(content.getvalue(), file_name="records.xlsx")
        self.assertEqual(body["validation"]["status"], "review_required")
        self.assertEqual(body["execution"]["skipped_sheets"], ["Other"])
        self.assertEqual(body["execution"]["sheets"][0]["unresolved_formula_cells"], ["B2"])
        self.assertEqual(body["structural_summary"]["sheets"][0]["formula_count"], 1)

    async def test_provider_timeout_is_an_evaluation_failure(self) -> None:
        self.provider_error = httpx.ReadTimeout("timeout")
        result = await self.evaluate()
        body = result
        self.assertEqual(body["validation"]["status"], "failed")
        self.assertIsNone(body["llm_response"])
        self.assertTrue(body["validation"]["errors"])
        self.assertGreater(body["metrics"]["llm_duration_ms"], 0)
        self.assertIsNone(body["response_format"]["schema_valid"])
        self.assertIsNone(body["response_format"]["recipe_executable"])

    async def test_model_can_report_review_without_an_executable_recipe(self) -> None:
        self.raw_content = json.dumps({
            "status": "review_required", "sheets": [], "requires_human_review": True,
            "warnings": ["No se identificó un encabezado confiable."],
        })
        result = await self.evaluate()
        body = result
        self.assertEqual(body["validation"]["status"], "review_required")
        self.assertEqual(body["execution"]["sheets"], [])
        self.assertTrue(body["response_format"]["recipe_valid"])
        self.assertFalse(body["response_format"]["recipe_executable"])

    async def test_failed_recipe_does_not_execute_even_if_it_includes_sheets(self) -> None:
        self.raw_content = json.dumps({
            **recipe(), "status": "failed", "warnings": ["No puede justificarse una propuesta utilizable."],
        })
        result = await self.evaluate()
        body = result
        self.assertEqual(body["validation"]["status"], "failed")
        self.assertEqual(body["execution"]["sheets"], [])
        self.assertEqual(body["metrics"]["execution_duration_ms"], 0)
        self.assertFalse(body["response_format"]["recipe_executable"])

    async def test_invalid_uploads_never_call_the_provider(self) -> None:
        for file_name, content in [
            ("data.pdf", b"pdf"), ("data.csv", b""), ("data.xlsx", b"broken"), (None, b"data"),
        ]:
            with self.subTest(file_name=file_name, content=content):
                with self.assertRaises(TabularInputError):
                    await self.evaluate(content, file_name=file_name)
        self.assertEqual(self.requests, [])

    async def test_upload_limit_rejects_before_processing(self) -> None:
        settings = TabularSettings(_env_file=None, max_upload_bytes=5)
        evaluator = self._create_evaluator(settings)
        with self.assertRaises(FileTooLargeError):
            await evaluator.execute("data.csv", b"123456")
        self.assertEqual(self.requests, [])

    async def test_invalid_model_and_rules_are_rejected_before_provider_call(self) -> None:
        for parameters in (
            {"model": "   "},
            {"model": "x" * 201},
            {"known_rules": "x" * 4001},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(TabularInputError):
                    await self.evaluate(**parameters)
        self.assertEqual(self.requests, [])

    async def test_provider_uses_shared_recipe_schema(self) -> None:
        await self.evaluate()
        self.assertEqual(self.requests[0]["format"], PreparationRecipe.model_json_schema())


if __name__ == "__main__":
    unittest.main()
