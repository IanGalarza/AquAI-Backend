import json
import unittest
from io import BytesIO
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openpyxl import Workbook

from aquai.application.use_cases.evaluate_tabular_file import EvaluateTabularFile
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.llm.ollama import OllamaProvider
from aquai.infrastructure.providers.tabular import get_tabular_evaluator, get_tabular_settings
from aquai.main import create_app


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


class TabularEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = TabularSettings(_env_file=None)
        self.raw_content = json.dumps(recipe())
        self.requests: list[dict] = []
        self.provider_error: Exception | None = None
        self.client = self._create_client(self.settings)
        self.addCleanup(self.client.close)

    def _create_client(self, settings: TabularSettings) -> TestClient:
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

        evaluator = EvaluateTabularFile(
            PandasTabularProcessor(),
            OllamaProvider(settings, transport=httpx.MockTransport(respond)),
            default_model=settings.default_model,
            max_upload_bytes=settings.max_upload_bytes,
        )
        app = create_app()
        app.dependency_overrides[get_tabular_evaluator] = lambda: evaluator
        app.dependency_overrides[get_tabular_settings] = lambda: settings
        return TestClient(app)

    def upload(self, content: bytes = b"name,score\n Ana ,10\nLuis,20\n", **kwargs):
        return self.client.post(
            "/experimental/tabular/evaluate",
            files={"file": (kwargs.pop("file_name", "data.csv"), content)},
            **kwargs,
        )

    def test_csv_full_pipeline_returns_counts_samples_raw_response_and_metrics(self) -> None:
        self.raw_content = json.dumps(recipe(transformations=[
            {"operation": "strip_strings"},
            {"operation": "convert_type", "columns": ["score"], "target_type": "integer"},
        ]))
        with patch("aquai.infrastructure.persistence.db_context.create_engine") as create_engine:
            response = self.upload(params={"model": "test-model:small"})
            create_engine.assert_not_called()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
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

    def test_known_rules_and_default_model_reach_provider(self) -> None:
        response = self.upload(data={"known_rules": "Conservar todos los registros."})
        self.assertEqual(response.status_code, 200, response.text)
        user_message = json.loads(self.requests[0]["messages"][1]["content"])
        self.assertEqual(user_message["known_rules"], "Conservar todos los registros.")
        self.assertEqual(self.requests[0]["model"], self.settings.default_model)

    def test_unit_headers_and_exact_views_keep_periods_exclusives_and_source_uncertainty(self) -> None:
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
        response = self.upload(content.getvalue(), file_name="measurements.xlsx")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
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

    def test_multiple_blocks_keep_totals_and_legend_without_reporting_data_loss(self) -> None:
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
        response = self.upload(b"taxon,count\nA,0\nB,2\nTotal,2\n\nnote,nd\n")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertIn(result["validation"]["status"], {"ready", "ready_with_warnings"})
        self.assertTrue(result["response_format"]["recipe_executable"])
        self.assertEqual(result["execution"]["coverage"][0]["uncovered_nonempty_cells"], 0)
        self.assertEqual(result["execution"]["sheets"][0]["sample"][0]["values"], ["A", 0])
        self.assertEqual(result["execution"]["sheets"][2]["sample"][0]["values"], ["note", "nd"])
        self.assertEqual(len(self.requests), 1)

    def test_text_around_recipe_is_tolerated_and_preserved(self) -> None:
        self.raw_content = "Propuesta:\n```json\n" + json.dumps(recipe()) + "\n```\nFin."
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["validation"]["status"], "ready")
        self.assertEqual(response.json()["llm_response"]["content"], self.raw_content)
        self.assertTrue(response.json()["response_format"]["cleanup_required"])
        self.assertFalse(response.json()["response_format"]["strict_json"])

    def test_invalid_or_executable_recipes_fail_without_executing(self) -> None:
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
                response = self.upload()
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["validation"]["status"], "failed")
                self.assertIsNone(body["recipe"])
                self.assertEqual(body["execution"]["sheets"], [])
                self.assertEqual(body["llm_response"]["content"], content)
                self.assertTrue(body["validation"]["errors"])

    def test_prose_with_json_metadata_before_recipe_is_tolerated(self) -> None:
        self.raw_content = 'Observación {"note":"muestra parcial"}. Receta: ' + json.dumps(recipe())
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["validation"]["status"], "ready")

    def test_failed_conversion_requires_review(self) -> None:
        self.raw_content = json.dumps(recipe(transformations=[
            {"operation": "convert_type", "columns": ["score"], "target_type": "integer"},
        ]))
        response = self.upload(b"name,score\nAna,invalid\nLuis,20\n")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["validation"]["status"], "review_required")
        self.assertEqual(response.json()["execution"]["sheets"][0]["conversion_failures"], {"score": 1})

    def test_nonempty_rows_excluded_by_recipe_require_review(self) -> None:
        self.raw_content = json.dumps(recipe(data_start_row=3))
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["validation"]["status"], "review_required")
        self.assertEqual(response.json()["execution"]["sheets"][0]["excluded_nonempty_rows"], 1)

    def test_excel_multiple_sheets_and_unresolved_formula_require_review(self) -> None:
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
        response = self.upload(content.getvalue(), file_name="records.xlsx")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["validation"]["status"], "review_required")
        self.assertEqual(body["execution"]["skipped_sheets"], ["Other"])
        self.assertEqual(body["execution"]["sheets"][0]["unresolved_formula_cells"], ["B2"])
        self.assertEqual(body["structural_summary"]["sheets"][0]["formula_count"], 1)

    def test_provider_timeout_is_an_evaluation_failure(self) -> None:
        self.provider_error = httpx.ReadTimeout("timeout")
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["validation"]["status"], "failed")
        self.assertIsNone(body["llm_response"])
        self.assertTrue(body["validation"]["errors"])
        self.assertGreater(body["metrics"]["llm_duration_ms"], 0)
        self.assertIsNone(body["response_format"]["schema_valid"])
        self.assertIsNone(body["response_format"]["recipe_executable"])

    def test_model_can_report_review_without_an_executable_recipe(self) -> None:
        self.raw_content = json.dumps({
            "status": "review_required", "sheets": [], "requires_human_review": True,
            "warnings": ["No se identificó un encabezado confiable."],
        })
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["validation"]["status"], "review_required")
        self.assertEqual(body["execution"]["sheets"], [])
        self.assertTrue(body["response_format"]["recipe_valid"])
        self.assertFalse(body["response_format"]["recipe_executable"])

    def test_failed_recipe_does_not_execute_even_if_it_includes_sheets(self) -> None:
        self.raw_content = json.dumps({
            **recipe(), "status": "failed", "warnings": ["No puede justificarse una propuesta utilizable."],
        })
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["validation"]["status"], "failed")
        self.assertEqual(body["execution"]["sheets"], [])
        self.assertEqual(body["metrics"]["execution_duration_ms"], 0)
        self.assertFalse(body["response_format"]["recipe_executable"])

    def test_invalid_uploads_never_call_the_provider(self) -> None:
        for file_name, content in [("data.pdf", b"pdf"), ("data.csv", b""), ("data.xlsx", b"broken")]:
            with self.subTest(file_name=file_name, content=content):
                response = self.upload(content, file_name=file_name)
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.requests, [])
        response = self.client.post("/experimental/tabular/evaluate")
        self.assertEqual(response.status_code, 422)

    def test_upload_limit_returns_413_before_processing(self) -> None:
        settings = TabularSettings(_env_file=None, max_upload_bytes=5)
        with self._create_client(settings) as client:
            response = client.post("/experimental/tabular/evaluate", files={"file": ("data.csv", b"123456")})
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(self.requests, [])

    def test_invalid_model_and_rules_are_rejected_before_provider_call(self) -> None:
        for parameters in (
            {"params": {"model": "   "}},
            {"params": {"model": "x" * 201}},
            {"data": {"known_rules": "x" * 4001}},
        ):
            with self.subTest(parameters=parameters):
                response = self.upload(**parameters)
                self.assertIn(response.status_code, (400, 422), response.text)
        self.assertEqual(self.requests, [])

    def test_endpoint_is_documented_in_openapi(self) -> None:
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        operation = schema["paths"]["/experimental/tabular/evaluate"]["post"]
        self.assertIn("multipart/form-data", operation["requestBody"]["content"])
        self.assertEqual(operation["parameters"][0]["name"], "model")


if __name__ == "__main__":
    unittest.main()
