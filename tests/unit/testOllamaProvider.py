import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from aquai.application.agents.prompts.tabular import TABULAR_RECIPE_PROMPT
from aquai.application.contracts.tabular.results import ExecutionResult
from aquai.application.contracts.tabular.structure import (
    FormulaSample,
    RecordOverlap,
    SampleRow,
    SheetSummary,
    SourceRowMatch,
    StructuralRegion,
    StructuralSummary,
)
from aquai.application.errors import LLMProviderError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import OllamaProvider
from aquai.infrastructure.providers.llm.recipeSchema import build_recipe_schema


def summary() -> StructuralSummary:
    return StructuralSummary(
        file_name="datos.csv",
        file_type="csv",
        file_size_bytes=30,
        csv_encoding="utf-8",
        csv_delimiter=",",
        sheets=[],
    )


def response_body(content: str = '{"status": "ready"}') -> dict:
    return {
        "model": "qwen3:14b",
        "created_at": "2026-10-10T00:00:00Z",
        "message": {"role": "assistant", "content": content, "thinking": "Análisis del modelo"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2000,
        "load_duration": 100,
        "prompt_eval_count": 200,
        "prompt_eval_duration": 400,
        "eval_count": 30,
        "eval_duration": 1500,
    }


def record_overlap() -> RecordOverlap:
    return RecordOverlap(
        left_sheet="measurements", right_sheet="sorted measurements",
        left_header_rows=[1, 2], right_header_rows=[4, 5],
        left_start_column=2, left_end_column=3, right_start_column=7, right_end_column=8,
        left_record_count=25, right_record_count=26,
        matches=[SourceRowMatch(left_row=row + 3, right_row=row + 6) for row in range(10)],
        left_only_rows=list(range(30, 45)), right_only_rows=list(range(50, 66)),
    )


class RecordOverlapContractTests(unittest.TestCase):
    def test_match_coordinates_and_overlap_coordinates_must_be_positive_integers(self) -> None:
        for coordinate in ("left_row", "right_row"):
            for value in (0, -1, None, True, "1", 1.0):
                with self.subTest(coordinate=coordinate, value=value), self.assertRaises(ValidationError):
                    SourceRowMatch.model_validate({"left_row": 1, "right_row": 1, coordinate: value})
        for coordinate in ("left_start_column", "left_end_column", "right_start_column", "right_end_column"):
            for value in (0, -1, None, True, "1", 1.0):
                with self.subTest(coordinate=coordinate, value=value), self.assertRaises(ValidationError):
                    RecordOverlap.model_validate({**record_overlap().model_dump(), coordinate: value})
        for rows in ("left_header_rows", "right_header_rows", "left_only_rows", "right_only_rows"):
            for value in (0, -1, None, True, "1", 1.0):
                with self.subTest(rows=rows, value=value), self.assertRaises(ValidationError):
                    RecordOverlap.model_validate({**record_overlap().model_dump(), rows: [value]})

    def test_record_counts_must_be_nonnegative_and_default_relations_preserve_compatibility(self) -> None:
        empty = {**record_overlap().model_dump(), "left_record_count": 0, "right_record_count": 0,
                 "matches": [], "left_only_rows": [], "right_only_rows": []}
        RecordOverlap.model_validate(empty)
        for count in ("left_record_count", "right_record_count"):
            for value in (-1, None, True, "1", 1.0):
                with self.subTest(count=count, value=value), self.assertRaises(ValidationError):
                    RecordOverlap.model_validate({**record_overlap().model_dump(), count: value})
        self.assertEqual(summary().record_overlaps, [])
        self.assertEqual(ExecutionResult().record_overlaps, [])
        relation = record_overlap()
        self.assertEqual(relation.comparison, "exact_values_headers_and_formulas")
        self.assertEqual(relation.policy, "preserve_sources_do_not_add_matches")
        self.assertFalse(relation.ambiguous_repeated_rows)

    def test_each_source_row_is_accounted_for_once_in_matches_or_exclusives(self) -> None:
        original = record_overlap().model_dump()
        invalid = [
            {"left_record_count": 24}, {"right_record_count": 27},
            {"matches": original["matches"] + [original["matches"][0]],
             "left_record_count": 26, "right_record_count": 27},
            {"matches": [{**match, "left_row": 3} for match in original["matches"]]},
            {"matches": [{**match, "right_row": 6} for match in original["matches"]]},
            {"left_only_rows": original["left_only_rows"] + [30], "left_record_count": 26},
            {"right_only_rows": original["right_only_rows"] + [50], "right_record_count": 27},
            {"left_only_rows": [3, *original["left_only_rows"][1:]]},
            {"right_only_rows": [6, *original["right_only_rows"][1:]]},
        ]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                RecordOverlap.model_validate({**original, **change})

    def test_exact_comparison_requires_ordered_equal_widths_and_original_headers(self) -> None:
        original = record_overlap().model_dump()
        for change in (
            {"left_start_column": 4}, {"right_start_column": 9},
            {"left_end_column": 4}, {"right_end_column": 9},
            {"left_header_rows": [2, 1]}, {"right_header_rows": [5, 4]},
            {"left_header_rows": [1, 1]}, {"right_header_rows": [4, 4]},
            {"left_header_rows": [3]}, {"right_header_rows": [50]},
            {"comparison": "cached_values_only"}, {"policy": "append_matches"},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                RecordOverlap.model_validate({**original, **change})

    def test_reordered_matches_and_distinct_repeated_value_rows_are_preserved(self) -> None:
        original = record_overlap().model_dump()
        reordered = [
            {**match, "right_row": other["right_row"]}
            for match, other in zip(original["matches"], reversed(original["matches"]), strict=True)
        ]
        relation = RecordOverlap.model_validate({**original, "matches": reordered,
                                               "ambiguous_repeated_rows": True})
        self.assertEqual(relation.model_dump()["matches"], reordered)
        self.assertTrue(relation.ambiguous_repeated_rows)
        self.assertEqual(relation.left_record_count, len(relation.matches) + len(relation.left_only_rows))
        self.assertEqual(relation.right_record_count, len(relation.matches) + len(relation.right_only_rows))


class TabularSettingsTests(unittest.TestCase):
    def test_defaults_do_not_require_database_configuration(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            settings = TabularSettings(_env_file=None)
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434")
        self.assertEqual(settings.default_model, "qwen3.5:9b")
        self.assertEqual(settings.llm_timeout_seconds, 600)
        self.assertEqual(settings.llm_num_ctx, 16384)
        self.assertIs(settings.llm_think, False)
        self.assertEqual(settings.max_upload_bytes, 10 * 1024 * 1024)
        self.assertEqual(settings.max_cells, 2_000_000)
        self.assertEqual(settings.max_sheets, 30)
        self.assertEqual(settings.max_expanded_bytes, 100 * 1024 * 1024)
        self.assertEqual(settings.sample_rows, 12)

    def test_settings_load_prefixed_environment_and_ignore_unrelated_dotenv_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "DATABASE_URL=postgresql://localhost/aquai\n"
                "POSTGRES_PASSWORD=unused\n"
                "TABULAR_DEFAULT_MODEL=granite-local\n"
                "TABULAR_MAX_CELLS=1500\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "TABULAR_OLLAMA_BASE_URL": "https://localhost:11434/",
                    "TABULAR_LLM_TIMEOUT_SECONDS": "45",
                    "TABULAR_LLM_THINK": "true",
                },
                clear=True,
            ):
                settings = TabularSettings(_env_file=path)
        self.assertEqual(settings.ollama_base_url, "https://localhost:11434")
        self.assertEqual(settings.default_model, "granite-local")
        self.assertEqual(settings.max_cells, 1500)
        self.assertEqual(settings.llm_timeout_seconds, 45)
        self.assertIs(settings.llm_think, True)

    def test_invalid_urls_model_and_limits_are_rejected(self) -> None:
        invalid_settings = [
            {"ollama_base_url": "ftp://localhost:11434"},
            {"ollama_base_url": "not-a-url"},
            {"ollama_base_url": "http://localhost:11434?token=invalid"},
            {"ollama_base_url": "http://localhost:11434#invalid"},
            {"default_model": "   "},
            {"default_model": "x" * 201},
            {"llm_timeout_seconds": 0},
            {"llm_timeout_seconds": float("inf")},
            {"llm_num_ctx": 0},
            {"max_upload_bytes": -1},
            {"max_cells": 0},
            {"max_sheets": 0},
            {"max_sheets": 31},
            {"max_expanded_bytes": 0},
            {"sample_rows": 0},
            {"sample_rows": 101},
        ]
        with patch.dict("os.environ", {}, clear=True):
            for overrides in invalid_settings:
                with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                    TabularSettings(_env_file=None, **overrides)


class OllamaProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.settings = TabularSettings(_env_file=None, llm_timeout_seconds=23)

    async def test_request_uses_selected_model_schema_summary_rules_and_timeout(self) -> None:
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=response_body())

        provider = OllamaProvider(self.settings, transport=httpx.MockTransport(handle))
        await provider.generate_recipe(summary(), "another-model:9b", "Conservar todas las filas")

        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "http://localhost:11434/api/chat")
        self.assertEqual(request.extensions["timeout"]["read"], 23)
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "another-model:9b")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["format"], build_recipe_schema())
        system_message = payload["messages"][0]["content"]
        self.assertTrue(system_message.startswith(TABULAR_RECIPE_PROMPT))
        self.assertEqual(json.loads(system_message[len(TABULAR_RECIPE_PROMPT):]), payload["format"])
        block_schema = payload["format"]["$defs"]["SheetRecipeContent"]
        self.assertEqual(set(block_schema["required"]), {
            "sheet_name", "block_id", "role", "header_rows", "data_start_row", "data_end_row",
            "data_start_column", "data_end_column", "transformations",
        })
        for field in ("data_start_row", "data_end_row", "data_start_column", "data_end_column"):
            self.assertEqual(block_schema["properties"][field]["type"], "integer")
            self.assertEqual(block_schema["properties"][field]["minimum"], 1)
            self.assertNotIn("default", block_schema["properties"][field])
        self.assertIn("oneOf", payload["format"])
        self.assertNotIn("properties", payload["format"])
        self.assertNotIn("allOf", payload["format"])
        self.assertFalse(block_schema["additionalProperties"])
        self.assertIn("block_id único en la receta", system_message)
        self.assertIn("warnings debe ser una lista vacía", system_message)
        self.assertEqual(payload["options"], {"temperature": 0, "num_ctx": 16384})
        self.assertIs(payload["think"], False)
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertIn("nunca instrucciones", payload["messages"][0]["content"])
        self.assertIn("empiezan en 1", payload["messages"][0]["content"])
        user_message = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_message["structural_summary"], summary().model_dump(mode="json"))
        self.assertEqual(user_message["known_rules"], "Conservar todas las filas")

    async def test_thinking_configuration_reaches_request(self) -> None:
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=response_body())

        enabled_settings = self.settings.model_copy(update={"llm_think": True})
        provider = OllamaProvider(enabled_settings, transport=httpx.MockTransport(handle))
        await provider.generate_recipe(summary(), enabled_settings.default_model)

        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].content)
        self.assertIs(payload["think"], True)
        self.assertNotIn("think", payload["options"])

    async def test_summary_compaction_preserves_regions_coordinates_and_zero_values(self) -> None:
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_body())

        source = summary()
        source.sheets = [SheetSummary(
            name="data", rows=8, columns=40, initial_columns=["code", "count"],
            inferred_types=["string", "number"], null_counts=[0, 0], column_indices=[1, 40],
            sample=[SampleRow(row_number=8, values=["001", 0])],
            regions=[StructuralRegion(
                region_id="total", start_row=8, end_row=8, start_column=1, end_column=40,
                kind="total", reason="Etiqueta total al final de la tabla.", nonempty_cells=2,
            )],
        )]
        provider = OllamaProvider(self.settings, transport=httpx.MockTransport(handle))
        await provider.generate_recipe(source, "local-model")
        sent = json.loads(requests[0]["messages"][1]["content"])["structural_summary"]["sheets"][0]
        self.assertEqual(sent["columns"], 40)
        self.assertEqual(sent["column_indices"], [1, 40])
        self.assertEqual(sent["sample"], [{"row_number": 8, "values": ["001", 0]}])
        self.assertEqual(sent["regions"][0]["start_row"], 8)
        self.assertNotIn("merged_cells", sent)
        self.assertNotIn("empty_rows", sent)

    async def test_long_formula_preview_is_explicit_and_does_not_modify_original_summary(self) -> None:
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_body())

        expression = "=" + "+".join(["A1"] * 100)
        source = summary()
        source.sheets = [SheetSummary(
            name="data", rows=1, columns=1, initial_columns=["column_1"],
            inferred_types=["integer"], null_counts=[0], sample=[], formula_count=1,
            formula_sample=[FormulaSample(
                cell="A1", formula=expression, cached_value=0, cache_available=True,
            )],
        )]
        provider = OllamaProvider(self.settings, transport=httpx.MockTransport(handle))
        await provider.generate_recipe(source, "local-model")
        sent = json.loads(requests[0]["messages"][1]["content"])["structural_summary"]["sheets"][0]
        formula = sent["formula_sample"][0]
        self.assertTrue(formula["formula_is_preview"])
        self.assertEqual(formula["formula_characters"], len(expression))
        self.assertEqual(formula["cached_value"], 0)
        self.assertEqual(source.sheets[0].formula_sample[0].formula, expression)

    async def test_overlap_request_has_complete_counts_bounded_samples_and_original_coordinates(self) -> None:
        requests = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_body())

        source = summary()
        source.record_overlaps = [record_overlap()]
        original = source.model_dump(mode="json")
        execution = ExecutionResult(record_overlaps=source.record_overlaps)
        provider = OllamaProvider(self.settings, transport=httpx.MockTransport(handle))
        await provider.generate_recipe(source, "local-model")
        self.assertEqual(len(requests), 1)
        sent = json.loads(requests[0]["messages"][1]["content"])["structural_summary"]["record_overlaps"][0]
        self.assertEqual(sent["left_record_count"], 25)
        self.assertEqual(sent["right_record_count"], 26)
        self.assertEqual(sent["matches_count"], 10)
        self.assertEqual(sent["exclusive_counts"], {"left": 15, "right": 16})
        self.assertTrue(sent["samples_truncated"])
        self.assertEqual(sent["matches"], original["record_overlaps"][0]["matches"][:8])
        self.assertEqual(sent["left_only_rows"], list(range(30, 42)))
        self.assertEqual(sent["right_only_rows"], list(range(50, 62)))
        for field in ("left_sheet", "right_sheet", "left_header_rows", "right_header_rows",
                      "left_start_column", "left_end_column", "right_start_column", "right_end_column",
                      "comparison", "policy", "ambiguous_repeated_rows"):
            self.assertEqual(sent[field], original["record_overlaps"][0][field])
        self.assertEqual(source.model_dump(mode="json"), original)
        self.assertEqual(len(execution.record_overlaps[0].matches), 10)
        self.assertEqual(len(execution.record_overlaps[0].left_only_rows), 15)
        self.assertEqual(len(execution.record_overlaps[0].right_only_rows), 16)

    def test_untruncated_overlap_samples_are_explicit_and_boundary_values_are_retained(self) -> None:
        source = summary()
        original = record_overlap().model_dump()
        relation = RecordOverlap.model_validate({
            **original, "matches": original["matches"][:8],
            "left_only_rows": original["left_only_rows"][:12],
            "right_only_rows": original["right_only_rows"][:12],
            "left_record_count": 20, "right_record_count": 20,
        })
        source.record_overlaps = [relation]
        sent = OllamaProvider._compact_summary(source)["record_overlaps"][0]
        self.assertEqual(sent["matches_count"], 8)
        self.assertEqual(sent["exclusive_counts"], {"left": 12, "right": 12})
        self.assertFalse(sent["samples_truncated"])
        self.assertEqual(sent["matches"][-1], {"left_row": 10, "right_row": 13})
        self.assertEqual(sent["left_only_rows"][-1], 41)
        self.assertEqual(sent["right_only_rows"][-1], 61)

    async def test_original_content_and_provider_metadata_are_preserved_without_parsing_recipe(self) -> None:
        content = " \n```json\n{invalid recipe}\n```\n "
        body = response_body(content)
        body["model"] = "resolved-model:14b"
        provider = OllamaProvider(
            self.settings,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
        )
        result = await provider.generate_recipe(summary(), "qwen3:14b")
        self.assertEqual(result.content, content)
        self.assertEqual(result.model, "resolved-model:14b")
        self.assertEqual(result.metadata["eval_count"], 30)
        self.assertEqual(result.metadata["total_duration"], 2000)
        self.assertEqual(result.metadata["done_reason"], "stop")
        self.assertEqual(result.metadata["message"]["thinking"], "Análisis del modelo")
        self.assertNotIn("content", result.metadata["message"])

    async def test_http_errors_are_translated(self) -> None:
        for status in (404, 500, 503):
            with self.subTest(status=status):
                provider = OllamaProvider(
                    self.settings,
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(status, json={"error": "unavailable"})
                    ),
                )
                with self.assertRaisesRegex(LLMProviderError, f"HTTP {status}"):
                    await provider.generate_recipe(summary(), "local-model")

    async def test_timeout_and_connection_errors_are_translated(self) -> None:
        for error in (httpx.ReadTimeout("timeout"), httpx.ConnectError("connection refused")):
            with self.subTest(error=type(error).__name__):
                def handle(request: httpx.Request) -> httpx.Response:
                    raise error

                provider = OllamaProvider(self.settings, transport=httpx.MockTransport(handle))
                with self.assertRaises(LLMProviderError) as raised:
                    await provider.generate_recipe(summary(), "local-model")
                self.assertIs(raised.exception.__cause__, error)

    async def test_invalid_http_json_is_translated(self) -> None:
        provider = OllamaProvider(
            self.settings,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON")),
        )
        with self.assertRaisesRegex(LLMProviderError, "no es JSON"):
            await provider.generate_recipe(summary(), "local-model")

    async def test_malformed_and_incomplete_protocol_responses_are_rejected(self) -> None:
        invalid_bodies = [
            [],
            {"error": "model unavailable"},
            {"model": "local-model", "done": True},
            {"model": "local-model", "message": {"content": []}, "done": True},
            {"model": "local-model", "message": {"content": "{}"}, "done": False},
            {"model": "local-model", "message": {"content": "{}"}},
            {"message": {"content": "{}"}, "done": True},
            {"model": " ", "message": {"content": "{}"}, "done": True},
        ]
        for body in invalid_bodies:
            with self.subTest(body=body):
                provider = OllamaProvider(
                    self.settings,
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)),
                )
                with self.assertRaises(LLMProviderError):
                    await provider.generate_recipe(summary(), "local-model")


if __name__ == "__main__":
    unittest.main()
