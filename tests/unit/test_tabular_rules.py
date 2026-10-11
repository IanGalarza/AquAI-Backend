import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from aquai.application.dtos.tabular import PreparationRecipe, SheetSummary, StructuralSummary
from aquai.application.errors import TabularInputError
from aquai.application.services.tabular_rules import TabularRuleService
from aquai.application.use_cases.evaluate_tabular_file import EvaluateTabularFile
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.llm.ollama import OllamaProvider
from aquai.infrastructure.providers.tabular import get_tabular_evaluator, get_tabular_settings
from aquai.infrastructure.providers.tabular_rules import FileTabularRuleCatalog
from aquai.main import create_app


def structural_summary(
    file_name: str = "Cyanobacterial_Alert_Levels.csv",
    sheets: tuple[str, ...] = ("data",),
) -> StructuralSummary:
    return StructuralSummary(
        file_name=file_name,
        file_type="csv" if file_name.endswith(".csv") else "xlsx",
        file_size_bytes=100,
        sheets=[SheetSummary(
            name=name,
            rows=3,
            columns=2,
            initial_columns=["sample_id", "alert_class"],
            inferred_types=["text", "text"],
            null_counts=[0, 0],
            sample=[],
        ) for name in sheets],
    )


class TabularRuleServiceTests(unittest.TestCase):
    reviewed_file = "Tabla completa ciano RdlP.xlsx"
    reviewed_sha256 = "79770bdd190fcd08dd8351e18f2f1450d0b250bdeb8b9588e5129492a82cc025"

    def setUp(self) -> None:
        self.service = TabularRuleService(FileTabularRuleCatalog())

    def test_common_rules_apply_without_profile(self) -> None:
        composition = self.service.compose(structural_summary())
        self.assertEqual(composition.metadata["version"], "1.0.0")
        self.assertIsNone(composition.metadata["profile"])
        self.assertTrue(composition.prompt_rules["common"])
        self.assertEqual(composition.prompt_rules["selected_profile"], [])
        self.assertEqual(composition.review_reasons, [])

    def test_only_selected_profile_is_sent_with_origins_and_versions_preserved(self) -> None:
        composition = self.service.compose(
            structural_summary(), "cyanobacterial_alerts", "Conservar las columnas originales."
        )
        prompt = json.dumps(composition.prompt_rules, ensure_ascii=False)
        self.assertIn("alert_sample_identity", prompt)
        self.assertNotIn("belen_abundance_modalities", prompt)
        self.assertNotIn("biological_subset_overlap", prompt)
        self.assertNotIn("sampling_campaign_periods", prompt)
        self.assertEqual(composition.prompt_rules["additional"], "Conservar las columnas originales.")
        self.assertEqual(composition.metadata["profile"], "cyanobacterial_alerts")
        self.assertTrue(all(
            rule["version"] == "1.0.0" and rule["origin"]
            for rule in composition.metadata["common_rules"] + composition.metadata["profile_rules"]
        ))

    def test_unknown_profile_fails_before_inspection(self) -> None:
        with self.assertRaisesRegex(TabularInputError, "desconocido"):
            self.service.validate_profile("not_a_profile")

    def test_profile_does_not_assume_file_family(self) -> None:
        composition = self.service.compose(structural_summary("other.csv"), "biological_counts")
        self.assertFalse(composition.metadata["evidence"]["file_name_matches"])
        self.assertEqual(len(composition.metadata["evidence"]["missing_expected_sheets"]), 4)
        self.assertTrue(composition.review_reasons)
        self.assertEqual(composition.findings[0]["code"], "rule_profile_unverified")

    def test_current_evidence_corrects_prior_merged_cell_observation(self) -> None:
        composition = self.service.compose(
            structural_summary(
                "datos biologicos tesis.XLSX", ("Recuentos", "Cianobacterias", "Hoja3", "Hoja1")
            ),
            "biological_counts",
        )
        self.assertEqual(composition.metadata["evidence"]["hoja3_merged_observation"], "not_observed")
        self.assertEqual(composition.review_reasons, [])
        self.assertEqual(composition.findings[0]["sheet_name"], "Hoja3")
        self.assertFalse(composition.findings[0]["requires_review"])

    def test_additional_conflict_is_preserved_and_requires_review(self) -> None:
        additional = "Reemplazar nd por cero; eliminar los ceros."
        composition = self.service.compose(structural_summary(), additional=additional)
        self.assertEqual(composition.metadata["additional_rules"], additional)
        self.assertEqual(composition.prompt_rules["additional"], additional)
        self.assertEqual(len(composition.review_reasons), 1)
        self.assertEqual(composition.findings[0]["code"], "rule_additional_conflict")

    def test_negative_protection_does_not_create_false_conflict(self) -> None:
        composition = self.service.compose(
            structural_summary(), additional="No convertir nd a cero; no eliminar los ceros."
        )
        self.assertEqual(composition.review_reasons, [])

    def test_catalog_records_verified_documentary_source(self) -> None:
        composition = self.service.compose(structural_summary())
        notes = composition.metadata["origins"]["researcher_notes"]
        self.assertEqual(notes["pages"], [1, 2])
        self.assertEqual(notes["verification"], "verified_document")
        self.assertEqual(len(notes["sha256"]), 64)

    def test_reviewed_file_confirmation_is_auditable_and_preserves_campaign_guidance(self) -> None:
        composition = self.service.compose(
            structural_summary(self.reviewed_file), "self_sampling", source_sha256=self.reviewed_sha256,
        )
        self.assertEqual(len(composition.identity_clarifications), 1)
        clarification = composition.identity_clarifications[0]
        self.assertEqual(clarification.original_note, "No se si habrá alguna repetida con las 13 y 14 PL y Bal")
        self.assertEqual(clarification.confirmed_by, "usuario")
        self.assertIn("2026-10-11", clarification.source_reference)
        applied = composition.metadata["file_specific_rules"]
        self.assertEqual(len(applied), 1)
        self.assertEqual(applied[0]["id"], clarification.rule_id)
        self.assertEqual(applied[0]["sha256"], self.reviewed_sha256)
        self.assertEqual(applied[0]["file_name"], self.reviewed_file)
        self.assertEqual(applied[0]["scope"], "file_specific")
        self.assertEqual(composition.prompt_rules["file_specific_rules"], applied)
        for text in (applied[0]["text"], clarification.statement):
            self.assertIn("conservarlos como observaciones independientes", text)
            self.assertIn("períodos diferentes según las leyendas y el PDF de anotaciones", text)
            self.assertIn("sólo", text)
            self.assertIn("copias coincidentes entre hojas", text)
        origin = composition.metadata["origins"][applied[0]["origin"]]
        self.assertEqual(origin["verification"], "user_confirmation")
        self.assertEqual(origin["date"], "2026-10-11")
        self.assertEqual(composition.metadata["version"], "1.0.0")
        campaign = next(rule for rule in composition.metadata["profile_rules"]
                        if rule["id"] == "sampling_campaign_periods")
        self.assertEqual(campaign["version"], "1.0.0")
        self.assertIn("1 = noviembre de 2005", campaign["text"])
        self.assertIn("2 = marzo-abril de 2006", campaign["text"])
        self.assertIn("leyenda", campaign["text"])
        self.assertEqual(composition.metadata["origins"]["researcher_notes"]["pages"], [1, 2])
        overlap = next(rule for rule in composition.metadata["profile_rules"]
                       if rule["id"] == "sampling_workbook_overlap")
        self.assertIn("comprobar el solapamiento", overlap["text"])
        self.assertIn("no deduplicar por semejanza", overlap["text"])

    def test_file_confirmation_accepts_only_casefolded_exact_basename(self) -> None:
        for name in (self.reviewed_file.upper(), f"uploads/{self.reviewed_file}",
                     f"uploads\\{self.reviewed_file}"):
            with self.subTest(name=name):
                composition = self.service.compose(
                    structural_summary(name), "self_sampling", source_sha256=self.reviewed_sha256,
                )
                self.assertEqual(len(composition.identity_clarifications), 1)

    def test_confirmation_requires_matching_profile_name_and_real_content_digest(self) -> None:
        origin = FileTabularRuleCatalog().load()["file_specific_rules"][0]["origin"]
        for name, profile, digest in (
            (self.reviewed_file, "self_sampling", None),
            (self.reviewed_file, "self_sampling", "0" * 64),
            ("another.xlsx", "self_sampling", self.reviewed_sha256),
            ("Tabla completa ciano RdlṔ.xlsx", "self_sampling", self.reviewed_sha256),
            (f" {self.reviewed_file}", "self_sampling", self.reviewed_sha256),
            (self.reviewed_file, None, self.reviewed_sha256),
            (self.reviewed_file, "biological_counts", self.reviewed_sha256),
        ):
            with self.subTest(name=name, profile=profile, digest=digest):
                composition = self.service.compose(
                    structural_summary(name), profile, source_sha256=digest,
                )
                self.assertEqual(composition.identity_clarifications, ())
                self.assertNotIn("file_specific_rules", composition.metadata)
                self.assertNotIn("file_specific_rules", composition.prompt_rules)
                self.assertNotIn(origin, composition.metadata["origins"])
        composition = self.service.compose(
            structural_summary(self.reviewed_file), "self_sampling",
            additional=f"source_sha256={self.reviewed_sha256}; muestras confirmadas distintas",
        )
        self.assertEqual(composition.identity_clarifications, ())

    def test_specific_rule_keeps_its_own_version_and_uses_catalog_scope(self) -> None:
        catalog = deepcopy(FileTabularRuleCatalog().load())
        rule = catalog["file_specific_rules"][0]
        rule.update({"version": "2.3.4", "profile": "cyanobacterial_alerts",
                     "file_name": "OtherAlerts.csv", "sha256": "f" * 64})
        with patch.object(FileTabularRuleCatalog, "load", return_value=catalog):
            service = TabularRuleService(FileTabularRuleCatalog())
        composition = service.compose(
            structural_summary("OtherAlerts.csv"), "cyanobacterial_alerts", source_sha256="f" * 64,
        )
        self.assertEqual(len(composition.identity_clarifications), 1)
        self.assertEqual(composition.metadata["file_specific_rules"][0]["version"], "2.3.4")
        self.assertEqual(composition.metadata["version"], "1.0.0")
        self.assertTrue(all(rule["version"] == "1.0.0" for rule in composition.metadata["profile_rules"]))

    def test_unmatched_confirmation_preserves_existing_profile_compositions(self) -> None:
        catalog = deepcopy(FileTabularRuleCatalog().load())
        catalog.pop("file_specific_rules")
        with patch.object(FileTabularRuleCatalog, "load", return_value=catalog):
            baseline = TabularRuleService(FileTabularRuleCatalog())
        for name, profile, sheets in (
            ("datos biologicos tesis.xlsx", "biological_counts", ("Recuentos", "Cianobacterias", "Hoja3", "Hoja1")),
            (self.reviewed_file, "self_sampling", ("data",)),
            ("other.xlsx", None, ("data",)),
        ):
            with self.subTest(profile=profile):
                source = structural_summary(name, sheets)
                expected = baseline.compose(source, profile, source_sha256="0" * 64)
                actual = self.service.compose(source, profile, source_sha256="0" * 64)
                self.assertEqual(actual.metadata, expected.metadata)
                self.assertEqual(actual.prompt_rules, expected.prompt_rules)
                self.assertEqual(actual.findings, expected.findings)
                self.assertEqual(actual.review_reasons, expected.review_reasons)
                self.assertEqual(actual.identity_clarifications, ())


class TabularRuleEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[dict] = []
        settings = TabularSettings(_env_file=None)
        self.recipe_content = {
            "status": "ready",
            "sheets": [{
                "sheet_name": "data", "header_rows": [1], "data_start_row": 2,
                "block_id": "data", "role": "observations", "data_end_row": 2,
                "data_start_column": 1, "data_end_column": 2,
                "transformations": [],
            }],
            "warnings": [],
            "requires_human_review": False,
        }

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "model": settings.default_model,
                "done": True,
                "message": {"role": "assistant", "content": json.dumps(self.recipe_content)},
            })

        self.processor = PandasTabularProcessor()
        evaluator = EvaluateTabularFile(
            self.processor,
            OllamaProvider(settings, transport=httpx.MockTransport(respond)),
            default_model=settings.default_model,
            max_upload_bytes=settings.max_upload_bytes,
            rule_service=TabularRuleService(FileTabularRuleCatalog()),
        )
        app = create_app()
        app.dependency_overrides[get_tabular_evaluator] = lambda: evaluator
        app.dependency_overrides[get_tabular_settings] = lambda: settings
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def upload(self, content: bytes = b"sample_id,alert_class\nsample1,medium\n", **kwargs):
        return self.client.post(
            "/experimental/tabular/evaluate",
            files={"file": (
                "Cyanobacterial_Alert_Levels.csv", content
            )},
            **kwargs,
        )

    def test_profile_and_additional_rules_reach_provider_and_response(self) -> None:
        response = self.upload(data={
            "rule_profile": "cyanobacterial_alerts",
            "known_rules": "Conservar el orden original.",
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["applied_rules"]["profile"], "cyanobacterial_alerts")
        user_message = json.loads(self.requests[0]["messages"][1]["content"])
        rules = json.loads(user_message["known_rules"])
        self.assertEqual(rules["profile"], "cyanobacterial_alerts")
        self.assertEqual(rules["additional"], "Conservar el orden original.")
        self.assertNotIn("biological_subset_overlap", json.dumps(rules))

    def test_unknown_profile_returns_400_without_inspection_or_provider_call(self) -> None:
        with patch.object(self.processor, "inspect", wraps=self.processor.inspect) as inspect:
            response = self.upload(data={"rule_profile": "unknown"})
        self.assertEqual(response.status_code, 400)
        inspect.assert_not_called()
        self.assertEqual(self.requests, [])

    def test_conflicting_rules_require_review_even_if_model_says_ready(self) -> None:
        response = self.upload(data={"known_rules": "Reemplazar nd por cero."})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["validation"]["status"], "review_required")
        self.assertTrue(any(
            item["code"] == "rule_additional_conflict"
            for item in response.json()["validation"]["findings"]
        ))

    def test_alert_category_conversion_requires_review_even_if_model_says_ready(self) -> None:
        self.recipe_content["sheets"][0]["transformations"] = [{
            "operation": "convert_type", "columns": ["alert_class"], "target_type": "boolean",
        }]
        response = self.upload(
            b"sample_id,alert_class\n001,0\n002,1\n",
            data={"rule_profile": "cyanobacterial_alerts"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["validation"]["status"], "review_required")
        self.assertTrue(any(
            item["code"] == "rule_profile_categorical_conversion"
            for item in body["validation"]["findings"]
        ))


class TabularRuleRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TabularRuleService(FileTabularRuleCatalog())

    def findings(self, source: StructuralSummary, profile: str | None, **block_changes):
        block = {
            "sheet_name": source.sheets[0].name,
            "block_id": "data",
            "role": "observations",
            "header_rows": [1],
            "data_start_row": 2,
            "data_end_row": source.sheets[0].rows,
            "data_start_column": 1,
            "data_end_column": source.sheets[0].columns,
            "transformations": [],
            **block_changes,
        }
        recipe = PreparationRecipe.model_validate({
            "status": "ready", "sheets": [block], "warnings": [],
            "requires_human_review": False,
        })
        composition = self.service.compose(source, profile)
        return self.service.validate_recipe(source, recipe, composition)

    def test_binary_alert_labels_cannot_be_reinterpreted_as_booleans(self) -> None:
        findings = self.findings(structural_summary(), "cyanobacterial_alerts", transformations=[{
            "operation": "convert_type", "columns": ["alert_class"], "target_type": "boolean",
        }])
        self.assertEqual(findings[0]["code"], "rule_profile_categorical_conversion")
        self.assertTrue(findings[0]["requires_review"])
        self.assertEqual(findings[0]["range"], "A2:B3")

    def test_sample_identifiers_are_preserved_as_original_codes(self) -> None:
        findings = self.findings(structural_summary(), "cyanobacterial_alerts", transformations=[{
            "operation": "convert_type", "columns": ["sample_id"], "target_type": "integer",
        }])
        self.assertEqual(findings[0]["code"], "rule_profile_categorical_conversion")

    def test_compacted_headers_cannot_bypass_named_profile_protection(self) -> None:
        source = structural_summary()
        source.sheets[0].initial_columns = ["sample_id", "cell_count"]
        source.sheets[0].columns = 20
        findings = self.findings(source, "cyanobacterial_alerts", transformations=[{
            "operation": "convert_type", "columns": ["alert_class"], "target_type": "boolean",
        }])
        self.assertEqual(findings[0]["code"], "rule_profile_categorical_conversion")

    def test_renamed_alert_column_remains_protected(self) -> None:
        findings = self.findings(structural_summary(), "cyanobacterial_alerts", transformations=[
            {"operation": "rename_columns", "mapping": {"alert_class": "alert"}},
            {"operation": "convert_type", "columns": ["alert"], "target_type": "boolean"},
        ])
        self.assertEqual([item["code"] for item in findings], [
            "rule_profile_columns_changed", "rule_profile_categorical_conversion",
        ])

    def test_other_numeric_measurements_remain_convertible(self) -> None:
        source = structural_summary()
        source.sheets[0].initial_columns.append("cell_count")
        source.sheets[0].columns = 3
        findings = self.findings(source, "cyanobacterial_alerts", transformations=[{
            "operation": "convert_type", "columns": ["cell_count"], "target_type": "integer",
        }])
        self.assertEqual(findings, [])

    def test_alert_restriction_is_not_a_universal_rule(self) -> None:
        findings = self.findings(structural_summary(), None, transformations=[{
            "operation": "convert_type", "columns": ["alert_class"], "target_type": "boolean",
        }])
        self.assertEqual(findings, [])

    def test_biological_presence_summary_cannot_be_classified_as_abundance(self) -> None:
        source = structural_summary(
            "datos biologicos tesis.xlsx", ("Hoja1", "Recuentos", "Cianobacterias", "Hoja3")
        )
        source.sheets[0].initial_columns = ["taxon", "presencias"]
        findings = self.findings(source, "biological_counts", role="observations")
        self.assertEqual(findings[0]["code"], "rule_profile_role_conflict")
        self.assertEqual(self.findings(source, "biological_counts", role="summary"), [])

    def test_biological_role_is_not_assumed_from_sheet_name_alone(self) -> None:
        source = structural_summary(
            "datos biologicos tesis.xlsx", ("Hoja1", "Recuentos", "Cianobacterias", "Hoja3")
        )
        self.assertEqual(self.findings(source, "biological_counts", role="observations"), [])
