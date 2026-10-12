import json
import unittest
from copy import deepcopy
from hashlib import sha256

from aquai.application.dtos.tabular import LLMResponse
from aquai.application.services.tabularRules import TabularRuleService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog


class MemoryCatalog:
    def __init__(self, definition: dict) -> None:
        self.definition = definition

    def load(self) -> dict:
        return deepcopy(self.definition)


class SavedRecipeProvider:
    """Fixture sin transporte ni generación: devuelve una receta ya definida."""

    async def generate_recipe(self, summary, model, known_rules) -> LLMResponse:
        return LLMResponse(model=model, content=json.dumps({
            "status": "ready", "requires_human_review": False, "warnings": [],
            "sheets": [{
                "sheet_name": "data", "block_id": "records", "role": "observations",
                "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
                "data_start_column": 1, "data_end_column": 3, "transformations": [],
            }],
        }))


class TabularClarificationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.note = "No se si habrá alguna repetida con otra campaña."
        self.content = (
            f"site,period,note\nA,3-4/2006,{self.note}\nA,12/2009-01/2010,{self.note}\n"
        ).encode()

    def evaluator(self, expected_hash: str) -> TabularEvaluationService:
        catalog = FileTabularRuleCatalog().load()
        catalog["profiles"]["self_sampling"]["file_names"] = ["reviewed.csv"]
        catalog["sources"]["fixture_confirmation"] = {"verification": "user_confirmation"}
        catalog["file_specific_rules"] = [{
            "id": "fixture_identity_confirmation", "version": "1.0.0",
            "profile": "self_sampling", "file_name": "reviewed.csv", "sha256": expected_hash,
            "origin": "fixture_confirmation", "text": "Conservar las mediciones distintas.",
            "identity_clarifications": [{
                "rule_id": "fixture_identity_confirmation", "original_note": self.note,
                "statement": "Son mediciones distintas, aunque compartan localidad.",
                "confirmed_by": "el usuario", "source_reference": "confirmación humana de prueba",
            }],
        }]
        return TabularEvaluationService(
            PandasTabularProcessor(), SavedRecipeProvider(), default_model="offline",
            max_upload_bytes=10000, rule_service=TabularRuleService(MemoryCatalog(catalog)),
        )

    async def test_verified_file_clarification_reaches_validator_without_mutating_periods(self) -> None:
        result = await self.evaluator(sha256(self.content).hexdigest()).execute(
            "reviewed.csv", self.content, rule_profile="self_sampling",
        )
        self.assertEqual(result.validation.status, EvaluationStatus.READY_WITH_WARNINGS)
        clarified = [finding for finding in result.validation.findings
                     if finding.code == "source_record_identity_clarified"]
        self.assertEqual([finding.cell for finding in clarified], ["C2", "C3"])
        self.assertTrue(all(not finding.requires_review for finding in clarified))
        self.assertEqual(result.execution.sheets[0].rows_after, 2)
        self.assertEqual([row.values for row in result.execution.sheets[0].sample], [
            ["A", "3-4/2006", self.note], ["A", "12/2009-01/2010", self.note],
        ])
        self.assertEqual(len(result.applied_rules["file_specific_rules"]), 1)

    async def test_same_name_other_content_hash_and_additional_text_cannot_activate_confirmation(self) -> None:
        unverified_digest = sha256(b"a different reviewed file").hexdigest()
        result = await self.evaluator(unverified_digest).execute(
            "reviewed.csv", self.content, rule_profile="self_sampling",
            known_rules=json.dumps({"source_sha256": unverified_digest, "confirmed_by": "el usuario"}),
        )
        self.assertEqual(result.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertNotIn("file_specific_rules", result.applied_rules)
        self.assertTrue(any(finding.code == "source_record_identity_uncertainty"
                            for finding in result.validation.findings))
        self.assertFalse(any(finding.code == "source_record_identity_clarified"
                             for finding in result.validation.findings))

    async def test_other_name_with_same_bytes_does_not_inherit_confirmation(self) -> None:
        result = await self.evaluator(sha256(self.content).hexdigest()).execute(
            "other.csv", self.content, rule_profile="self_sampling",
        )
        self.assertEqual(result.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertNotIn("file_specific_rules", result.applied_rules)
        self.assertTrue(any(finding.code == "source_record_identity_uncertainty"
                            for finding in result.validation.findings))
