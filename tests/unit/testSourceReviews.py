import json
import asyncio
import unittest
from copy import deepcopy
from hashlib import sha256
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from threading import Event

from jsonschema import Draft202012Validator

from aquai.application.contracts.tabular.results import LLMResponse, TabularFinding
from aquai.application.errors import SourceDecisionError
from aquai.application.services.sourceReviews import SourceReviewService
from aquai.application.services.tabularRules import RuleComposition, TabularRuleService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog


class SourcePipelineReviewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.content = b"name,score\nA,not-a-number\n"
        self.document = {
            "status": "ready", "requires_human_review": False, "warnings": [],
            "sheets": [{
                "sheet_name": "data", "block_id": "records", "role": "observations",
                "header_rows": [1], "data_start_row": 2, "data_end_row": 2,
                "data_start_column": 1, "data_end_column": 2,
                "transformations": [{"operation": "convert_type", "columns": ["score"], "target_type": "integer"}],
            }],
        }
        self.provider = Mock()
        self.provider.generate_recipe = AsyncMock(return_value=LLMResponse(
            model="offline", content=json.dumps(self.document),
        ))
        self.processor = Mock(wraps=PandasTabularProcessor())
        self.evaluator = TabularEvaluationService(
            self.processor, self.provider, default_model="offline", max_upload_bytes=10000,
            rule_service=TabularRuleService(FileTabularRuleCatalog()),
        )
        self.snapshots = []
        self.reviews = SourceReviewService()
        self.source_id = uuid4()

    async def checkpoint(self, stage, result, rules):
        self.snapshots.append((stage, result.model_copy(deep=True) if result else None, deepcopy(rules)))

    async def initial(self):
        result = await self.evaluator.process_source("data.csv", self.content, checkpoint=self.checkpoint)
        return result, self.snapshots[-1][2]

    async def test_recipe_is_checkpointed_before_execution_and_rules_round_trip(self):
        result, snapshot = await self.initial()
        stages = [stage for stage, _, _ in self.snapshots]
        self.assertEqual(stages, [
            "inspecting", "inspected", "generating_recipe", "recipe_generated", "executing", "executed", "validating", "completed",
        ])
        archived = self.snapshots[stages.index("recipe_generated")][1]
        self.assertEqual(archived.recipe.model_dump(), self.document)
        self.assertEqual(archived.execution.sheets, [])
        self.assertIsNone(snapshot["metadata"]["profile"])
        self.assertTrue(snapshot["prompt_rules"]["common"])
        self.assertEqual(RuleComposition.restore(json.loads(json.dumps(snapshot))).snapshot(), snapshot)
        self.assertEqual(result.validation.status, EvaluationStatus.REVIEW_REQUIRED)

    async def test_correcting_transformations_reexecutes_without_inspection_or_llm(self):
        result, snapshot = await self.initial()
        review = next(review for review in self.reviews.build_reviews(self.source_id, result, snapshot)
                      if review["code"] == "conversion_failures")
        Draft202012Validator(review["options"][0]["values"]).validate({"transformations": []})
        corrected, rules, restart = self.reviews.apply(
            review, "correct_transformations", {"transformations": []}, result, snapshot,
        )
        resumed = await self.evaluator.process_source(
            "data.csv", self.content, checkpoint=self.checkpoint,
            saved_evaluation=corrected, saved_rules=rules, resume_stage=restart,
        )
        self.provider.generate_recipe.assert_awaited_once()
        self.processor.inspect.assert_called_once()
        self.assertEqual(self.processor.execute.call_count, 2)
        self.assertIn(resumed.validation.status, {EvaluationStatus.READY, EvaluationStatus.READY_WITH_WARNINGS})
        self.assertEqual(resumed.execution.sheets[0].sample[0].values, ["A", "not-a-number"])
        self.assertEqual(json.loads(resumed.llm_response.content), self.document)
        self.assertEqual(self.snapshots[-1][2]["source_decisions"], rules["source_decisions"])
        self.assertEqual(result.recipe.sheets[0].transformations[0].operation, "convert_type")
        self.assertEqual(review["range"], "A2:B2")

    async def test_identity_clarification_revalidates_only_affected_note(self):
        note = "No se si habrá alguna repetida con otra campaña."
        self.content = f"name,note\nA,{note}\nB,{note}\n".encode()
        self.document["sheets"][0].update(data_end_row=3, transformations=[])
        self.provider.generate_recipe.return_value.content = json.dumps(self.document)
        result, snapshot = await self.initial()
        reviews = self.reviews.build_reviews(self.source_id, result, snapshot)
        review = next(review for review in reviews if review["code"] == "source_record_identity_uncertainty" and review["cell"] == "B2")
        Draft202012Validator(review["options"][0]["values"]).validate({"identity": "distinct", "statement": "Son registros distintos."})
        corrected, rules, restart = self.reviews.apply(
            review, "clarify_record_identity", {"identity": "distinct", "statement": "Son registros distintos."}, result, snapshot,
        )
        self.assertEqual(restart, "validation")
        resumed = await self.evaluator.process_source(
            "data.csv", self.content, checkpoint=self.checkpoint,
            saved_evaluation=corrected, saved_rules=rules, resume_stage=restart,
        )
        self.provider.generate_recipe.assert_awaited_once()
        self.processor.inspect.assert_called_once()
        self.processor.execute.assert_called_once()
        remaining = [finding.cell for finding in resumed.validation.findings if finding.code == "source_record_identity_uncertainty"]
        self.assertEqual(remaining, ["B3"])
        self.assertEqual(resumed.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual(resumed.validation.original_proposal, result.validation.original_proposal)
        clarification = rules["identity_clarifications"][-1]
        self.assertEqual((clarification["sheet_name"], clarification["cell"]), ("data", "B2"))

    async def test_review_ids_are_stable_for_reads_and_change_with_revision(self):
        result, snapshot = await self.initial()
        first = self.reviews.build_reviews(self.source_id, result, snapshot)
        self.assertEqual(first, self.reviews.build_reviews(self.source_id, result, snapshot))
        changed = self.reviews.build_reviews(self.source_id, result, snapshot, revision=1)
        self.assertNotEqual(first[0]["id"], changed[0]["id"])

    async def test_unsupported_errors_are_not_offered_as_user_questions(self):
        result, snapshot = await self.initial()
        result.validation.findings = [TabularFinding(
            code="source_missing_formula_cache", message="Resultado de fórmula ausente.",
            sheet_name="data", cell="B2", severity="error",
        )]
        result.validation.errors = ["Error técnico de lectura."]
        self.assertEqual(self.reviews.build_reviews(self.source_id, result, snapshot), [])
        self.assertTrue(self.reviews.is_technical_finding(result.validation.findings[0]))

    async def test_unsupported_uncertainty_keeps_stable_review_without_unsafe_waiver(self):
        result, snapshot = await self.initial()
        result.validation.findings = [TabularFinding(
            code="potential_record_overlap", message="Contrastar identidades entre hojas.", sheet_name="data",
        )]
        reviews = self.reviews.build_reviews(self.source_id, result, snapshot)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["options"], [])
        self.assertTrue(reviews[0]["blocked_reason"])
        with self.assertRaises(SourceDecisionError):
            self.reviews.apply(reviews[0], "approve", {}, result, snapshot)

    async def test_cancellation_drains_cpu_stage_and_preserves_cancellation_on_processor_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                started, finish = Event(), Event()

                def work():
                    started.set()
                    finish.wait(3)
                    if fail:
                        raise RuntimeError("Concurrent processor error")
                    return "done"

                task = asyncio.create_task(self.evaluator._run_processor(work))
                try:
                    while not started.is_set():
                        await asyncio.sleep(0.001)
                    task.cancel()
                    await asyncio.sleep(0.01)
                    self.assertFalse(task.done())
                    task.cancel()
                    await asyncio.sleep(0.01)
                    self.assertFalse(task.done())
                finally:
                    finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_decision_validates_action_values_and_current_finding(self):
        result, snapshot = await self.initial()
        review = self.reviews.build_reviews(self.source_id, result, snapshot)[0]
        for action, values in (
            ("approve", {}),
            ("correct_transformations", {"transformations": [], "waive": True}),
            ("correct_transformations", {"transformations": [{"operation": "drop_all_data"}]}),
            ("correct_transformations", {"transformations": self.document["sheets"][0]["transformations"]}),
        ):
            with self.subTest(action=action, values=values), self.assertRaises(SourceDecisionError):
                self.reviews.apply(review, action, values, result, snapshot)
        result.validation.findings = []
        with self.assertRaisesRegex(SourceDecisionError, "vigente"):
            self.reviews.apply(review, "correct_transformations", {"transformations": []}, result, snapshot)

    async def test_block_decision_cannot_change_other_block_or_escape_original_sheet(self):
        result, snapshot = await self.initial()
        result.validation.findings = [TabularFinding(
            code="unresolved_block_role", message="Revisar rol.", sheet_name="data", block_id="records",
        )]
        review = self.reviews.build_reviews(self.source_id, result, snapshot)[0]
        for update in ({"sheet_name": "other"}, {"block_id": "other"}, {"data_end_row": 100}):
            block = {**self.document["sheets"][0], **update}
            with self.subTest(update=update), self.assertRaises(SourceDecisionError):
                self.reviews.apply(review, "replace_block", {"block": block}, result, snapshot)
        schema = review["options"][0]["values"]
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate({"block": self.document["sheets"][0]})

    async def test_adding_scoped_block_preserves_uncovered_values_and_revalidates(self):
        self.content = b"name,score\nA,not-a-number\nB,4\n"
        self.document["sheets"][0]["transformations"] = []
        self.provider.generate_recipe.return_value.content = json.dumps(self.document)
        result, snapshot = await self.initial()
        review = next(item for item in self.reviews.build_reviews(self.source_id, result, snapshot)
                      if item["code"] == "uncovered_source_cells")
        block = {**self.document["sheets"][0], "block_id": "tail", "data_start_row": 3, "data_end_row": 3}
        schema = review["options"][0]["values"]
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate({"block": block})
        corrected, snapshot, stage = self.reviews.apply(review, "add_block", {"block": block}, result, snapshot)
        resumed = await self.evaluator.process_source(
            "data.csv", self.content, checkpoint=self.checkpoint,
            saved_evaluation=corrected, saved_rules=snapshot, resume_stage=stage,
        )
        self.assertNotEqual(resumed.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual([sheet.block_id for sheet in resumed.execution.sheets], ["records", "tail"])
        self.assertEqual(resumed.execution.sheets[1].sample[0].values, ["B", "4"])
        self.provider.generate_recipe.assert_awaited_once()

    async def test_resuming_rebuilds_recipe_rule_findings_from_effective_recipe(self):
        catalog = FileTabularRuleCatalog().load()
        catalog["profiles"]["cyanobacterial_alerts"]["file_names"] = ["data.csv"]
        catalog["file_specific_rules"] = [{
            "id": "exact_source", "version": "1", "profile": "cyanobacterial_alerts",
            "file_name": "data.csv", "sha256": sha256(self.content).hexdigest(),
            "origin": "corrected_guidance", "text": "Conservar columnas originales.",
        }]
        self.document["sheets"][0]["transformations"] = [{"operation": "rename_columns", "mapping": {"score": "value"}}]
        self.provider.generate_recipe.return_value.content = json.dumps(self.document)
        self.evaluator._rule_service = TabularRuleService(Mock(load=Mock(return_value=catalog)))
        result, snapshot = await self.initial()
        self.assertEqual(snapshot["metadata"]["profile"], "cyanobacterial_alerts")
        self.assertTrue(any(finding.code == "rule_profile_columns_changed" for finding in result.validation.findings))
        review = next(item for item in self.reviews.build_reviews(self.source_id, result, snapshot)
                      if item["code"] == "rule_profile_columns_changed")
        corrected, snapshot, stage = self.reviews.apply(review, "correct_transformations", {"transformations": []}, result, snapshot)
        resumed = await self.evaluator.process_source(
            "data.csv", self.content, checkpoint=self.checkpoint,
            saved_evaluation=corrected, saved_rules=snapshot, resume_stage=stage,
        )
        self.assertFalse(any(finding.code == "rule_profile_columns_changed" for finding in resumed.validation.findings))
        self.assertNotEqual(resumed.validation.status, EvaluationStatus.REVIEW_REQUIRED)


class AutomaticRuleProfileTests(unittest.TestCase):
    def test_automatic_profile_requires_exact_basename_digest_and_expected_sheets(self):
        service = TabularRuleService(FileTabularRuleCatalog())
        from tests.unit.testTabularRules import structural_summary
        summary = structural_summary("Tabla completa ciano RdlP.xlsx")
        digest = "79770bdd190fcd08dd8351e18f2f1450d0b250bdeb8b9588e5129492a82cc025"
        self.assertEqual(service.select_compatible_profile(summary, digest), "self_sampling")
        self.assertIsNone(service.select_compatible_profile(summary, "0" * 64))
        summary.file_name = "another.xlsx"
        self.assertIsNone(service.select_compatible_profile(summary, digest))
        summary.file_name = "BaseCiano_PARA BELEN_5MAR2024 (2).XLSX"
        self.assertIsNone(service.select_compatible_profile(
            summary, "5bbf70623f4cbc0e2e8d82c22ab097c0144f1a3d39e1182c64b622829ee0bd2f",
        ))


if __name__ == "__main__":
    unittest.main()
