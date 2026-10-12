"""Persisted reviews captured before the service extraction, without a provider."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID

from jsonschema import Draft202012Validator

from aquai.application.contracts.tabular.recipe import SheetRecipe
from aquai.application.contracts.tabular.results import TabularEvaluation
from aquai.application.errors import SourceDecisionError
from aquai.application.services.sourceReviews import SourceReviewService


class StoredSourceReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(Path(__file__).with_name("sourceReviewFixtures.json").read_text(encoding="utf-8"))
        self.evaluation = TabularEvaluation.model_validate(fixture["evaluation"])
        self.saved_reviews = fixture["reviews"]
        self.service = SourceReviewService()
        self.rules = {
            "metadata": {"profile": None, "version": "stored", "evidence": {"original": True}},
            "prompt_rules": {"common": [{"id": "stored_rule", "text": "Conservar originales."}]},
            "findings": [], "review_reasons": [], "identity_clarifications": [],
            "preparation_policies": None,
            "source_decisions": [{"review_id": "earlier", "values": {"nested": ["original"]}}],
        }

    def saved_review(self, action):
        return deepcopy(next(review for review in self.saved_reviews
                             if any(option["action"] == action for option in review["options"])))

    def assert_rejected(self, review, action, values, message=None):
        original_evaluation = self.evaluation.model_dump(mode="json")
        original_rules = deepcopy(self.rules)
        original_review = deepcopy(review)
        assertion = (self.assertRaisesRegex(SourceDecisionError, message) if message
                     else self.assertRaises(SourceDecisionError))
        with assertion:
            self.service.apply(review, action, values, self.evaluation, self.rules)
        self.assertEqual(self.evaluation.model_dump(mode="json"), original_evaluation)
        self.assertEqual(self.rules, original_rules)
        self.assertEqual(review, original_review)

    def test_saved_review_ids_options_order_and_finding_serialization_are_unchanged(self):
        actual = self.service.build_reviews(
            UUID("00000000-0000-0000-0000-000000000001"), self.evaluation, self.rules, revision=2,
        )
        self.assertEqual(actual, self.saved_reviews)
        self.assertEqual(len(actual), 5)
        self.assertEqual(actual[-1]["options"], [])
        self.assertTrue(actual[-1]["blocked_reason"])

    def test_saved_reviews_resolve_all_four_actions_with_independent_copies_and_scope(self):
        block = self.evaluation.recipe.sheets[0].model_dump(mode="json")
        cases = (
            ("clarify_record_identity", {"identity": "distinct", "statement": " Son registros distintos. "}, "validation"),
            ("correct_transformations", {"transformations": [{"operation": "strip_strings"}]}, "execution"),
            ("replace_block", {"block": {**block, "role": "legend"}}, "execution"),
            ("add_block", {"block": {**block, "block_id": "tail", "data_start_row": 4}}, "execution"),
        )
        for action, values, expected_stage in cases:
            with self.subTest(action=action):
                review = self.saved_review(action)
                evaluation = self.evaluation.model_copy(deep=True)
                if action == "add_block":
                    evaluation.recipe.sheets[0].data_end_row = 3
                original_evaluation = evaluation.model_dump(mode="json")
                original_rules = deepcopy(self.rules)
                original_review = deepcopy(review)
                original_values = deepcopy(values)
                Draft202012Validator(review["options"][0]["values"]).validate(values)
                result, rules, stage = self.service.apply(
                    review, action, values, evaluation, self.rules,
                )
                self.assertEqual(stage, expected_stage)
                self.assertEqual(evaluation.model_dump(mode="json"), original_evaluation)
                self.assertEqual(self.rules, original_rules)
                self.assertEqual(review, original_review)
                self.assertEqual(values, original_values)
                self.assertEqual(rules["source_decisions"][-1], {
                    "review_id": review["id"], "action": action, "values": original_values,
                    "scope": {
                        "sheet_name": "data", "block_id": "records", "cell": review["cell"],
                        "range": review["range"], "region_id": None,
                    },
                })
                if action == "clarify_record_identity":
                    clarification = rules["identity_clarifications"][-1]
                    self.assertEqual(clarification["original_note"], evaluation.structural_summary.sheets[0].sample[0].values[0])
                    self.assertEqual(clarification["statement"], "Son registros distintos.")
                    self.assertEqual((clarification["sheet_name"], clarification["cell"]), ("data", "A3"))
                    self.assertEqual(result.recipe, evaluation.recipe)
                elif action == "correct_transformations":
                    self.assertEqual(result.recipe.sheets[0].transformations[0].operation, "strip_strings")
                elif action == "replace_block":
                    self.assertEqual(result.recipe.sheets[0].role, "legend")
                else:
                    self.assertEqual([item.block_id for item in result.recipe.sheets], ["records", "tail"])
                result.structural_summary.sheets[0].sample[0].values[0] = "changed copy"
                rules["metadata"]["evidence"]["original"] = False
                rules["source_decisions"][0]["values"]["nested"].append("changed copy")
                self.assertEqual(evaluation.model_dump(mode="json"), original_evaluation)
                self.assertEqual(self.rules, original_rules)

    def test_saved_reviews_cannot_waive_or_correct_a_finding_that_is_no_longer_current(self):
        review = self.saved_review("correct_transformations")
        self.assert_rejected(review, "approve", {}, "no está admitida")
        self.assert_rejected(self.saved_reviews[-1], "add_block", {"block": {}}, "no está admitida")
        self.evaluation.validation.findings = []
        self.assert_rejected(review, "correct_transformations", {"transformations": []}, "vigente")

    def test_all_actions_reject_values_outside_their_contracts_without_mutation(self):
        block = self.evaluation.recipe.sheets[0].model_dump(mode="json")
        cases = (
            ("clarify_record_identity", {"identity": "same", "statement": "Confirmación."}),
            ("clarify_record_identity", {"identity": "distinct", "statement": " "}),
            ("clarify_record_identity", {"identity": "distinct", "statement": "a" * 4001}),
            ("clarify_record_identity", {"identity": "distinct", "statement": "Confirmación.", "waive": True}),
            ("correct_transformations", None),
            ("correct_transformations", {"transformations": {}}),
            ("correct_transformations", {"transformations": [{"operation": "drop_everything"}]}),
            ("correct_transformations", {"transformations": [], "waive": True}),
            ("replace_block", {"block": {**block, "unexpected": True}}),
            ("replace_block", {"block": block, "waive": True}),
            ("add_block", {"block": {**block, "block_id": "tail"}, "waive": True}),
        )
        for action, values in cases:
            with self.subTest(action=action, values=values):
                self.assert_rejected(self.saved_review(action), action, values)

    def test_block_actions_reject_wrong_sheet_identity_bounds_transformations_and_noops(self):
        block = self.evaluation.recipe.sheets[0].model_dump(mode="json")
        cases = (
            ("replace_block", block),
            ("replace_block", {**block, "sheet_name": "other"}),
            ("replace_block", {**block, "block_id": "other"}),
            ("replace_block", {**block, "data_end_row": 5}),
            ("replace_block", {**block, "data_end_column": 3}),
            ("replace_block", {**block, "transformations": [{"operation": "strip_strings"}]}),
            ("add_block", {**block, "sheet_name": "other", "block_id": "tail"}),
            ("add_block", {**block, "block_id": "tail", "data_end_row": 5}),
            ("add_block", block),
        )
        for action, candidate in cases:
            with self.subTest(action=action, candidate=candidate):
                self.assert_rejected(self.saved_review(action), action, {"block": candidate})
        self.assert_rejected(self.saved_review("correct_transformations"), "correct_transformations",
                             {"transformations": []}, "no modifica")

    def test_structural_action_validates_complete_recipe_status_after_validating_the_block(self):
        block = self.evaluation.recipe.sheets[0].model_dump(mode="json")
        candidate = {**block, "role": "unknown"}
        SheetRecipe.model_validate(candidate)
        self.assert_rejected(self.saved_review("replace_block"), "replace_block",
                             {"block": candidate}, "contrato de la receta")
        self.assert_rejected(self.saved_review("add_block"), "add_block",
                             {"block": {**candidate, "block_id": "tail"}}, "contrato de la receta")

    def test_block_corrections_preserve_other_blocks_and_transformation_scope(self):
        self.evaluation.recipe.sheets[0].data_end_row = 3
        other = self.evaluation.recipe.sheets[0].model_copy(deep=True, update={
            "block_id": "other", "data_start_row": 4, "data_end_row": 4,
        })
        self.evaluation.recipe.sheets.append(other)
        original_other = other.model_dump(mode="json")
        for action, values in (
            ("correct_transformations", {"transformations": [{"operation": "strip_strings"}]}),
            ("replace_block", {"block": {
                **self.evaluation.recipe.sheets[0].model_dump(mode="json"), "role": "legend",
            }}),
        ):
            with self.subTest(action=action):
                result, _, stage = self.service.apply(
                    self.saved_review(action), action, values, self.evaluation, self.rules,
                )
                self.assertEqual(stage, "execution")
                self.assertEqual(result.recipe.sheets[1].model_dump(mode="json"), original_other)
                self.assertEqual(self.evaluation.recipe.sheets[0].transformations, [])

    def test_transformation_option_schema_does_not_depend_on_recipe_branch_position(self):
        original_schema = SheetRecipe.model_json_schema()
        reordered_schema = deepcopy(original_schema)
        reordered_schema["oneOf"] = original_schema["oneOf"][1:] + original_schema["oneOf"][:1]
        with patch.object(SheetRecipe, "model_json_schema", return_value=reordered_schema):
            actual = self.service.build_reviews(
                UUID("00000000-0000-0000-0000-000000000001"), self.evaluation, self.rules, revision=2,
            )
        self.assertEqual(actual[0]["options"], self.saved_reviews[0]["options"])
        schema = actual[0]["options"][0]["values"]
        Draft202012Validator(schema).validate({"transformations": [{"operation": "drop_empty_rows"}]})
        self.assertFalse(Draft202012Validator(schema).is_valid({
            "transformations": [{"operation": "drop_empty_rows"}] * 51,
        }))
