"""Regresiones offline con recetas fijas y referencias anteriores al refactor."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import unittest
from uuid import UUID

from aquai.application.contracts.tabular.recipe import PreparationRecipe
from aquai.application.contracts.tabular.results import TabularEvaluation, TabularFinding
from aquai.application.errors import SourceDecisionError
from aquai.application.services.sourceReviews import SourceReviewService
from aquai.application.services.tabularRules import RuleComposition, TabularRuleService
from aquai.application.services.tabularValidation import TabularValidationService
from aquai.infrastructure.processing.tabular.normalization import trace_value
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog
from tests.evaluation.workbookFixtures import WORKBOOKS


class WorkbookRegressionTests(unittest.TestCase):
    def setUp(self):
        self.datasets = Path(os.environ.get(
            "AQUAI_TEST_DATASETS", str(Path(__file__).resolve().parents[3] / "Datasets"),
        ))
        self.processor = PandasTabularProcessor(sample_rows=2500)
        self.rules = TabularRuleService(FileTabularRuleCatalog())
        self.validator = TabularValidationService()
        self.reviews = SourceReviewService()
        self.source_id = UUID("00000000-0000-0000-0000-000000000001")

    def test_biological_counts(self):
        self._check_workbook(WORKBOOKS[0])

    def test_self_sampling(self):
        self._check_workbook(WORKBOOKS[1])

    def test_belen_cyanobacteria(self):
        self._check_workbook(WORKBOOKS[2])

    def _check_workbook(self, fixture):
        path = self.datasets / fixture["path"]
        if not path.is_file():
            self.skipTest(f"Excel externo ausente: {path}; configurar AQUAI_TEST_DATASETS.")
        content = path.read_bytes()
        digest = sha256(content).hexdigest()
        self.assertEqual(digest, fixture["file_sha256"], "Cambió el Excel de referencia.")
        summary = self.processor.inspect(path.name, content)
        recipe = PreparationRecipe.model_validate(fixture["recipe"])
        composition = self.rules.compose(summary, fixture["profile"], source_sha256=digest)
        execution = self.processor.execute(
            path.name, content, recipe, summary, policies=composition.preparation_policies,
        )
        validation = self.validator.validate(
            summary, recipe, execution, identity_clarifications=composition.identity_clarifications,
            preparation_policies=composition.preparation_policies,
        )
        rule_findings = self.rules.validate_recipe(summary, recipe, composition)
        validation.findings.extend(
            TabularFinding.model_validate(item) for item in composition.findings + rule_findings
        )
        evaluation = TabularEvaluation(
            model="offline", structural_summary=summary, recipe=recipe, execution=execution,
            validation=validation, applied_rules=composition.metadata,
        )
        snapshot = composition.snapshot()
        self.assertEqual(RuleComposition.restore(snapshot).snapshot(), snapshot)
        active_reviews = self.reviews.build_reviews(self.source_id, evaluation, snapshot)
        self.assertEqual(active_reviews, self.reviews.build_reviews(self.source_id, evaluation, snapshot))
        next_reviews = self.reviews.build_reviews(self.source_id, evaluation, snapshot, revision=1)
        self.assertTrue({item["id"] for item in active_reviews}.isdisjoint(
            {item["id"] for item in next_reviews},
        ))
        for review in active_reviews:
            self.assertIn(TabularFinding.model_validate(review["finding"]), validation.findings)
            if review["options"]:
                stale = deepcopy(review)
                stale["finding"]["requires_review"] = False
                with self.assertRaises(SourceDecisionError):
                    self.reviews.apply(stale, review["options"][0]["action"], {}, evaluation, snapshot)
            for option in review["options"]:
                # La representación del schema se prueba aparte; aquí se conserva su acción y alcance.
                option.pop("values", None)
        workbook = self.processor.reader.read(path.name, content)
        formulas = {
            sheet.name: [
                {"row": row, "column": column, "expression": formula.expression,
                 "cached": trace_value(formula.cached_value)}
                for (row, column), formula in sheet.formula_cells.items()
            ] for sheet in workbook.sheets
        }
        document = {
            "structure": summary.model_dump(mode="json"), "execution": execution.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"), "rules": snapshot,
            "reviews": active_reviews, "formulas": formulas,
        }
        canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(sha256(canonical).hexdigest(), fixture["expected_sha256"],
                         "Cambió la preparación, procedencia, fórmulas, reglas o revisiones aceptadas.")
        self.assertEqual(len(recipe.sheets), fixture["blocks"])
        self.assertEqual(validation.status.value, fixture["status"])
        self.assertEqual(
            [[sheet.sheet_name, sheet.block_id, sheet.rows_after] for sheet in execution.sheets], fixture["rows"],
        )
