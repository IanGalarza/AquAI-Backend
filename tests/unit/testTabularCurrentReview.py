import json
import unittest

from aquai.application.contracts.tabular.results import (
    EmptySheetTreatment,
    ExecutionResult,
    LLMResponse,
    SheetCoverage,
    SheetExecution,
    TabularFinding,
)
from aquai.application.contracts.tabular.structure import (
    FormulaSample,
    SampleRow,
    SheetContentEvidence,
    SheetSummary,
    StructuralSummary,
)
from aquai.application.contracts.tabular.recipe import PreparationRecipe, SheetRecipe
from aquai.application.contracts.tabular.policies import RecordIdentityClarification
from aquai.application.services.tabularRules import TabularRuleService
from aquai.application.services.tabularValidation import TabularValidationService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog


class CurrentReviewPresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = TabularValidationService()
        self.summary = StructuralSummary(
            file_name="measurements.xlsx", file_type="xlsx", file_size_bytes=100,
            sheets=[SheetSummary(
                name="Measurements", rows=3, columns=2,
                initial_columns=["site", "value"], inferred_types=["string", "number"],
                null_counts=[0, 0], sample=[],
                content_evidence=SheetContentEvidence(
                    sheet_name="Measurements", inspected_cells=6, value_cells=6, formula_cells=0,
                ),
            )],
        )
        self.recipe = PreparationRecipe(
            status="ready", requires_human_review=False, warnings=[],
            sheets=[SheetRecipe(
                sheet_name="Measurements", block_id="records", role="observations",
                header_rows=[1], data_start_row=2, data_end_row=3,
                data_start_column=1, data_end_column=2, transformations=[],
            )],
        )
        self.execution = ExecutionResult(
            sheets=[SheetExecution(
                sheet_name="Measurements", block_id="records", role="observations",
                data_start_row=2, data_end_row=3, data_start_column=1, data_end_column=2,
                source_rows=3, source_columns=2, rows_before=2, rows_after=2,
                columns_before=["site", "value"], columns_after=["site", "value"],
            )],
            sheet_content_evidence=[self.summary.sheets[0].content_evidence.model_copy(deep=True)],
        )

    def add_verified_empty_unknown(self) -> None:
        evidence = SheetContentEvidence(
            sheet_name="Formatted", inspected_cells=13000, value_cells=0, formula_cells=0,
        )
        self.summary.sheets.append(SheetSummary(
            name="Formatted", rows=500, columns=26,
            initial_columns=[], inferred_types=[], null_counts=[], sample=[],
            empty_row_count=500, empty_column_count=26, content_evidence=evidence,
        ))
        self.recipe = PreparationRecipe(
            status="review_required", requires_human_review=True,
            warnings=["La propuesta original conserva una hoja vacía como unknown."],
            sheets=self.recipe.sheets + [SheetRecipe(
                sheet_name="Formatted", block_id="formatted_unknown", role="unknown",
                header_rows=[], data_start_row=1, data_end_row=500,
                data_start_column=1, data_end_column=26, transformations=[],
            )],
        )
        self.execution.skipped_sheets = ["Formatted"]
        self.execution.sheet_content_evidence.append(evidence.model_copy(deep=True))
        self.execution.empty_sheets.append(EmptySheetTreatment(
            sheet_name="Formatted", evidence=evidence.model_copy(deep=True),
            skipped_block_ids=["formatted_unknown"],
        ))

    def validate(self):
        return self.validator.validate(self.summary, self.recipe, self.execution)

    def test_verified_empty_unknown_is_separate_treatment_and_not_current_review(self) -> None:
        self.add_verified_empty_unknown()
        original_recipe = self.recipe.model_dump()
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertFalse(result.errors)
        self.assertEqual(result.review_reasons, [])
        self.assertFalse(any(item.code == "unresolved_block_role" for item in result.findings))
        self.assertEqual(result.original_proposal.model_dump(), {
            "status": "review_required", "requires_human_review": True,
            "warnings": original_recipe["warnings"],
        })
        self.assertEqual(self.recipe.model_dump(), original_recipe)
        self.assertEqual([sheet.sheet_name for sheet in self.execution.sheets], ["Measurements"])
        treatment = self.execution.empty_sheets[0]
        self.assertEqual(treatment.action, "verified_empty_no_table")
        self.assertEqual(treatment.skipped_block_ids, ["formatted_unknown"])

    def test_review_flag_and_original_warnings_are_history_not_unresolved_current_facts(self) -> None:
        self.recipe = PreparationRecipe(
            status="review_required", sheets=self.recipe.sheets,
            warnings=["La propuesta original recomienda verificar las fuentes."],
            requires_human_review=True,
        )
        before = self.recipe.model_dump()
        result = self.validate()
        self.assertNotEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual(result.review_reasons, [])
        self.assertEqual(result.original_proposal.warnings, before["warnings"])
        self.assertEqual(result.original_proposal.status, "review_required")
        self.assertTrue(result.original_proposal.requires_human_review)
        self.assertEqual(self.recipe.model_dump(), before)

    def test_nonempty_unknown_remains_a_concrete_current_reason(self) -> None:
        self.recipe = PreparationRecipe(
            status="review_required", requires_human_review=True, warnings=["Rol sin resolver."],
            sheets=[self.recipe.sheets[0].model_copy(update={"role": "unknown", "header_rows": []})],
        )
        self.execution.sheets[0].role = "unknown"
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        reasons = [item for item in result.review_reasons if item.code == "unresolved_block_role"]
        self.assertTrue(reasons)
        self.assertTrue(all(item.requires_review for item in reasons))
        self.assertEqual(reasons[0].sheet_name, "Measurements")
        self.assertEqual(reasons[0].block_id, "records")
        self.assertEqual(reasons[0].range, "A2:B3")

    def test_empty_samples_and_format_dimensions_cannot_substitute_full_evidence(self) -> None:
        for missing in ("inspection", "execution", "treatment", "incomplete_count"):
            with self.subTest(missing=missing):
                self.setUp()
                self.add_verified_empty_unknown()
                if missing == "inspection":
                    self.summary.sheets[-1].content_evidence = None
                elif missing == "execution":
                    self.execution.sheet_content_evidence.pop()
                elif missing == "treatment":
                    self.execution.empty_sheets.clear()
                else:
                    # Matching samples are not proof that the whole sheet was inspected.
                    self.summary.sheets[-1].content_evidence.inspected_cells = 100
                    self.execution.sheet_content_evidence[-1].inspected_cells = 100
                    self.execution.empty_sheets[-1].evidence.inspected_cells = 100
                result = self.validate()
                self.assertNotIn(result.status, (EvaluationStatus.READY, EvaluationStatus.READY_WITH_WARNINGS))
                self.assertTrue(result.errors or result.review_reasons)

    def test_unknown_with_uncached_formula_remains_nonempty_and_requires_review(self) -> None:
        evidence = SheetContentEvidence(
            sheet_name="Formula source", inspected_cells=4, value_cells=0, formula_cells=1,
        )
        formula = FormulaSample(cell="A1", formula="=1+1", cache_available=False)
        self.summary.sheets.append(SheetSummary(
            name="Formula source", rows=2, columns=2,
            initial_columns=[], inferred_types=[], null_counts=[], sample=[],
            empty_row_count=2, empty_column_count=2, formula_count=1,
            missing_formula_cache_count=1, formula_sample=[formula], content_evidence=evidence,
        ))
        self.recipe = PreparationRecipe(
            status="review_required", requires_human_review=True, warnings=["Fórmula no evaluada."],
            sheets=self.recipe.sheets + [SheetRecipe(
                sheet_name="Formula source", block_id="formula_unknown", role="unknown",
                header_rows=[], data_start_row=1, data_end_row=2,
                data_start_column=1, data_end_column=2, transformations=[],
            )],
        )
        self.execution.sheets.append(SheetExecution(
            sheet_name="Formula source", block_id="formula_unknown", role="unknown",
            data_start_row=1, data_end_row=2, data_start_column=1, data_end_column=2,
            source_rows=2, source_columns=2, rows_before=2, rows_after=2,
            columns_before=["A", "B"], columns_after=["A", "B"],
            formula_count=1, unresolved_formula_count=1,
            unresolved_formula_cells=["A1"], formula_sample=[formula],
        ))
        self.execution.sheet_content_evidence.append(evidence.model_copy(deep=True))
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        current_codes = {item.code for item in result.review_reasons}
        self.assertIn("source_missing_formula_cache", current_codes)
        self.assertIn("unresolved_block_role", current_codes)
        self.assertFalse(evidence.is_empty)
        self.assertEqual(self.execution.empty_sheets, [])

    def test_resolved_source_note_is_history_while_unrelated_conflict_still_reviews(self) -> None:
        self.add_verified_empty_unknown()
        note = "No sé si hay alguna repetida con otra campaña."
        self.summary.sheets[0].column_indices = [1, 2]
        self.summary.sheets[0].sample = [SampleRow(row_number=3, values=["A", note])]
        self.execution.findings = [TabularFinding(
            code="ambiguous_period", message="No hay información suficiente para fijar el período.",
            sheet_name="Measurements", block_id="records", cell="A2",
        )]
        clarification = RecordIdentityClarification(
            rule_id="reviewed_note", original_note=note,
            statement="El usuario confirmó que son mediciones distintas.",
            confirmed_by="usuario", source_reference="confirmación humana verificada",
        )
        result = self.validator.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=(clarification,),
        )
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual([item.code for item in result.review_reasons], ["ambiguous_period"])
        self.assertEqual(result.review_reasons[0].cell, "A2")
        clarified = next(item for item in result.findings if item.code == "source_record_identity_clarified")
        self.assertFalse(clarified.requires_review)
        self.assertNotIn(clarified, result.review_reasons)
        self.assertFalse(any(item.code == "unresolved_block_role" for item in result.findings))

    def test_verified_empty_does_not_hide_coverage_conflict(self) -> None:
        self.add_verified_empty_unknown()
        self.execution.coverage = [SheetCoverage(
            sheet_name="Measurements", nonempty_cells=6, covered_nonempty_cells=5,
            uncovered_nonempty_cells=1, overlapping_nonempty_cells=0, uncovered_sample=["B3"],
        )]
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        reason = next(item for item in result.review_reasons if item.code == "uncovered_source_cells")
        self.assertEqual(reason.cell, "B3")
        self.assertFalse(any(item.code == "unresolved_block_role" for item in result.review_reasons))

    def test_verified_empty_does_not_hide_failed_execution(self) -> None:
        self.add_verified_empty_unknown()
        self.execution.sheets[0].errors = ["No se pudo conservar una columna."]
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertIn("Hoja 'Measurements': No se pudo conservar una columna.", result.errors)
        self.assertFalse(any(item.code == "unresolved_block_role" for item in result.review_reasons))


class FixedOfflineRecipeProvider:
    """Fixture con contenido inmutable y sin cliente de red."""

    def __init__(self) -> None:
        self.content = json.dumps({
            "status": "ready", "requires_human_review": False, "warnings": [],
            "sheets": [{
                "sheet_name": "data", "block_id": "records", "role": "observations",
                "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
                "data_start_column": 1, "data_end_column": 2, "transformations": [],
            }],
        })

    async def generate_recipe(self, summary, model, known_rules):
        return LLMResponse(model=model, content=self.content)


class CurrentRuleReviewPresentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_rule_conflict_added_by_use_case_is_in_review_reasons(self) -> None:
        provider = FixedOfflineRecipeProvider()
        evaluator = TabularEvaluationService(
            PandasTabularProcessor(), provider, default_model="offline",
            max_upload_bytes=10000, rule_service=TabularRuleService(FileTabularRuleCatalog()),
        )
        result = await evaluator.execute(
            "measurements.csv", b"site,value\nA,0\nB,2\n",
            known_rules="Eliminar todos los ceros de los datos.",
        )
        self.assertEqual(result.validation.status, EvaluationStatus.REVIEW_REQUIRED)
        rule_reason = next(item for item in result.validation.review_reasons
                           if item.code == "rule_additional_conflict")
        self.assertTrue(rule_reason.requires_review)
        self.assertIn("preserve_original_values", rule_reason.message)
        self.assertEqual(result.validation.original_proposal.status, "ready")
        self.assertEqual(result.llm_response.content, provider.content)
        self.assertEqual([row.values for row in result.execution.sheets[0].sample], [["A", "0"], ["B", "2"]])


if __name__ == "__main__":
    unittest.main()
