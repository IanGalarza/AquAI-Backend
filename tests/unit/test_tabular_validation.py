import unittest

from aquai.application.dtos.tabular import (
    ExecutionResult,
    PreparationRecipe,
    SheetExecution,
    SheetRecipe,
    SheetSummary,
    StructuralSummary,
    ValidationResult,
)
from aquai.application.services.tabular_validation import TabularValidationService
from aquai.domain.enums.tabular import EvaluationStatus


class TabularValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = TabularValidationService()
        self.summary = StructuralSummary(
            file_name="measurements.xlsx",
            file_type="xlsx",
            file_size_bytes=100,
            sheets=[self._summary_sheet("Measurements")],
        )
        self.recipe = PreparationRecipe(
            status="ready",
            sheets=[SheetRecipe(sheet_name="Measurements", header_rows=[1], data_start_row=2)],
        )
        self.execution = ExecutionResult(sheets=[self._execution_sheet("Measurements")])

    def _summary_sheet(self, name: str, **changes: object) -> SheetSummary:
        values: dict[str, object] = {
            "name": name,
            "rows": 4,
            "columns": 2,
            "initial_columns": ["location", "value"],
            "inferred_types": ["string", "number"],
            "null_counts": [0, 0],
            "sample": [],
        }
        values.update(changes)
        return SheetSummary.model_validate(values)

    def _execution_sheet(self, name: str, **changes: object) -> SheetExecution:
        values: dict[str, object] = {
            "sheet_name": name,
            "source_rows": 4,
            "source_columns": 2,
            "rows_before": 3,
            "rows_after": 3,
            "columns_before": ["location", "value"],
            "columns_after": ["location", "value"],
        }
        values.update(changes)
        return SheetExecution.model_validate(values)

    def _validate(self) -> ValidationResult:
        return self.service.validate(self.summary, self.recipe, self.execution)

    def test_valid_result_does_not_treat_removed_header_as_lost_record(self) -> None:
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.errors, [])

    def test_expected_empty_cleanup_only_requires_warnings(self) -> None:
        self.execution.sheets[0] = self._execution_sheet(
            "Measurements",
            rows_after=2,
            dropped_empty_rows=1,
            columns_after=["location"],
            dropped_empty_columns=["value"],
        )
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertFalse(result.errors)
        self.assertEqual(len(result.warnings), 2)

    def test_unexpected_record_loss_requires_review_even_if_model_says_ready(self) -> None:
        self.execution.sheets[0].rows_after = 2
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any("Se perdieron 1 filas" in warning for warning in result.warnings))

    def test_uncertainty_signals_require_review(self) -> None:
        signals = {
            "excluded_nonempty_rows": 1,
            "conversion_failures": {"value": 1},
            "unresolved_formula_cells": ["B2"],
            "error_cells": ["B2"],
            "suspicious_header_rows": [1],
            "columns_after": ["location", "location"],
        }
        for field, value in signals.items():
            with self.subTest(field=field):
                self.execution.sheets[0] = self._execution_sheet("Measurements", **{field: value})
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
                self.assertFalse(result.errors)

    def test_unexpected_column_loss_requires_review_but_rename_preserves_shape(self) -> None:
        self.execution.sheets[0].columns_after = ["place", "measurement"]
        self.assertEqual(self._validate().status, EvaluationStatus.READY)
        self.execution.sheets[0].columns_after = ["place"]
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_zero_failed_conversions_do_not_require_review(self) -> None:
        self.execution.sheets[0].conversion_failures = {"value": 0}
        self.assertEqual(self._validate().status, EvaluationStatus.READY)

    def test_warnings_from_every_stage_are_preserved_and_deduplicated(self) -> None:
        self.summary.warnings = ["Archivo inusual."]
        self.summary.sheets[0].warnings = ["Celdas combinadas."]
        self.recipe.warnings = ["Archivo inusual.", "Encabezado inferido."]
        self.execution.sheets[0].warnings = ["Espacios eliminados."]
        self.execution.sheets[0].empty_columns = ["value"]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.warnings.count("Archivo inusual."), 1)
        self.assertTrue(any("Celdas combinadas" in warning for warning in result.warnings))
        self.assertTrue(any("Espacios eliminados" in warning for warning in result.warnings))
        self.assertTrue(any("completamente vacías" in warning for warning in result.warnings))

    def test_model_review_flags_are_preserved(self) -> None:
        for flag in ("status", "requires_human_review"):
            with self.subTest(flag=flag):
                self.recipe.status = "review_required" if flag == "status" else "ready"
                self.recipe.requires_human_review = flag == "requires_human_review"
                self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_failed_recipe_cannot_be_overridden_by_successful_execution(self) -> None:
        self.recipe.status = "failed"
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(result.errors)

    def test_recipe_without_executable_proposal_keeps_review_status(self) -> None:
        self.recipe = PreparationRecipe(status="review_required")
        self.execution = ExecutionResult(skipped_sheets=["Measurements"])
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertFalse(result.errors)
        self.assertTrue(any("no propone ninguna hoja" in warning for warning in result.warnings))

    def test_execution_errors_have_priority_over_review_and_warnings(self) -> None:
        self.recipe.requires_human_review = True
        self.execution.sheets[0].errors = ["Conversión incompatible."]
        self.execution.sheets[0].empty_columns = ["value"]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertIn("Hoja 'Measurements': Conversión incompatible.", result.errors)
        self.assertTrue(result.warnings)

    def test_missing_execution_for_recipe_sheet_is_a_failure(self) -> None:
        self.recipe.sheets.append(SheetRecipe(sheet_name="Second", data_start_row=1))
        self.summary.sheets.append(self._summary_sheet("Second"))
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(any("No se ejecutó" in error for error in result.errors))

    def test_nonempty_skipped_sheet_requires_review(self) -> None:
        self.summary.sheets.append(self._summary_sheet("Second"))
        self.execution.skipped_sheets = ["Second"]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any("Second" in warning and "contiene datos" in warning for warning in result.warnings))

    def test_unreported_omission_of_nonempty_sheet_also_requires_review(self) -> None:
        self.summary.sheets.append(self._summary_sheet("Second"))
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_empty_skipped_sheet_uses_exact_count_when_indices_are_truncated(self) -> None:
        self.summary.sheets.append(
            self._summary_sheet("Empty", rows=500, empty_rows=list(range(1, 101)), empty_row_count=500)
        )
        self.execution.skipped_sheets = ["Empty"]
        self.assertEqual(self._validate().status, EvaluationStatus.READY_WITH_WARNINGS)

    def test_skipped_sheet_with_unresolved_formulas_is_not_classified_as_empty(self) -> None:
        self.summary.sheets.append(
            self._summary_sheet("Formulas", empty_row_count=4, formula_count=1)
        )
        self.execution.skipped_sheets = ["Formulas"]
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_skipped_sheet_with_excel_errors_is_not_classified_as_empty(self) -> None:
        self.summary.sheets.append(
            self._summary_sheet("Errors", empty_row_count=4, error_cell_count=1, error_cells=["A2"])
        )
        self.execution.skipped_sheets = ["Errors"]
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_no_execution_or_empty_final_tables_are_failures(self) -> None:
        for changes in ({"rows_after": 0}, {"columns_after": []}):
            with self.subTest(changes=changes):
                self.execution.sheets[0] = self._execution_sheet("Measurements", **changes)
                self.assertEqual(self._validate().status, EvaluationStatus.FAILED)
        self.execution.sheets = []
        self.assertEqual(self._validate().status, EvaluationStatus.FAILED)

    def test_execution_cannot_repeat_or_silently_add_sheets(self) -> None:
        for name in ("Measurements", "Unexpected"):
            with self.subTest(name=name):
                self.execution.sheets = [
                    self._execution_sheet("Measurements"), self._execution_sheet(name)
                ]
                self.assertEqual(self._validate().status, EvaluationStatus.FAILED)

    def test_contradictory_counts_do_not_produce_ready_status(self) -> None:
        self.execution.sheets[0].dropped_empty_rows = 1
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)
        self.execution.sheets[0].rows_before = -1
        self.assertEqual(self._validate().status, EvaluationStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
