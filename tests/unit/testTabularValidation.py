import unittest

from aquai.application.contracts.tabular.results import (
    ExecutionResult,
    SheetExecution,
    SheetCoverage,
    TabularFinding,
    ValidationResult,
)
from aquai.application.contracts.tabular.structure import (
    FormulaSample,
    RecordOverlap,
    SheetSummary,
    SampleRow,
    StructuralSummary,
    StructuralRegion,
    SourceRowMatch,
)
from aquai.application.contracts.tabular.recipe import (
    PreparationRecipe,
    RenameColumns,
    SheetRecipe,
)
from aquai.application.contracts.tabular.policies import RecordIdentityClarification
from aquai.application.services.tabularValidation import TabularValidationService
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
            sheets=[SheetRecipe(
                sheet_name="Measurements", block_id="measurements", role="observations",
                header_rows=[1], data_start_row=2, data_end_row=4,
                data_start_column=1, data_end_column=2, transformations=[],
            )],
            warnings=[],
            requires_human_review=False,
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
            "block_id": "measurements",
            "data_start_row": 2,
            "data_end_row": 4,
            "data_start_column": 1,
            "data_end_column": 2,
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

    def _add_overlapping_view(self, ambiguous: bool = False) -> None:
        self.summary.sheets.append(self._summary_sheet("Sorted", rows=3))
        self.recipe.sheets.append(SheetRecipe(
            sheet_name="Sorted", block_id="sorted", role="observations",
            header_rows=[1], data_start_row=2, data_end_row=3,
            data_start_column=1, data_end_column=2, transformations=[],
        ))
        self.execution.sheets.append(self._execution_sheet(
            "Sorted", block_id="sorted", source_rows=3, rows_before=2, rows_after=2, data_end_row=3,
        ))
        relation = RecordOverlap(
            left_sheet="Measurements", right_sheet="Sorted",
            left_header_rows=[1], right_header_rows=[1],
            left_start_column=1, left_end_column=2, right_start_column=1, right_end_column=2,
            left_record_count=3, right_record_count=2,
            matches=[SourceRowMatch(left_row=2, right_row=3), SourceRowMatch(left_row=4, right_row=2)],
            left_only_rows=[3], right_only_rows=[], ambiguous_repeated_rows=ambiguous,
        )
        self.summary.record_overlaps = [relation]
        self.execution.record_overlaps = [relation.model_copy(deep=True)]

    def test_exact_overlapping_views_preserve_both_sources_without_requiring_review(self) -> None:
        self._add_overlapping_view()
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.errors, [])
        finding = next(item for item in result.findings if item.code == "source_record_overlap")
        self.assertFalse(finding.requires_review)
        self.assertEqual(finding.range, "A2:B4")
        self.assertIn("2 registros", finding.message)
        self.assertIn("1 registros exclusivos", finding.message)
        self.assertEqual([item.rows_after for item in self.execution.sheets], [3, 2])
        self.assertEqual(self.execution.record_overlaps[0].policy, "preserve_sources_do_not_add_matches")

    def test_missing_executed_overlap_evidence_is_an_error(self) -> None:
        self._add_overlapping_view()
        self.execution.record_overlaps = []
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(any(item.code == "record_overlap_evidence_mismatch" for item in result.findings))

    def test_overlap_evidence_multiplicity_must_be_preserved(self) -> None:
        self._add_overlapping_view()
        relation = self.summary.record_overlaps[0]
        for source_count, execution_count in ((1, 2), (2, 1)):
            with self.subTest(source_count=source_count, execution_count=execution_count):
                self.summary.record_overlaps = [relation] * source_count
                self.execution.record_overlaps = [relation] * execution_count
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(any(item.code == "record_overlap_evidence_mismatch" for item in result.findings))

    def test_overlap_evidence_order_does_not_change_validation(self) -> None:
        self._add_overlapping_view()
        relation = self.summary.record_overlaps[0]
        reverse = RecordOverlap.model_validate({
            **relation.model_dump(), "left_sheet": "Sorted", "right_sheet": "Measurements",
            "left_record_count": 2, "right_record_count": 3,
            "matches": [{"left_row": match.right_row, "right_row": match.left_row}
                        for match in relation.matches],
            "left_only_rows": [], "right_only_rows": [3],
        })
        self.summary.record_overlaps = [relation, reverse]
        self.execution.record_overlaps = [reverse, relation]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertFalse(result.errors)
        self.assertFalse(any(item.code == "record_overlap_evidence_mismatch" for item in result.findings))

    def test_repeated_identical_content_inside_a_view_requires_identity_review(self) -> None:
        self._add_overlapping_view(ambiguous=True)
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_internal_repetitions_require_review_without_claiming_shared_records(self) -> None:
        self._add_overlapping_view()
        original = self.summary.record_overlaps[0].model_dump()
        for ambiguous in (False, True):
            with self.subTest(ambiguous=ambiguous):
                relation = RecordOverlap.model_validate({
                    **original, "matches": [], "left_only_rows": [2, 3, 4], "right_only_rows": [2, 3],
                    "ambiguous_repeated_rows": ambiguous,
                })
                self.summary.record_overlaps = [relation]
                self.execution.record_overlaps = [relation.model_copy(deep=True)]
                result = self._validate()
                expected = EvaluationStatus.REVIEW_REQUIRED if ambiguous else EvaluationStatus.READY
                self.assertEqual(result.status, expected)
                self.assertFalse(result.errors)
                self.assertFalse(any(item.code == "source_record_overlap" for item in result.findings))
                repetitions = [item for item in result.findings if item.code == "source_repeated_record_content"]
                self.assertEqual(len(repetitions), int(ambiguous))
                if repetitions:
                    self.assertTrue(repetitions[0].requires_review)
                    self.assertIn("No hay registros compartidos", repetitions[0].message)
                self.assertEqual(self.execution.record_overlaps[0].matches, [])

    def test_source_duplicate_uncertainty_is_located_and_deduplicated_across_samples(self) -> None:
        note = SampleRow(row_number=3, values=["A", "No sé si hay alguna repetida con otra campaña."])
        self.summary.sheets[0].column_indices = [1, 2]
        self.summary.sheets[0].sample = [note]
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="records", start_row=2, end_row=4, start_column=1, end_column=2,
            kind="observations", reason="Registros por sitio.", sample=[note], column_indices=[1, 2],
        )]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        findings = [item for item in result.findings if item.code == "source_record_identity_uncertainty"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].cell, "B3")
        self.assertEqual(findings[0].sheet_name, "Measurements")
        self.assertEqual(self.execution.sheets[0].rows_after, 3)

    def test_uncertain_identity_notes_allow_line_breaks_and_impersonal_wording(self) -> None:
        self.summary.sheets[0].column_indices = [1, 2]
        for text in ("No se sabe si hay registros repetidos.", "No sé si hay alguna\nrepetida.",
                     "No\tse\nsabe si hay alguna duplicada."):
            with self.subTest(text=text):
                note = SampleRow(row_number=3, values=["A", text])
                self.summary.sheets[0].sample = [note]
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
                findings = [item for item in result.findings if item.code == "source_record_identity_uncertainty"]
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].cell, "B3")
                self.assertEqual(note.values[1], text)

    def _confirmed_note_context(self) -> tuple[RecordIdentityClarification, ...]:
        text = "No sé si hay alguna repetida con otra campaña."
        self.summary.sheets[0].column_indices = [1, 2]
        self.summary.sheets[0].sample = [
            SampleRow(row_number=2, values=["A", 0]),
            SampleRow(row_number=3, values=["A", text]),
            SampleRow(row_number=4, values=["A", 42]),
        ]
        return (RecordIdentityClarification(
            rule_id="reviewed_note", original_note=text,
            statement="Son mediciones distintas; conservarlas independientes y usar sus períodos originales.",
            confirmed_by="el usuario", source_reference="confirmación humana verificada",
        ),)

    def test_confirmed_note_is_historical_evidence_not_a_review_trigger(self) -> None:
        context = self._confirmed_note_context()
        source_before = self.summary.model_dump()
        recipe_before = self.recipe.model_dump()
        result = self.service.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=context,
        )
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertFalse(result.errors)
        self.assertFalse(any(item.requires_review for item in result.findings))
        finding = next(item for item in result.findings if item.code == "source_record_identity_clarified")
        self.assertEqual((finding.sheet_name, finding.cell), ("Measurements", "B3"))
        self.assertIn(context[0].original_note, finding.message)
        self.assertIn(context[0].rule_id, finding.message)
        self.assertIn("usuario", finding.message)
        self.assertEqual(self.summary.model_dump(), source_before)
        self.assertEqual(self.recipe.model_dump(), recipe_before)
        self.assertEqual(self.execution.sheets[0].rows_after, 3)

    def test_note_confirmation_does_not_cover_another_uncertain_note(self) -> None:
        context = self._confirmed_note_context()
        self.summary.sheets[0].sample[-1].values[1] = "Posible registro duplicado sin verificar."
        result = self.service.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=context,
        )
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        unresolved = next(item for item in result.findings if item.code == "source_record_identity_uncertainty")
        self.assertEqual(unresolved.cell, "B4")
        self.assertTrue(unresolved.requires_review)

    def test_note_confirmation_preserves_duplicate_and_overlap_integrity_checks(self) -> None:
        context = self._confirmed_note_context()
        self._add_overlapping_view(ambiguous=True)
        result = self.service.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=context,
        )
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        overlap = next(item for item in result.findings if item.code == "source_record_overlap")
        self.assertTrue(overlap.requires_review)
        self.assertEqual(self.execution.record_overlaps, self.summary.record_overlaps)
        self.execution.record_overlaps = []
        result = self.service.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=context,
        )
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(any(item.code == "record_overlap_evidence_mismatch" for item in result.findings))

    def test_note_confirmation_does_not_clear_an_unknown_block(self) -> None:
        context = self._confirmed_note_context()
        self.recipe = PreparationRecipe(
            status="review_required", sheets=[self.recipe.sheets[0].model_copy(update={"role": "unknown"})],
            warnings=["Hay un bloque cuyo rol sigue sin resolver."], requires_human_review=True,
        )
        self.execution.sheets[0].role = "unknown"
        result = self.service.validate(
            self.summary, self.recipe, self.execution, identity_clarifications=context,
        )
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any(item.code == "unresolved_block_role" for item in result.findings))
        self.assertFalse(next(item for item in result.findings
                              if item.code == "source_record_identity_clarified").requires_review)

    def test_observation_notes_without_explicit_uncertainty_do_not_create_identity_review(self) -> None:
        self.summary.sheets[0].column_indices = [1, 2]
        for text in ("No hay duplicados.", "No hay\nduplicados."):
            with self.subTest(text=text):
                self.summary.sheets[0].sample = [SampleRow(row_number=3, values=["A", text])]
                self.assertEqual(self._validate().status, EvaluationStatus.READY)

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

    def test_independent_integrity_checks_all_run_and_keep_finding_order(self) -> None:
        self.summary.sheets[0].formula_count = 1
        self.summary.sheets[0].missing_formula_cache_count = 1
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="uncertain", kind="unknown", reason="Tratamiento sin confirmar.",
            start_row=2, end_row=4, start_column=1, end_column=2,
        )]
        self.execution.coverage = [SheetCoverage(
            sheet_name="Measurements", nonempty_cells=6, covered_nonempty_cells=6,
            uncovered_nonempty_cells=0, overlapping_nonempty_cells=1,
        )]
        self.execution.sheets[0] = self._execution_sheet(
            "Measurements", rows_after=2, conversion_failures={"value": 1},
            source_nonempty_cells=6, preserved_nonempty_cells=5, unresolved_formula_cells=["B2"],
        )

        result = self._validate()

        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertEqual([finding.code for finding in result.findings], [
            "source_missing_formula_cache", "overlapping_source_cells", "unexplained_row_loss",
            "conversion_failures", "nonempty_value_loss", "unresolved_formula_cache",
            "unresolved_structural_region",
        ])

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

    def test_model_review_flags_are_preserved_as_history_separate_from_current_findings(self) -> None:
        for flag in ("status", "requires_human_review"):
            with self.subTest(flag=flag):
                self.recipe.status = "review_required" if flag == "status" else "ready"
                self.recipe.requires_human_review = flag == "requires_human_review"
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.READY)
                self.assertEqual(result.original_proposal.status, self.recipe.status)
                self.assertEqual(result.original_proposal.requires_human_review, self.recipe.requires_human_review)
                self.assertEqual(result.review_reasons, [])

    def test_failed_recipe_cannot_be_overridden_by_successful_execution(self) -> None:
        self.recipe.status = "failed"
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(result.errors)

    def test_recipe_without_executable_proposal_keeps_review_status(self) -> None:
        self.recipe = PreparationRecipe(
            status="review_required", sheets=[], warnings=["No se pudieron localizar bloques."], requires_human_review=True,
        )
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
        self.recipe.sheets.append(SheetRecipe(
            sheet_name="Second", block_id="second", role="observations", header_rows=[],
            data_start_row=1, data_end_row=4, data_start_column=1, data_end_column=2, transformations=[],
        ))
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

    def test_empty_indices_and_counts_without_content_verification_cannot_clear_omission(self) -> None:
        self.summary.sheets.append(
            self._summary_sheet("Empty", rows=500, empty_rows=list(range(1, 101)), empty_row_count=500)
        )
        self.execution.skipped_sheets = ["Empty"]
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

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

    def test_total_separation_with_exact_coverage_preserves_information(self) -> None:
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="total", start_row=4, end_row=4, start_column=1, end_column=2,
            kind="total", reason="Etiqueta de total al final de las observaciones.",
        )]
        self.recipe.sheets = [
            SheetRecipe(
                sheet_name="Measurements", block_id="observations", role="observations", header_rows=[1],
                data_start_row=2, data_end_row=3, data_start_column=1, data_end_column=2, transformations=[],
            ),
            SheetRecipe(
                sheet_name="Measurements", block_id="total", role="total", header_rows=[1],
                data_start_row=4, data_end_row=4, data_start_column=1, data_end_column=2, transformations=[],
            ),
        ]
        self.execution.sheets = [
            self._execution_sheet(
                "Measurements", block_id="observations", data_end_row=3,
                rows_before=2, rows_after=2, excluded_nonempty_rows=1,
            ),
            self._execution_sheet(
                "Measurements", block_id="total", role="total", data_start_row=4,
                rows_before=1, rows_after=1, excluded_nonempty_rows=2,
            ),
        ]
        self.execution.coverage = [SheetCoverage(
            sheet_name="Measurements", nonempty_cells=8, covered_nonempty_cells=8,
            uncovered_nonempty_cells=0, overlapping_nonempty_cells=0,
        )]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.READY)
        self.assertEqual(result.findings, [])

    def test_mixing_total_or_legend_regions_with_observations_requires_review(self) -> None:
        for kind in ("total", "legend", "derived", "summary", "repeated_header"):
            with self.subTest(kind=kind):
                self.summary.sheets[0].regions = [StructuralRegion(
                    region_id="separate", start_row=4, end_row=4, start_column=1, end_column=2,
                    kind=kind, reason="Cambio estructural detectado.",
                )]
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
                finding = next(item for item in result.findings if item.code == "structural_role_conflict")
                self.assertEqual(finding.region_id, "separate")
                self.assertEqual(finding.range, "A2:B4")

    def test_candidate_initial_header_does_not_override_a_headerless_numeric_table(self) -> None:
        self.summary.sheets[0].initial_columns = ["1", "2"]
        self.summary.sheets[0].inferred_types = ["number", "number"]
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="first_row", start_row=1, end_row=1, start_column=1, end_column=2,
            kind="header", reason="Primera fila con valores; posible encabezado.",
            sample=[SampleRow(row_number=1, values=[1, 2])],
        )]
        self.recipe.sheets[0].header_rows = []
        self.recipe.sheets[0].data_start_row = 1
        self.execution.sheets[0].rows_before = 4
        self.execution.sheets[0].rows_after = 4
        self.execution.sheets[0].data_start_row = 1
        self.assertEqual(self._validate().status, EvaluationStatus.READY)

    def test_false_total_candidate_does_not_force_an_identical_heuristic_partition(self) -> None:
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="candidate", start_row=3, end_row=4, start_column=1, end_column=2,
            kind="total", reason="Candidato heurístico que debe contrastarse.",
            sample=[SampleRow(row_number=3, values=["sample_2", 0]),
                    SampleRow(row_number=4, values=["sample_3", 1])],
        )]
        self.assertEqual(self._validate().status, EvaluationStatus.READY)

    def test_observations_cannot_absorb_an_evidenced_initial_textual_header(self) -> None:
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="header", start_row=1, end_row=1, start_column=1, end_column=2,
            kind="header", reason="Cabecera textual respaldada por la muestra.",
            sample=[SampleRow(row_number=1, values=["location", "value"])],
        )]
        self.recipe.sheets[0].header_rows = []
        self.recipe.sheets[0].data_start_row = 1
        self.execution.sheets[0].data_start_row = 1
        self.execution.sheets[0].rows_before = 4
        self.execution.sheets[0].rows_after = 4
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any(finding.code == "structural_role_conflict" for finding in result.findings))

    def test_uncovered_cells_require_review_with_source_coordinate(self) -> None:
        self.execution.coverage = [SheetCoverage(
            sheet_name="Measurements", nonempty_cells=8, covered_nonempty_cells=7,
            uncovered_nonempty_cells=1, overlapping_nonempty_cells=0, uncovered_sample=["B4"],
        )]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any(finding.code == "uncovered_source_cells" and finding.cell == "B4" for finding in result.findings))

    def test_overlapping_data_coverage_or_inconsistent_counts_fail_validation(self) -> None:
        for changes in ({"overlapping_nonempty_cells": 1}, {"covered_nonempty_cells": 7}):
            with self.subTest(changes=changes):
                self.execution.coverage = [SheetCoverage(
                    sheet_name="Measurements", **{
                        "nonempty_cells": 8, "covered_nonempty_cells": 8,
                        "uncovered_nonempty_cells": 0, "overlapping_nonempty_cells": 0, **changes,
                    },
                )]
                self.assertEqual(self._validate().status, EvaluationStatus.FAILED)

    def test_infrastructure_findings_are_preserved_and_control_review(self) -> None:
        self.execution.findings = [TabularFinding(
            code="ambiguous_period", message="No se convirtió un período a una fecha exacta.",
            sheet_name="Measurements", cell="B2",
        )]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual(result.findings[0].cell, "B2")
        self.execution.findings[0].severity = "error"
        self.assertEqual(self._validate().status, EvaluationStatus.FAILED)

    def test_unknown_roles_and_unresolved_observation_regions_require_review(self) -> None:
        self.recipe.sheets[0].role = "unknown"
        self.execution.sheets[0].role = "unknown"
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)
        self.recipe.sheets[0].role = "observations"
        self.execution.sheets[0].role = "observations"
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="uncertain", start_row=2, end_row=4, start_column=1, end_column=2,
            kind="unknown", reason="No hay evidencia suficiente para asignar un rol.",
        )]
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_out_of_bounds_recipe_cannot_be_reported_as_ready(self) -> None:
        self.recipe.sheets[0].data_end_column = 3
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(any(finding.code == "block_out_of_bounds" for finding in result.findings))

    def test_execution_role_must_match_the_declared_role(self) -> None:
        self.execution.sheets[0].role = "legend"
        self.assertEqual(self._validate().status, EvaluationStatus.FAILED)

    def test_execution_provenance_must_match_the_recipe_coordinates(self) -> None:
        self.execution.sheets[0].data_start_row = 3
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertTrue(any(
            finding.code == "execution_range_mismatch" and finding.range == "A2:B4"
            for finding in result.findings
        ))

    def test_nonempty_value_loss_requires_review_even_without_row_loss(self) -> None:
        self.execution.sheets[0].source_nonempty_cells = 6
        self.execution.sheets[0].preserved_nonempty_cells = 5
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_renaming_explicit_units_cannot_silently_change_measurement_meaning(self) -> None:
        mappings = (
            {"colonias/mL": "células/mL"},
            {"abundance (%)": "abundance (cells/mL)"},
            {"concentration mg/L": "concentration"},
            {"value": "value cells/mL"},
            {"concentration mg/L": "concentration Mg/L"},
            {"abundance cells/mL": "abundance cells/ML"},
        )
        for mapping in mappings:
            with self.subTest(mapping=mapping):
                original, renamed = next(iter(mapping.items()))
                self.recipe.sheets[0].transformations = [RenameColumns(operation="rename_columns", mapping=mapping)]
                self.execution.sheets[0].columns_before = ["location", original]
                self.execution.sheets[0].columns_after = ["location", renamed]
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
                finding = next(item for item in result.findings if item.code == "unit_label_changed")
                self.assertEqual(finding.range, "A2:B4")

    def test_renaming_keeps_ordinary_names_and_equivalent_unit_labels_executable(self) -> None:
        mappings = (
            {"location": "place"},
            {"células / mL": "CELULAS/mL"},
            {"abundance %": "abundance percentage"},
            {"ID Datos Físicoquimicos/Recuento spp toxicas": "physicochemical_count_id"},
        )
        for mapping in mappings:
            with self.subTest(mapping=mapping):
                self.recipe.sheets[0].transformations = [RenameColumns(operation="rename_columns", mapping=mapping)]
                original, renamed = next(iter(mapping.items()))
                self.execution.sheets[0].columns_before = [original, "measurement"]
                self.execution.sheets[0].columns_after = [renamed, "measurement"]
                self.assertEqual(self._validate().status, EvaluationStatus.READY)

    def test_formula_and_error_counts_are_checked_when_samples_are_empty(self) -> None:
        self.execution.sheets[0].unresolved_formula_count = 1
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)
        self.execution.sheets[0].unresolved_formula_count = 0
        self.execution.sheets[0].error_cell_count = 1
        self.assertEqual(self._validate().status, EvaluationStatus.REVIEW_REQUIRED)

    def test_formula_missing_cache_in_header_requires_review_without_data_formula_failures(self) -> None:
        self.summary.sheets[0].formula_count = 1
        self.summary.sheets[0].missing_formula_cache_count = 1
        self.summary.sheets[0].formula_sample = [FormulaSample(cell="B1", formula="=SUM(B2:B4)")]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual(self.execution.sheets[0].unresolved_formula_count, 0)
        finding = next(item for item in result.findings if item.code == "source_missing_formula_cache")
        self.assertEqual(finding.cell, "B1")

    def test_missing_cache_outside_formula_sample_uses_exact_source_count(self) -> None:
        self.summary.sheets[0].formula_count = 20
        self.summary.sheets[0].missing_formula_cache_count = 1
        self.summary.sheets[0].formula_sample = [FormulaSample(
            cell="B2", formula="=0", cached_value=0, cache_available=True,
        )]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        finding = next(item for item in result.findings if item.code == "source_missing_formula_cache")
        self.assertIsNone(finding.cell)
        self.assertIn("fuente: 1.", finding.message)

    def test_excel_error_in_header_requires_review_without_data_error_cells(self) -> None:
        self.summary.sheets[0].error_cell_count = 1
        self.summary.sheets[0].error_cells = ["B1"]
        result = self._validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual(self.execution.sheets[0].error_cell_count, 0)
        finding = next(item for item in result.findings if item.code == "source_excel_error")
        self.assertEqual(finding.cell, "B1")

    def test_observation_region_cannot_be_silently_labeled_as_summary_or_derived(self) -> None:
        self.summary.sheets[0].regions = [StructuralRegion(
            region_id="counts", start_row=2, end_row=4, start_column=1, end_column=2,
            kind="observations", reason="Rango compatible con observaciones.",
        )]
        for role in ("summary", "derived"):
            with self.subTest(role=role):
                self.recipe.sheets[0].role = role
                self.execution.sheets[0].role = role
                result = self._validate()
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
                finding = next(
                    item for item in result.findings if item.code == "observations_in_nonobservational_block"
                )
                self.assertEqual(finding.region_id, "counts")
                self.assertEqual(finding.range, "A2:B4")


if __name__ == "__main__":
    unittest.main()
