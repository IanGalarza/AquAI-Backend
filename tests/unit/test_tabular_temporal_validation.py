import unittest
from copy import deepcopy

from pydantic import ValidationError

from aquai.application.dtos.tabular import (
    ExecutionResult, PreparationRecipe, SheetExecution, SheetSummary, StructuralSummary, TabularFinding,
)
from aquai.application.dtos.tabular_policies import (
    DateColumnPolicy, PreparationPolicies, SourceHeaderEvidence, TemporalDateTreatment,
)
from aquai.application.services.tabular_validation import TabularValidationService
from aquai.domain.enums.tabular import EvaluationStatus


def policy():
    return DateColumnPolicy(
        rule_id="user_temporal_priority", version="2", origin="user", sheet_name="records",
        column_indices=(2,), expected_headers={2: "Fecha", 3: "Código Fecha", 4: "Mes", 5: "Año"},
        text_formats=("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"),
        related_date_columns={"date_code": 3, "month": 4, "year": 5},
        related_date_code_formats=("%d%m%Y", "%m%Y"), temporal_priority="date_code_then_date",
    )


def fixture():
    summary = StructuralSummary(
        file_name="fixture.csv", file_type="csv", file_size_bytes=40,
        sheets=[SheetSummary(name="records", rows=2, columns=5,
                             initial_columns=["id", "Fecha", "Código Fecha", "Mes", "Año"],
                             inferred_types=["string"] * 5, null_counts=[0] * 5, sample=[])],
    )
    recipe = PreparationRecipe.model_validate({
        "status": "ready", "warnings": [], "requires_human_review": False,
        "sheets": [{"sheet_name": "records", "block_id": "records", "role": "observations",
                    "header_rows": [1], "data_start_row": 2, "data_end_row": 2,
                    "data_start_column": 1, "data_end_column": 5, "transformations": []}],
    })
    execution = ExecutionResult(
        sheets=[SheetExecution(sheet_name="records", block_id="records", source_rows=1, source_columns=5,
                               data_start_row=2, data_end_row=2, data_start_column=1, data_end_column=5,
                               rows_before=1, rows_after=1, columns_before=summary.sheets[0].initial_columns,
                               columns_after=summary.sheets[0].initial_columns)],
        source_headers=[SourceHeaderEvidence(sheet_name="records", row_number=1, inspected_columns=5,
                                             values=dict(enumerate(summary.sheets[0].initial_columns, 1)))],
    )
    return summary, recipe, execution


def treatment(**changes):
    values = {"date": "2004-12-03T00:00:00", "date_code": "03112004", "month": 12, "year": 2004}
    cells = {"date": "B2", "date_code": "C2", "month": "D2", "year": "E2"}
    result = {
        "sheet_name": "records", "block_id": "records", "source_row": 2, "date_cell": "B2",
        "normalized_date": "2004-11-03", "source_field": "date_code", "source_cell": "C2",
        "rule_id": "user_temporal_priority", "version": "2", "origin": "user",
        "original_values": values,
        "original_evidence": {field: {"cell": cells[field], "original_value": value}
                              for field, value in values.items()},
        "discrepancies": [{"code": "date_difference", "message": "Fecha y código difieren.",
                           "fields": ["date", "date_code"], "resolved_by_policy": True}],
        "resolution": "user_priority", "code_interpretation": "exact_8_digits",
    }
    result.update(changes)
    return TemporalDateTreatment.model_validate(result)


class TemporalDateContractTests(unittest.TestCase):
    def test_temporal_priority_requires_corrobated_full_code_format_and_defaults_off(self):
        values = policy().model_dump()
        values.pop("temporal_priority")
        self.assertIsNone(DateColumnPolicy.model_validate(values).temporal_priority)
        for changes in ({"related_date_code_formats": ("%m%Y",)},
                        {"related_date_columns": {"month": 4, "year": 5}}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                DateColumnPolicy.model_validate({**policy().model_dump(), **changes})

    def test_selected_date_requires_a_real_calendar_date_and_exact_original_provenance(self):
        for changes in ({"normalized_date": "2004-02-30"}, {"normalized_date": "20041103"},
                        {"source_cell": "B2"}, {"code_interpretation": "partial"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                treatment(**changes)

    def test_unresolved_date_cannot_carry_a_selection_or_resolved_discrepancy(self):
        with self.assertRaises(ValidationError):
            treatment(resolution="review_required")
        with self.assertRaises(ValidationError):
            treatment(resolution="review_required", normalized_date=None, source_field=None, source_cell=None)

    def test_empty_temporal_sidecar_does_not_change_existing_serialization(self):
        self.assertNotIn("temporal_dates", ExecutionResult().model_dump())
        self.assertIn("temporal_dates", ExecutionResult(temporal_dates=[treatment()]).model_dump())


class TemporalDateValidationTests(unittest.TestCase):
    def validate(self, execution, policies=None):
        summary, recipe, _ = fixture()
        return TabularValidationService().validate(summary, recipe, execution,
                                                  preparation_policies=policies or PreparationPolicies(date_columns=(policy(),)))

    def test_corrobated_priority_warning_can_be_ready_with_warnings_and_preserves_originals(self):
        _, _, execution = fixture()
        execution.temporal_dates = [treatment()]
        execution.findings = [TabularFinding(code="temporal_priority_discrepancies_resolved",
                                            message="Diferencia resuelta por política, sin confirmación científica.",
                                            requires_review=False)]
        originals = deepcopy(execution.temporal_dates[0].original_values)
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.review_reasons, [])
        self.assertEqual(execution.temporal_dates[0].original_values, originals)

    def test_valid_code_cannot_be_ignored_by_selecting_date_or_a_different_normalized_date(self):
        for changed in (treatment(normalized_date="2004-12-03", source_field="date", source_cell="B2"),
                        treatment(normalized_date="2004-11-04")):
            with self.subTest(changed=changed.normalized_date):
                _, _, execution = fixture()
                execution.temporal_dates = [changed]
                result = self.validate(execution)
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertIn("temporal_date_evidence_mismatch", [finding.code for finding in result.review_reasons])

    def test_invalid_full_code_does_not_override_a_valid_day_first_text_date(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        for field, original in (("date", "03/12/2004"), ("date_code", "31022004")):
            value["original_values"][field] = original
            value["original_evidence"][field]["original_value"] = original
        value.update(normalized_date="2004-12-03", source_field="date", source_cell="B2", code_interpretation="invalid")
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertNotIn("temporal_date_evidence_mismatch", [finding.code for finding in result.findings])

    def test_numeric_seven_digit_code_recovers_its_leading_zero_only_with_display_evidence(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        value["original_values"]["date_code"] = 3112004
        value["original_evidence"]["date_code"]["original_value"] = 3112004
        value["original_evidence"]["date_code"].update(source_kind="native_number", number_format="00000000")
        value["code_interpretation"] = "numeric_leading_zero"
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        self.assertNotIn("temporal_date_evidence_mismatch", [finding.code for finding in self.validate(execution).findings])
        self.assertEqual(execution.temporal_dates[0].original_values["date_code"], 3112004)

    def test_seven_digit_number_without_a_guarded_eight_digit_display_cannot_select_code(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        value["original_values"]["date_code"] = 3112004
        value["original_evidence"]["date_code"]["original_value"] = 3112004
        value["code_interpretation"] = "numeric_leading_zero"
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertIn("temporal_date_evidence_mismatch", [finding.code for finding in result.review_reasons])

    def test_seven_digit_code_without_display_evidence_falls_back_to_original_date(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        value["original_values"]["date_code"] = 3112004
        value["original_evidence"]["date_code"]["original_value"] = 3112004
        value.update(normalized_date="2004-12-03", source_field="date", source_cell="B2", code_interpretation="invalid")
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.review_reasons, [])

    def test_iso_timestamp_fallback_strips_only_the_parsing_input_and_preserves_original_spaces(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        original_date = "  2004-12-03T12:34:56Z  "
        for field, original in (("date", original_date), ("date_code", "122004")):
            value["original_values"][field] = original
            value["original_evidence"][field]["original_value"] = original
        value.update(normalized_date="2004-12-03", source_field="date", source_cell="B2", code_interpretation="partial")
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(execution.temporal_dates[0].original_values["date"], original_date)
        self.assertEqual(execution.temporal_dates[0].original_evidence["date"].original_value, original_date)

    def test_a_timestamp_outside_the_authorized_iso_grammar_does_not_validate_a_forged_selection(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        for field, original in (("date", "2004-12-03T12:34:56,5"), ("date_code", "122004")):
            value["original_values"][field] = original
            value["original_evidence"][field]["original_value"] = original
        value.update(normalized_date="2004-12-03", source_field="date", source_cell="B2", code_interpretation="partial")
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.FAILED)
        self.assertIn("temporal_date_evidence_mismatch", [finding.code for finding in result.review_reasons])

    def test_original_unresolved_engine_finding_remains_concrete_without_duplicate_reason(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        for field, original in (("date", "31/02/2004"), ("date_code", "31022004")):
            value["original_values"][field] = original
            value["original_evidence"][field]["original_value"] = original
        value.update(normalized_date=None, source_field=None, source_cell=None,
                     code_interpretation="invalid", resolution="review_required", discrepancies=[])
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        original = TabularFinding(code="temporal_date_unresolved", sheet_name="records", block_id="records",
                                  cell="B2", message="B2=31/02/2004 y C2=31022004 son fechas inválidas.")
        execution.findings = [original]
        result = self.validate(execution)
        self.assertEqual(result.review_reasons, [original])

    def test_no_valid_code_or_date_requires_review_even_if_engine_did_not_emit_a_finding(self):
        _, _, execution = fixture()
        value = treatment().model_dump()
        for field, original in (("date", "31/02/2004"), ("date_code", "31022004")):
            value["original_values"][field] = original
            value["original_evidence"][field]["original_value"] = original
        value.update(normalized_date=None, source_field=None, source_cell=None,
                     code_interpretation="invalid", resolution="review_required", discrepancies=[])
        execution.temporal_dates = [TemporalDateTreatment.model_validate(value)]
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual([finding.code for finding in result.review_reasons], ["temporal_date_unresolved"])

    def test_missing_or_foreign_rule_trace_cannot_be_used_to_remove_review(self):
        for traces in ([], [treatment(version="unknown")]):
            with self.subTest(traces=len(traces)):
                _, _, execution = fixture()
                execution.temporal_dates = traces
                result = self.validate(execution)
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertIn("temporal_date_evidence_missing", [finding.code for finding in result.review_reasons])

    def test_unrelated_review_conflict_is_preserved_after_successful_temporal_selection(self):
        _, _, execution = fixture()
        execution.temporal_dates = [treatment()]
        execution.findings = [TabularFinding(code="identifier_conflict", message="Dos identificadores incompatibles.")]
        result = self.validate(execution)
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertEqual([finding.code for finding in result.review_reasons], ["identifier_conflict"])

