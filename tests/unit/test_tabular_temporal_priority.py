import io
import unittest
from datetime import date, datetime

import pandas as pd
from openpyxl import Workbook

from aquai.application.dtos.tabular import PreparationRecipe, SheetRecipe
from aquai.application.dtos.tabular_policies import AssociationDiagnostic, DateColumnPolicy, PreparationPolicies
from aquai.infrastructure.processing.tabular.date_consistency import date_consistency_findings
from aquai.infrastructure.processing.tabular.processor import PandasTabularProcessor
from aquai.infrastructure.processing.tabular.reader import CellMetadata, SheetData
from aquai.infrastructure.processing.tabular.temporal_priority import select_temporal_dates


class TemporalPriorityTests(unittest.TestCase):
    def sheet(self, *rows, name="observations"):
        return SheetData(name=name, frame=pd.DataFrame(
            [["ID", "Fecha", "Mes", "Año", "Código Fecha", "link"], *rows], dtype=object,
        ))

    def policy(self, name="observations", **updates):
        values = dict(
            rule_id="user_temporal_priority", version="2.0.0", origin="user_confirmation",
            sheet_name=name, column_indices=(2,), expected_headers={2: "Fecha", 3: "Mes", 4: "Año", 5: "Código Fecha"},
            text_formats=("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d"),
            related_date_columns={"month": 3, "year": 4, "date_code": 5},
            related_date_code_formats=("%d%m%Y", "%m%Y"), temporal_priority="date_code_then_date",
        )
        values.update(updates)
        return DateColumnPolicy(**values)

    def recipe(self, sheet, **updates):
        values = dict(sheet_name=sheet.name, block_id="observations", role="observations", header_rows=[1],
                      data_start_row=2, data_end_row=len(sheet.frame), data_start_column=1, data_end_column=6, transformations=[])
        values.update(updates)
        return SheetRecipe(**values)

    def select(self, sheet, policy=None, recipe=None, rows=None):
        return select_temporal_dates(
            sheet, recipe or self.recipe(sheet),
            PreparationPolicies(date_columns=(policy or self.policy(sheet.name),)),
            rows if rows is not None else list(range(2, len(sheet.frame) + 1)),
        )

    def test_complete_code_has_priority_without_inverting_native_date(self):
        sheet = self.sheet(["unchanged-ID", datetime(2004, 3, 12), "marzo", 2004, "03122004", "unchanged-link"])
        sheet.cell_metadata[(2, 2)] = CellMetadata("native_datetime", "mm/dd/yyyy", "38058", "1899-12-30")
        before = sheet.frame.copy(deep=True)
        treatment, findings = self.select(sheet)
        chosen = treatment[0]
        self.assertEqual((chosen.normalized_date, chosen.source_field, chosen.source_cell), ("2004-12-03", "date_code", "E2"))
        self.assertEqual(chosen.original_values["date"], "2004-03-12T00:00:00")
        self.assertEqual(chosen.original_evidence["date"].raw_value, "38058")
        self.assertEqual(chosen.original_values["date_code"], "03122004")
        self.assertEqual({item.code for item in chosen.discrepancies}, {"date_field_disagrees", "month_disagrees"})
        self.assertTrue(all(item.resolved_by_policy for item in chosen.discrepancies))
        self.assertFalse(findings[0].requires_review)
        self.assertIn("no una confirmación científica", findings[0].message)
        pd.testing.assert_frame_equal(sheet.frame, before)

    def test_leading_zeros_and_original_whitespace_remain_in_evidence(self):
        sheet = self.sheet(["id", None, None, None, " 01022024 ", "link"])
        selected, findings = self.select(sheet)
        self.assertEqual(selected[0].normalized_date, "2024-02-01")
        self.assertEqual(selected[0].original_values["date_code"], " 01022024 ")
        self.assertEqual(selected[0].code_interpretation, "exact_8_digits")
        self.assertEqual(findings, [])

    def test_code_requires_calendar_validity_and_leap_year_rules(self):
        for code, expected, field in (("29022000", "2000-02-29", "date_code"),
                                      ("29021900", "2024-03-01", "date"),
                                      ("30022024", "2024-03-01", "date")):
            with self.subTest(code=code):
                selected, findings = self.select(self.sheet(["id", "01/03/2024", "marzo", 2024, code, None]))
                self.assertEqual((selected[0].normalized_date, selected[0].source_field), (expected, field))
                self.assertTrue(all(not item.requires_review for item in findings))

    def test_code_valid_and_invalid_fallback_date_is_resolved_warning(self):
        selected, findings = self.select(self.sheet(["id", "31/02/2024", "febrero", 2024, "01022024", None]))
        self.assertEqual(selected[0].normalized_date, "2024-02-01")
        self.assertEqual([item.code for item in selected[0].discrepancies], ["date_field_invalid"])
        self.assertFalse(findings[0].requires_review)

    def test_invalid_code_falls_back_to_explicit_dmy_and_retains_original(self):
        selected, findings = self.select(self.sheet(["id", "03-12-2004", "diciembre", 2004, "31132004", None]))
        self.assertEqual((selected[0].normalized_date, selected[0].source_field, selected[0].source_cell), ("2004-12-03", "date", "B2"))
        self.assertEqual(selected[0].original_values["date"], "03-12-2004")
        self.assertEqual(selected[0].code_interpretation, "invalid")
        self.assertFalse(findings[0].requires_review)

    def test_native_iso_and_dmy_fallback_are_never_inverted(self):
        for original, expected in ((datetime(2004, 3, 12), "2004-03-12"), (date(2004, 3, 12), "2004-03-12"),
                                   ("2004-03-12T14:30:00", "2004-03-12"), ("2004-03-12", "2004-03-12"),
                                   ("03/12/2004", "2004-12-03"), ("03.12.2004", "2004-12-03")):
            with self.subTest(original=original):
                sheet = self.sheet(["id", original, "diciembre", 2004, None, None])
                selected, findings = self.select(sheet)
                self.assertEqual(selected[0].normalized_date, expected)
                self.assertEqual(selected[0].source_field, "date")
                self.assertEqual(sheet.frame.iat[1, 1], original)
                self.assertTrue(all(not item.requires_review for item in findings))

    def test_partial_month_year_code_uses_full_date_without_fabricating_day(self):
        for code in ("022005", 22005, "112015", 112015):
            with self.subTest(code=code):
                selected, findings = self.select(self.sheet(["id", "22/02/2005", "febrero", 2005, code, None]))
                self.assertEqual((selected[0].normalized_date, selected[0].source_field), ("2005-02-22", "date"))
                self.assertEqual(selected[0].code_interpretation, "partial")
                self.assertFalse(findings[0].requires_review)

    def test_missing_or_invalid_candidates_never_use_month_year_as_date(self):
        for date_value, code in ((None, None), ("31/02/2024", "30022024"), (None, "022024"), (45323, "invalid")):
            with self.subTest(date_value=date_value, code=code):
                selected, findings = self.select(self.sheet(["id", date_value, "febrero", 2024, code, None]))
                self.assertEqual(selected[0].resolution, "review_required")
                self.assertIsNone(selected[0].normalized_date)
                self.assertIsNone(selected[0].source_field)
                self.assertTrue(findings[0].requires_review)
                self.assertTrue(all(not item.resolved_by_policy for item in selected[0].discrepancies))

    def test_partial_code_contrast_records_only_real_month_year_disagreement(self):
        for code, expected_codes in (("022005", {"date_code_partial"}),
                                     ("112015", {"date_code_partial", "date_code_partial_disagrees"})):
            with self.subTest(code=code):
                sheet = self.sheet(["id", "22/02/2005", "febrero", 2005, code, None])
                selected, findings = self.select(sheet)
                self.assertEqual(selected[0].normalized_date, "2005-02-22")
                self.assertEqual(selected[0].source_field, "date")
                self.assertEqual({item.code for item in selected[0].discrepancies}, expected_codes)
                self.assertEqual(selected[0].original_values["date_code"], code)
                self.assertEqual(selected[0].original_values["date"], "22/02/2005")
                self.assertTrue(all(item.resolved_by_policy for item in selected[0].discrepancies))
                self.assertFalse(findings[0].requires_review)

    def test_numeric_seven_digits_require_evidence_of_eight_digit_display(self):
        for number_format, expected, field, interpretation in (("00000000", "2004-12-03", "date_code", "numeric_leading_zero"),
                                                             ("General", "2004-03-12", "date", "invalid")):
            with self.subTest(number_format=number_format):
                sheet = self.sheet(["id", datetime(2004, 3, 12), "marzo", 2004, 3122004, None])
                sheet.cell_metadata[(2, 5)] = CellMetadata("native_number", number_format)
                selected, _ = self.select(sheet)
                self.assertEqual((selected[0].normalized_date, selected[0].source_field, selected[0].code_interpretation),
                                 (expected, field, interpretation))
                self.assertEqual(selected[0].original_values["date_code"], 3122004)

    def test_seven_digit_text_is_not_padded_or_mutated(self):
        selected, _ = self.select(self.sheet(["id", "12/03/2004", "marzo", 2004, "3122004", None]))
        self.assertEqual((selected[0].normalized_date, selected[0].source_field), ("2004-03-12", "date"))
        self.assertEqual(selected[0].original_values["date_code"], "3122004")

    def test_numeric_eight_digits_are_valid_but_boolean_negative_fraction_are_not(self):
        for code, field in ((12032004, "date_code"), (12032004.0, "date_code"), (True, "date"),
                            (-12032004, "date"), (12032004.5, "date")):
            with self.subTest(code=code):
                selected, _ = self.select(self.sheet(["id", "12/03/2004", "marzo", 2004, code, None]))
                self.assertEqual(selected[0].source_field, field)
                self.assertEqual(selected[0].original_values["date_code"], code)

    def test_unknown_or_conflicting_month_year_are_only_contrast(self):
        selected, findings = self.select(self.sheet(["id", "03/12/2004", "unknown-month", 2005, "03122004", None]))
        self.assertEqual(selected[0].normalized_date, "2004-12-03")
        self.assertEqual({item.code for item in selected[0].discrepancies}, {"month_disagrees", "year_disagrees"})
        self.assertFalse(findings[0].requires_review)

    def test_only_retained_observational_source_rows_receive_selection(self):
        sheet = self.sheet(["id-1", "03/12/2004", None, None, "03122004", None],
                           ["id-2", "12/03/2004", None, None, "12032004", None])
        selected, _ = self.select(sheet, rows=[3])
        self.assertEqual([(item.source_row, item.block_id) for item in selected], [(3, "observations")])
        for role in ("legend", "header", "derived", "unknown"):
            self.assertEqual(self.select(sheet, recipe=self.recipe(sheet, role=role, header_rows=[] if role == "header" else [1])), ([], []))

    def test_header_guard_and_disabled_policy_cannot_apply_priority(self):
        sheet = self.sheet(["id", "12/03/2004", None, None, "03122004", None])
        self.assertEqual(self.select(sheet, policy=self.policy(temporal_priority=None)), ([], []))
        sheet.frame.iat[0, 4] = "Other code"
        self.assertEqual(self.select(sheet), ([], []))

    def linked(self, source, target):
        selected = [*self.select(source)[0], *self.select(target)[0]]
        policies = PreparationPolicies(date_columns=(self.policy(source.name), self.policy(target.name)))
        association = AssociationDiagnostic(source_sheet=source.name, source_row=2, source_cell="F2", source_value="same-link",
                                            target_sheet=target.name, target_rows=(2,), rule_id="optional_link", version="1", origin="user",
                                            status="matched", requires_review=False)
        return date_consistency_findings([source, target], policies, [association], temporal_dates=selected)

    def test_associated_original_difference_is_retained_as_resolved_warning(self):
        source = self.sheet(["id", datetime(2004, 12, 3), "diciembre", 2004, "03122004", "same-link"])
        target = self.sheet(["id", datetime(2004, 3, 12), "marzo", 2004, "03122004", "same-link"], name="environment")
        findings = self.linked(source, target)
        self.assertEqual([item.code for item in findings], ["linked_date_discrepancy_resolved"])
        self.assertFalse(findings[0].requires_review)
        self.assertIn("2004-03-12", findings[0].message)
        self.assertIn("2004-12-03", findings[0].message)
        self.assertIn("E2", findings[0].message)

    def test_incompatible_chosen_dates_of_linked_records_still_require_review(self):
        source = self.sheet(["id", "03/12/2004", None, None, "03122004", "same-link"])
        target = self.sheet(["id", "03/12/2004", None, None, "12032004", "same-link"], name="environment")
        findings = self.linked(source, target)
        self.assertEqual([item.code for item in findings], ["linked_date_conflict"])
        self.assertTrue(findings[0].requires_review)

    def test_partial_codes_do_not_remove_the_day_of_selected_fallback_for_links(self):
        source = self.sheet(["id", "01/02/2005", None, None, "022005", "same-link"])
        target = self.sheet(["id", "22/02/2005", None, None, "022005", "same-link"], name="environment")
        findings = self.linked(source, target)
        self.assertEqual([item.code for item in findings], ["linked_date_conflict"])
        self.assertTrue(findings[0].requires_review)

    def test_processor_keeps_normalization_and_id_link_values_and_resolves_invalid_fallback(self):
        book = Workbook()
        sheet = book.active
        sheet.title = "arbitrary-data"
        sheet.append(["ID", "Fecha", "Mes", "Año", "Código Fecha", "link"])
        sheet.append(["immutable-ID", "31/02/2024", "febrero", 2024, "01022024", "immutable-link"])
        sheet.append(["other-ID", "03-12-2004", "diciembre", 2004, "03122004", "other-link"])
        binary = io.BytesIO()
        book.save(binary)
        content = binary.getvalue()
        processor = PandasTabularProcessor()
        summary = processor.inspect("arbitrary.xlsx", content)
        plan = PreparationRecipe(status="ready", sheets=[self.recipe(self.sheet(name=sheet.title), data_end_row=3)],
                                 warnings=[], requires_human_review=False)
        result = processor.execute("arbitrary.xlsx", content, plan, summary,
                                   policies=PreparationPolicies(date_columns=(self.policy(sheet.title),)))
        self.assertEqual([item.normalized_date for item in result.temporal_dates], ["2024-02-01", "2004-12-03"])
        self.assertEqual([(item.cell, item.original_value, item.prepared_value) for item in result.normalizations],
                         [("B3", "03-12-2004", "2004-12-03")])
        self.assertFalse(any(item.requires_review for item in result.findings))
        self.assertNotIn("date_text_unresolved", [item.code for item in result.findings])
        self.assertEqual(result.sheets[0].sample[0].values, ["immutable-ID", "31/02/2024", "febrero", 2024, "01022024", "immutable-link"])
        self.assertEqual(result.sheets[0].rows_after, 2)


if __name__ == "__main__":
    unittest.main()
