import unittest
from copy import deepcopy
from datetime import datetime

import pandas as pd

from aquai.application.contracts.tabular.results import AssociationDiagnostic
from aquai.application.contracts.tabular.policies import DateColumnPolicy, PreparationPolicies
from aquai.infrastructure.processing.tabular.dateConsistency import date_consistency_findings
from aquai.infrastructure.processing.tabular.reader import CellMetadata, SheetData


class DateConsistencyTests(unittest.TestCase):
    def sheet(self, name, *rows):
        return SheetData(name=name, frame=pd.DataFrame(
            [["id", "Fecha", "Mes", "Año", "Código fecha", "Día"], *rows], dtype=object,
        ))

    def policy(self, name, **overrides):
        definition = dict(
            rule_id=f"dates_{name}", version="1.0.0", origin="confirmed_dates",
            sheet_name=name, column_indices=(2,),
            expected_headers={2: "Fecha", 3: "Mes", 4: "Año", 5: "Código fecha", 6: "Día"},
            text_formats=("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"),
            related_date_columns={"month": 3, "year": 4, "date_code": 5},
            related_date_code_formats=("%d%m%Y", "%m%Y"),
        )
        definition.update(overrides)
        return DateColumnPolicy(**definition)

    def association(self, source="observations", target="environment", **overrides):
        definition = dict(source_sheet=source, source_row=2, source_cell="A2", source_value="sample",
                          target_sheet=target, target_rows=(2,), status="matched", requires_review=False,
                          rule_id="link", version="1.0.0", origin="confirmed_links")
        definition.update(overrides)
        return AssociationDiagnostic(**definition)

    def evaluate(self, sheets, associations=(), **overrides):
        policies = PreparationPolicies(date_columns=tuple(self.policy(sheet.name, **overrides) for sheet in sheets))
        return date_consistency_findings(sheets, policies, list(associations))

    def test_authorized_dmy_with_small_day_and_month_and_iso_are_same_date(self):
        source = self.sheet("observations", ["sample", "03-12-2004", "diciembre", 2004, "03122004", None])
        target = self.sheet("environment", ["sample", "2004-12-03", "December", 2004, "03122004", None])
        self.assertEqual(self.evaluate([source, target], [self.association()]), [])
        self.assertEqual(source.frame.iat[1, 1], "03-12-2004")

    def test_native_date_is_not_inverted_to_match_month_and_code(self):
        source = self.sheet("arbitrary_sheet", ["sample", datetime(2004, 3, 12), "diciembre", 2004, "03122004", None])
        source.cell_metadata[(2, 2)] = CellMetadata(source_kind="native_datetime", number_format="mm/dd/yyyy", raw_value="38058")
        before = source.frame.copy(deep=True)
        findings = self.evaluate([source])
        self.assertEqual([(item.code, item.cell) for item in findings], [("date_components_conflict", "B2")])
        self.assertIn("2004-03-12T00:00:00", findings[0].message)
        self.assertIn("mm/dd/yyyy", findings[0].message)
        self.assertIn("38058", findings[0].message)
        self.assertIn("C2", findings[0].message)
        self.assertIn("E2", findings[0].message)
        self.assertTrue(findings[0].requires_review)
        pd.testing.assert_frame_equal(source.frame, before)

    def test_month_conflict_is_located_even_when_full_date_code_agrees(self):
        source = self.sheet("data", ["sample", datetime(2015, 11, 3), "septiembre", 2015, "03112015", None])
        findings = self.evaluate([source])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].cell, "B2")
        self.assertIn("C2 Mes='septiembre'", findings[0].message)
        self.assertNotIn("E2", findings[0].message)

    def test_partial_month_year_code_does_not_assert_a_sampling_day(self):
        source = self.sheet("observations", ["sample", datetime(2005, 2, 1), "febrero", 2005, "022005", None])
        target = self.sheet("environment", ["sample", datetime(2005, 2, 22), "February", 2005, "22022005", None])
        self.assertEqual(self.evaluate([source, target], [self.association()]), [])
        self.assertEqual(source.frame.iat[1, 4], "022005")

    def test_six_digit_numeric_month_year_is_not_padded_to_full_date(self):
        source = self.sheet("data", ["sample", datetime(2015, 11, 3), "noviembre", 2015, 112015, None])
        self.assertEqual(self.evaluate([source]), [])

    def test_numeric_full_date_code_can_retain_a_lost_leading_zero(self):
        source = self.sheet("data", ["sample", datetime(2004, 12, 3), "diciembre", 2004, 3122004, None])
        self.assertEqual(self.evaluate([source]), [])
        self.assertEqual(source.frame.iat[1, 4], 3122004)

    def test_native_linked_dates_disagree_with_original_values_and_coordinates(self):
        source = self.sheet("observations", ["sample", datetime(2004, 12, 3), "diciembre", 2004, "03122004", None])
        target = self.sheet("environment", ["sample", datetime(2004, 3, 12), "March", 2004, "12032004", None])
        findings = self.evaluate([source, target], [self.association()])
        self.assertEqual([(item.code, item.cell) for item in findings], [("linked_date_conflict", "B2")])
        self.assertIn("observations!B2", findings[0].message)
        self.assertIn("environment!B2", findings[0].message)
        self.assertIn("2004-12-03", findings[0].message)
        self.assertIn("2004-03-12", findings[0].message)
        self.assertTrue(findings[0].requires_review)

    def test_iso_and_native_same_date_do_not_require_review(self):
        source = self.sheet("observations", ["sample", "2021-03-19T00:00:00", "March", 2021, "19032021", None])
        target = self.sheet("environment", ["sample", datetime(2021, 3, 19), "marzo", 2021, "19032021", None])
        self.assertEqual(self.evaluate([source, target], [self.association()]), [])

    def test_single_configured_date_fields_can_have_different_arbitrary_headers(self):
        source = self.sheet("observations", ["sample", "03/12/2004", "diciembre", 2004, "03122004", None])
        target = self.sheet("environment", ["sample", "12/03/2004", "marzo", 2004, "12032004", None])
        source.frame.iat[0, 1] = "Sampling date"
        target.frame.iat[0, 1] = "Fecha observada"
        source_headers = dict(self.policy(source.name).expected_headers)
        target_headers = dict(self.policy(target.name).expected_headers)
        source_headers[2], target_headers[2] = "Sampling date", "Fecha observada"
        policies = PreparationPolicies(date_columns=(
            self.policy(source.name, expected_headers=source_headers),
            self.policy(target.name, expected_headers=target_headers),
        ))
        findings = date_consistency_findings([source, target], policies, [self.association()])
        self.assertEqual([finding.code for finding in findings], ["linked_date_conflict"])

    def test_missing_optional_or_ambiguous_association_does_not_compare_dates(self):
        source = self.sheet("observations", ["sample", "03/12/2004", "diciembre", 2004, "03122004", None])
        target = self.sheet("environment", ["sample", "12/03/2004", "marzo", 2004, "12032004", None])
        for diagnostic in (
            self.association(status="missing", target_rows=()),
            self.association(status="ambiguous", target_rows=(2, 3), requires_review=True),
        ):
            with self.subTest(status=diagnostic.status):
                self.assertEqual(self.evaluate([source, target], [diagnostic]), [])

    def test_header_guards_prevent_applying_the_wrong_component_policy(self):
        source = self.sheet("data", ["sample", datetime(2004, 3, 12), "diciembre", 2004, "03122004", None])
        source.frame.iat[0, 2] = "Unrelated category"
        findings = self.evaluate([source])
        self.assertEqual([item.code for item in findings], ["date_consistency_policy_mismatch"])

    def test_related_year_and_day_conflicts_are_one_finding_per_date_cell(self):
        source = self.sheet("data", ["sample", datetime(2020, 1, 10), "Jan.", 2021, "10012020", 11])
        findings = self.evaluate([source], related_date_columns={"month": 3, "year": 4, "day": 6, "date_code": 5})
        self.assertEqual(len(findings), 1)
        self.assertIn("D2", findings[0].message)
        self.assertIn("F2", findings[0].message)

    def test_unformatted_serial_does_not_become_a_native_date(self):
        source = self.sheet("data", ["sample", 38058, "diciembre", 2004, "03122004", None])
        self.assertEqual(self.evaluate([source]), [])
        self.assertEqual(source.frame.iat[1, 1], 38058)

    def test_duplicate_policies_and_diagnostics_do_not_duplicate_findings(self):
        source = self.sheet("observations", ["sample", datetime(2004, 3, 12), "diciembre", 2004, "03122004", None])
        target = self.sheet("environment", ["sample", datetime(2004, 12, 3), "December", 2004, "03122004", None])
        policy = self.policy(source.name)
        policies = PreparationPolicies(date_columns=(policy, deepcopy(policy), self.policy(target.name)))
        diagnostic = self.association()
        findings = date_consistency_findings([source, target], policies, [diagnostic, diagnostic])
        self.assertEqual([item.code for item in findings], ["date_components_conflict", "linked_date_conflict"])

    def test_no_policies_does_not_expand_confirmation_to_other_profiles(self):
        source = self.sheet("unconfigured", ["sample", datetime(2004, 3, 12), "diciembre", 2004, "03122004", None])
        self.assertEqual(date_consistency_findings([source], PreparationPolicies(), []), [])


if __name__ == "__main__":
    unittest.main()
