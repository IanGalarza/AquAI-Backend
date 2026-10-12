"""Preparación autorizada determinística, sin inferencias ni cambios en fuentes."""

import io
import unittest
from datetime import datetime
from hashlib import sha256
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook

from aquai.application.dtos.tabular import PreparationRecipe
from aquai.application.dtos.tabularPolicies import (
    AssociationPolicy,
    ConfirmedCellPolicy,
    DateColumnPolicy,
    NumberColumnPolicy,
    ObservationRowPolicy,
    PreparationPolicies,
)
from aquai.infrastructure.processing.tabular import PandasTabularProcessor
from aquai.infrastructure.processing.tabular.normalization import parse_explicit_date, parse_mixed_decimal


RULE = {"rule_id": "user_confirmed", "version": "2.0.0", "origin": "user_confirmation"}
DATE_FORMATS = ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d")


def workbook_bytes(workbook):
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def observation_recipe(sheet, end_row, end_column, transformations=None):
    return PreparationRecipe.model_validate({
        "status": "ready", "requires_human_review": False, "warnings": [], "sheets": [{
            "sheet_name": sheet, "block_id": "records", "role": "observations", "header_rows": [1],
            "data_start_row": 2, "data_end_row": end_row, "data_start_column": 1,
            "data_end_column": end_column, "transformations": transformations or [],
        }],
    })


class ExplicitParsingTests(unittest.TestCase):
    def test_explicit_argentine_dates_and_iso_do_not_swap_month_and_day(self):
        for original, expected in (("03-12-2004", "2004-12-03"), ("03/04/2004", "2004-04-03"),
                                   ("3.4.2004", "2004-04-03"), ("2004-12-03", "2004-12-03")):
            with self.subTest(original=original):
                parsed = parse_explicit_date(original, DATE_FORMATS)
                self.assertEqual(parsed.value, expected)
                self.assertIsNone(parsed.conflict)
        native = datetime(2023, 5, 2)
        parsed = parse_explicit_date(native, DATE_FORMATS)
        self.assertIs(parsed.value, native)
        self.assertFalse(parsed.changed)
        self.assertIsNone(parsed.conflict)

    def test_invalid_dates_and_unproved_serials_keep_the_original(self):
        for original in ("31/02/2024", "2024-02-31", 45048):
            with self.subTest(original=original):
                parsed = parse_explicit_date(original, DATE_FORMATS)
                self.assertEqual(parsed.value, original)
                self.assertFalse(parsed.changed)
                self.assertIsNotNone(parsed.conflict)

    def test_decimal_dot_comma_and_unambiguous_grouping_preserve_magnitude(self):
        for original, expected in (("2.5", 2.5), ("2,5", 2.5), ("0,001", 0.001),
                                   ("0", 0), ("1.234,56", 1234.56), ("1,234.56", 1234.56)):
            with self.subTest(original=original):
                parsed = parse_mixed_decimal(original)
                self.assertEqual(parsed.value, expected)
                self.assertIsNone(parsed.conflict)

    def test_ambiguous_thousands_and_precision_loss_are_localized_without_guessing(self):
        for original in ("1.234", "1,234", "12.345.678", "2.5000000000000000001"):
            with self.subTest(original=original):
                parsed = parse_mixed_decimal(original)
                self.assertEqual(parsed.value, original)
                self.assertFalse(parsed.changed)
                self.assertIsNotNone(parsed.conflict)

    def test_native_numbers_zero_and_categories_are_not_reinterpreted(self):
        for original in (0, 2.5, 1000, False, "nd", "n.d.", "<0,001", "---", "baja", "+++", "2004-12-03"):
            with self.subTest(original=original):
                parsed = parse_mixed_decimal(original)
                self.assertEqual(parsed.value, original)
                self.assertFalse(parsed.changed)
                self.assertIsNone(parsed.conflict)


class ApprovedPreparationTests(unittest.TestCase):
    def setUp(self):
        self.processor = PandasTabularProcessor()

    def execute(self, content, recipe, policies):
        summary = self.processor.inspect("arbitrary.xlsx", content)
        return self.processor.execute("arbitrary.xlsx", content, recipe, summary, policies=policies)

    def test_normalization_and_auxiliary_rows_preserve_markers_other_modalities_and_codes(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["id", "date", "abundance", "category", "link", "code"])
        sheet.append(["S1", "03-12-2004", "2,5", "+++", "missing", "001"])
        sheet.append(["S2", datetime(2023, 5, 2), 0, "baja", "found", "002"])
        sheet.append(['=B4&"_"&C4', "2004-12-03", "nd", "other modality", "found", "003"])
        sheet.append(['=B5&"_"&C5'])
        content = workbook_bytes(workbook)
        recipe = observation_recipe("Samples", 5, 6)
        original_recipe = recipe.model_dump_json()
        policies = PreparationPolicies(
            date_columns=(DateColumnPolicy(**RULE, sheet_name="Samples", column_indices=(2,),
                                          expected_headers={2: "date"}, text_formats=DATE_FORMATS),),
            number_columns=(NumberColumnPolicy(**RULE, sheet_name="Samples", column_indices=(3,),
                                              expected_headers={3: "abundance"}),),
            observation_rows=(ObservationRowPolicy(**RULE, sheet_name="Samples", auxiliary_column_indices=(1,),
                                                  expected_headers={1: "id"}),),
        )
        execution = self.execute(content, recipe, policies)
        block = execution.sheets[0]
        self.assertEqual((block.rows_before, block.rows_after, block.auxiliary_rows_removed), (4, 3, 1))
        self.assertEqual([sample.row_number for sample in block.sample], [2, 3, 4])
        self.assertEqual(block.sample[0].values, ["S1", "2004-12-03", 2.5, "+++", "missing", "001"])
        self.assertEqual(block.sample[1].values[1:4], ["2023-05-02T00:00:00", 0, "baja"])
        self.assertEqual(block.sample[2].values[2:], ["nd", "other modality", "found", "003"])
        self.assertEqual((block.formula_count, block.unresolved_formula_count), (2, 1))
        self.assertEqual(block.unresolved_formula_cells, ["A4"])
        self.assertEqual({item.cell for item in block.formula_sample}, {"A4", "A5"})
        trace = {item.cell: item for item in execution.normalizations}
        self.assertEqual(set(trace), {"B2", "C2"})
        self.assertEqual((trace["B2"].original_value, trace["B2"].prepared_value), ("03-12-2004", "2004-12-03"))
        self.assertEqual((trace["C2"].original_value, trace["C2"].prepared_value), ("2,5", 2.5))
        self.assertEqual((trace["C2"].source_kind, trace["C2"].number_format), ("text", "General"))
        auxiliary = execution.auxiliary_rows[0]
        self.assertEqual((auxiliary.row_number, auxiliary.inspected_columns), (5, 6))
        self.assertEqual((auxiliary.auxiliary_cells[0].cell, auxiliary.auxiliary_cells[0].formula), ("A5", '=B5&"_"&C5'))
        self.assertEqual(recipe.model_dump_json(), original_recipe)
        raw = self.processor.reader.read("arbitrary.xlsx", content).sheets[0]
        self.assertEqual(raw.frame.iat[1, 1], "03-12-2004")
        self.assertEqual(raw.frame.iat[1, 2], "2,5")
        self.assertIn((5, 1), raw.formula_cells)

    def test_auxiliary_formula_rows_with_zero_markers_categories_or_other_values_remain(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["id", "abundance", "relative", "other modality"])
        for row, value in enumerate((0, "nd", "n.d.", "<0,001", "+++", " ", 5), 2):
            sheet.append([f'=B{row}&"_"&C{row}', None, None, value])
        content = workbook_bytes(workbook)
        policy = ObservationRowPolicy(**RULE, sheet_name="Samples", auxiliary_column_indices=(1,), expected_headers={1: "id"})
        execution = self.execute(content, observation_recipe("Samples", 8, 4), PreparationPolicies(observation_rows=(policy,)))
        self.assertEqual(execution.sheets[0].rows_after, 7)
        self.assertEqual(execution.auxiliary_rows, [])

    def test_separate_auxiliary_blocks_keep_formulas_without_active_cache_conflict(self):
        for role in ("derived", "legend", "unknown"):
            with self.subTest(role=role):
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Samples"
                sheet.append(["id", "abundance"])
                sheet.append(['=B2&"_"&C2'])
                sheet.append(['=B3&"_"&C3'])
                content = workbook_bytes(workbook)
                raw_recipe = observation_recipe("Samples", 3, 1).model_dump()
                raw_recipe["sheets"][0]["role"] = role
                if role == "unknown":
                    raw_recipe.update(status="review_required", requires_human_review=True, warnings=["Resolver el rol."])
                recipe = PreparationRecipe.model_validate(raw_recipe)
                policy = ObservationRowPolicy(**RULE, sheet_name="Samples", auxiliary_column_indices=(1,), expected_headers={1: "id"})
                execution = self.execute(content, recipe, PreparationPolicies(observation_rows=(policy,)))
                block = execution.sheets[0]
                self.assertEqual((block.rows_before, block.rows_after, block.auxiliary_rows_removed), (2, 2, 0))
                self.assertEqual(block.formula_count, 2)
                self.assertEqual(block.unresolved_formula_count, 0)
                self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)
                self.assertEqual(len(execution.auxiliary_rows), 2)
                self.assertFalse(any(item.removed_from_observations for item in execution.auxiliary_rows))
                self.assertEqual([item.row_number for item in execution.auxiliary_rows], [2, 3])
                self.assertTrue(all(item.auxiliary_cells[0].formula for item in execution.auxiliary_rows))
                if role == "unknown":
                    self.assertIn("unresolved_block_role", [finding.code for finding in execution.findings])

    def test_confirmed_date_to_decimal_requires_exact_cell_evidence_and_file_digest(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["id", "date", "absolute"])
        sheet.append(["A", "03-12-2004", datetime(2023, 5, 2)])
        sheet.append(["B", "04-12-2004", datetime(2023, 6, 3)])
        sheet["C2"].number_format = r"d\.m"
        sheet["C3"].number_format = r"d\.m"
        content = workbook_bytes(workbook)
        confirmed = ConfirmedCellPolicy(**RULE, sheet_name="Samples", cell="C2", source_sha256=sha256(content).hexdigest(),
                                        expected_source_value="2023-05-02T00:00:00", expected_number_format=r"d\.m",
                                        expected_source_kind="native_datetime", expected_raw_value="45048",
                                        expected_header="absolute", prepared_value=2.5)
        number = NumberColumnPolicy(**RULE, sheet_name="Samples", column_indices=(3,), expected_headers={3: "absolute"})
        execution = self.execute(content, observation_recipe("Samples", 3, 3),
                                 PreparationPolicies(confirmed_cells=(confirmed,), number_columns=(number,)))
        self.assertEqual(execution.sheets[0].sample[0].values[-1], 2.5)
        self.assertEqual(execution.sheets[0].sample[1].values[-1], "2023-06-03T00:00:00")
        trace = execution.normalizations[0]
        self.assertEqual((trace.cell, trace.raw_value, trace.source_kind, trace.number_format), ("C2", "45048", "native_datetime", r"d\.m"))
        self.assertEqual(trace.excel_epoch, "1899-12-30T00:00:00")
        self.assertEqual([finding.cell for finding in execution.findings if finding.code == "abundance_native_date_unresolved"], ["C3"])
        for change in ({"source_sha256": "0" * 64}, {"expected_raw_value": "45049"},
                       {"expected_number_format": "General"}, {"expected_source_value": "2023-02-05T00:00:00"}):
            with self.subTest(change=change):
                mismatch = confirmed.model_copy(update=change)
                execution = self.execute(content, observation_recipe("Samples", 3, 3), PreparationPolicies(confirmed_cells=(mismatch,)))
                self.assertEqual(execution.sheets[0].sample[0].values[-1], "2023-05-02T00:00:00")
                self.assertEqual(execution.normalizations, [])
                self.assertIn("confirmed_cell_evidence_mismatch", [finding.code for finding in execution.findings])

    def test_wrong_headers_do_not_authorize_column_or_auxiliary_row_conversions(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["code", "name", "category"])
        sheet.append(["001", "03-12-2004", "2,5"])
        sheet.append(['=B3&"_"&C3'])
        policies = PreparationPolicies(
            date_columns=(DateColumnPolicy(**RULE, sheet_name="Samples", column_indices=(2,), expected_headers={2: "date"}, text_formats=DATE_FORMATS),),
            number_columns=(NumberColumnPolicy(**RULE, sheet_name="Samples", column_indices=(3,), expected_headers={3: "abundance"}),),
            observation_rows=(ObservationRowPolicy(**RULE, sheet_name="Samples", auxiliary_column_indices=(1,), expected_headers={1: "id"}),),
        )
        execution = self.execute(workbook_bytes(workbook), observation_recipe("Samples", 3, 3), policies)
        self.assertEqual(execution.sheets[0].sample[0].values, ["001", "03-12-2004", "2,5"])
        self.assertEqual(execution.sheets[0].rows_after, 2)
        self.assertEqual(execution.normalizations, [])
        self.assertEqual(execution.auxiliary_rows, [])
        self.assertEqual({finding.code for finding in execution.findings}, {
            "preparation_policy_header_mismatch", "date_consistency_policy_mismatch",
        })
        self.assertTrue(all(finding.requires_review for finding in execution.findings))

    def test_optional_associations_preserve_unmatched_rows_and_localize_duplicates(self):
        workbook = Workbook()
        source = workbook.active
        source.title = "Samples"
        source.append(["id", "link", "abundance"])
        source.append(["S1", "found", 0])
        source.append(["S2", "missing", "nd"])
        source.append(["S3", "duplicate", "+++"])
        target = workbook.create_sheet("Chemistry")
        target.append(["id", "value"])
        target.append(["found", 3])
        target.append(["duplicate", 4])
        target.append(["duplicate", 5])
        policy = AssociationPolicy(**RULE, source_sheet="Samples", source_column=2, expected_source_header="link",
                                   target_sheet="Chemistry", target_column=1, expected_target_header="id")
        execution = self.execute(workbook_bytes(workbook), observation_recipe("Samples", 4, 3), PreparationPolicies(associations=(policy,)))
        self.assertEqual(execution.sheets[0].rows_after, 3)
        self.assertEqual([row.values[0] for row in execution.sheets[0].sample], ["S1", "S2", "S3"])
        self.assertEqual([item.status for item in execution.associations], ["matched", "missing", "ambiguous"])
        self.assertEqual([item.target_rows for item in execution.associations], [(2,), (), (3, 4)])
        self.assertEqual([item.requires_review for item in execution.associations], [False, False, True])
        self.assertEqual(execution.associations[1].source_value, "missing")

    def test_numeric_policy_prevents_datetime_transformation_even_after_rename(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["id", "abundance"])
        sheet.append(["S1", "2004-12-03"])
        recipe = observation_recipe("Samples", 2, 2, [
            {"operation": "rename_columns", "mapping": {"abundance": "quantity"}},
            {"operation": "convert_type", "columns": ["quantity"], "target_type": "datetime"},
        ])
        original_recipe = recipe.model_dump_json()
        number = NumberColumnPolicy(**RULE, sheet_name="Samples", column_indices=(2,), expected_headers={2: "abundance"})
        execution = self.execute(workbook_bytes(workbook), recipe, PreparationPolicies(number_columns=(number,)))
        self.assertEqual(execution.sheets[0].sample[0].values[-1], "2004-12-03")
        self.assertIn("rule_policy_type_conflict", [finding.code for finding in execution.findings])
        self.assertEqual(recipe.model_dump_json(), original_recipe)

    def test_record_identifier_formula_stays_with_observations_but_aggregate_does_not(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["id", "site", "abundance", "calculation"])
        sheet.append(['=CONCATENATE(B2,"_",C2)', "A", 0, "=SUM(C2:C3)"])
        sheet.append(['=B3&"_"&C3', "B", 2, "=SUM(C2:C3)"])
        summary = self.processor.inspect("generic.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.regions if region.start_row == 2], [
            ("observations", 1, 3), ("derived", 4, 4),
        ])

    def test_cross_row_identifier_control_keeps_records_original_formulas_and_caches(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Records"
        sheet.append(["id", "site", "abundance"])
        sheet.append(['=CONCATENATE(B2,"_",C2)', "A", 0])
        sheet.append(['=CONCATENATE(B4,"_",C3)', "B", 2])
        sheet.append(['=B4&"_"&C4', "C", 3])
        original = workbook_bytes(workbook)
        output = io.BytesIO()
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        with ZipFile(io.BytesIO(original)) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
            for entry in source.infolist():
                body = source.read(entry.filename)
                if entry.filename == "xl/worksheets/sheet1.xml":
                    xml = ElementTree.fromstring(body)
                    for cell in xml.iter(namespace + "c"):
                        if cell.attrib.get("r") in {"A2", "A3", "A4"}:
                            cell.set("t", "str")
                            cell.find(namespace + "v").text = "original_" + cell.attrib["r"]
                    body = ElementTree.tostring(xml, encoding="utf-8")
                target.writestr(entry, body)
        content = output.getvalue()
        original_hash = sha256(content).hexdigest()
        recipe = observation_recipe("Records", 4, 3)
        recipe_before = recipe.model_dump_json()
        summary = self.processor.inspect("generic.xlsx", content)
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.sheets[0].regions if region.start_row == 2],
                         [("observations", 1, 3)])
        without_policy = self.processor.execute("generic.xlsx", content, recipe, summary)
        self.assertNotIn("record_identifier_cross_row_reference", [finding.code for finding in without_policy.findings])
        policy = ObservationRowPolicy(**RULE, sheet_name="Records", auxiliary_column_indices=(1,), expected_headers={1: "id"})
        execution = self.processor.execute("generic.xlsx", content, recipe, summary,
                                           policies=PreparationPolicies(observation_rows=(policy,)))
        controls = [finding for finding in execution.findings if finding.code == "record_identifier_cross_row_reference"]
        self.assertEqual([finding.cell for finding in controls], ["A3"])
        self.assertTrue(controls[0].requires_review)
        self.assertIn('=CONCATENATE(B4,"_",C3)', controls[0].message)
        self.assertIn("original_A3", controls[0].message)
        self.assertEqual([row.values for row in execution.sheets[0].sample],
                         [["original_A2", "A", 0], ["original_A3", "B", 2], ["original_A4", "C", 3]])
        proof = next(item for item in execution.sheets[0].formula_sample if item.cell == "A3")
        self.assertEqual((proof.formula, proof.cached_value, proof.cache_available),
                         ('=CONCATENATE(B4,"_",C3)', "original_A3", True))
        self.assertEqual(execution.auxiliary_rows, [])
        self.assertEqual(recipe.model_dump_json(), recipe_before)
        self.assertEqual(sha256(content).hexdigest(), original_hash)

    def test_aggregate_in_identifier_column_is_not_exempted_by_text_constructions(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["id", "measurement"])
        sheet.append(['=CONCATENATE(B2)', 0])
        sheet.append(['=CONCATENATE(B3)', 2])
        sheet.append(['=SUM(B2:B4)', 3])
        summary = self.processor.inspect("generic.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.regions if region.start_row == 2],
                         [("derived", 1, 1), ("observations", 2, 2)])

    def test_equivalent_cross_row_identifier_operands_are_diagnostic_without_review(self):
        for other_operand, requires_review in (("B", False), ("=B3", True)):
            with self.subTest(other_operand=other_operand):
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Records"
                sheet.append(["id", "site", "abundance"])
                sheet.append(['=CONCATENATE(B2,"_",C2)', "A", 0])
                sheet.append(['=CONCATENATE(B4,"_",C3)', "B", 2])
                sheet.append(['=B4&"_"&C4', other_operand, 3])
                policy = ObservationRowPolicy(**RULE, sheet_name="Records", auxiliary_column_indices=(1,), expected_headers={1: "id"})
                execution = self.execute(workbook_bytes(workbook), observation_recipe("Records", 4, 3),
                                         PreparationPolicies(observation_rows=(policy,)))
                controls = [finding for finding in execution.findings if finding.code == "record_identifier_cross_row_reference"]
                self.assertEqual([finding.cell for finding in controls], ["A3"])
                self.assertEqual(controls[0].requires_review, requires_review)
                self.assertIn("B4=", controls[0].message)
                self.assertIn("B3='B'", controls[0].message)
                self.assertIn("evidencia o caché insuficiente" if requires_review else "igualdad comprobada", controls[0].message)
                self.assertEqual(execution.sheets[0].rows_after, 3)
                self.assertEqual(execution.sheets[0].formula_sample[1].formula, '=CONCATENATE(B4,"_",C3)')

    def test_identifier_tokenization_does_not_misread_function_digits_as_cell_references(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["id", "site", "abundance"])
        sheet.append(['=CONCATENATE(B2,LOG10(C2))', "A", 10])
        sheet.append(['=CONCATENATE(B3,LOG10(C3))', "B", 100])
        summary = self.processor.inspect("generic.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.regions if region.start_row == 2],
                         [("observations", 1, 3)])

    def test_policy_header_evidence_covers_original_wide_positions_and_full_values(self):
        workbook = Workbook()
        source = workbook.active
        source.title = "Records"
        source.cell(1, 1, "id")
        source.cell(1, 2, "date")
        source.cell(1, 3, "abundance")
        source.cell(1, 5, "=1+1")
        source.cell(1, 28, "link")
        long_header = "  " + "documented description " * 20 + "  "
        source.cell(1, 30, long_header)
        source.cell(2, 1, "r1")
        source.cell(2, 2, "03-12-2004")
        source.cell(2, 3, "2,5")
        source.cell(2, 28, "chem1")
        target = workbook.create_sheet("Chemistry")
        target.cell(2, 45, "record key")
        target.cell(3, 45, "chem1")
        content = workbook_bytes(workbook)
        policies = PreparationPolicies(
            date_columns=(DateColumnPolicy(**RULE, sheet_name="Records", column_indices=(2,),
                                           expected_headers={2: "date"}, text_formats=DATE_FORMATS),),
            number_columns=(NumberColumnPolicy(**RULE, sheet_name="Records", column_indices=(3,),
                                               expected_headers={3: "abundance"}),),
            confirmed_cells=(ConfirmedCellPolicy(**RULE, sheet_name="Records", cell="C2",
                source_sha256=sha256(content).hexdigest(), expected_source_value="2,5",
                expected_number_format="General", expected_source_kind="text", expected_header="abundance", prepared_value=2.5),),
            observation_rows=(ObservationRowPolicy(**RULE, sheet_name="Records", auxiliary_column_indices=(1,), expected_headers={1: "id"}),),
            associations=(AssociationPolicy(**RULE, source_sheet="Records", source_column=28,
                expected_source_header="link", target_sheet="Chemistry", target_column=45,
                expected_target_header="record key", header_row=2, source_start_row=3, target_start_row=3),),
        )
        execution = self.execute(content, observation_recipe("Records", 2, 30), policies)
        headers = {(evidence.sheet_name, evidence.row_number): evidence for evidence in execution.source_headers}
        self.assertEqual(set(headers), {("Records", 1), ("Records", 2), ("Chemistry", 2)})
        evidence = headers["Records", 1]
        self.assertEqual(evidence.inspected_columns, 30)
        self.assertEqual(set(evidence.values), set(range(1, 31)))
        self.assertEqual(evidence.values[28], "link")
        self.assertEqual(evidence.values[30], long_header)
        self.assertIsNone(evidence.values[4])
        self.assertIsNone(evidence.values[5])
        self.assertEqual(evidence.formula_columns, (5,))
        self.assertEqual(headers["Chemistry", 2].inspected_columns, 45)
        self.assertEqual(headers["Chemistry", 2].values[45], "record key")
        empty = self.execute(content, observation_recipe("Records", 2, 30), PreparationPolicies())
        self.assertEqual(empty.source_headers, [])
        self.assertNotIn("source_headers", empty.model_dump())

    def test_documentary_blocks_keep_original_values_and_reject_overlap(self):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Dictionary"
        sheet.append(["code", "meaning"])
        sheet.append(["001", "relative abundance"])
        sheet.append(["nd", "not detected"])
        content = workbook_bytes(workbook)
        dictionary = observation_recipe("Dictionary", 3, 2).model_dump()
        dictionary["sheets"][0]["role"] = "legend"
        recipe = PreparationRecipe.model_validate(dictionary)
        execution = self.execute(content, recipe, PreparationPolicies())
        self.assertEqual(execution.sheets[0].role, "legend")
        self.assertEqual([row.values for row in execution.sheets[0].sample], [["001", "relative abundance"], ["nd", "not detected"]])
        self.assertEqual(execution.coverage[0].uncovered_nonempty_cells, 0)
        overlap = {**dictionary["sheets"][0], "block_id": "duplicate", "data_start_row": 3}
        dictionary["sheets"].append(overlap)
        with self.assertRaises(ValueError):
            PreparationRecipe.model_validate(dictionary)


if __name__ == "__main__":
    unittest.main()
