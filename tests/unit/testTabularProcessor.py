"""Pruebas del lector y de las operaciones declarativas sin servicios externos."""

import io
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook

from aquai.application.dtos.tabular import PreparationRecipe, SheetRecipe
from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.application.services.tabularValidation import TabularValidationService
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.processing.tabular import PandasTabularProcessor


def workbook_bytes(workbook: Workbook) -> bytes:
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def recipe(sheet_name: str = "data", **overrides: object) -> PreparationRecipe:
    sheet = {
        "sheet_name": sheet_name, "block_id": "main", "role": "observations",
        "header_rows": [1], "data_start_row": 2, "data_end_row": 2,
        "data_start_column": 1, "data_end_column": 1,
        "transformations": [],
    }
    sheet.update(overrides)
    return PreparationRecipe.model_validate({
        "status": "ready", "sheets": [sheet], "warnings": [], "requires_human_review": False,
    })


class TabularProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = PandasTabularProcessor()

    def execute(self, content: bytes, preparation: PreparationRecipe, file_name: str = "test.csv"):
        summary = self.processor.inspect(file_name, content)
        return self.processor.execute(file_name, content, preparation, summary)

    def test_csv_preserves_physical_rows_nulls_and_literal_na(self) -> None:
        content = b"name,score,code\nAna,10,NA\n,,\nLuis,,001\n"
        summary = self.processor.inspect("test.csv", content)
        sheet = summary.sheets[0]
        self.assertEqual((sheet.rows, sheet.columns), (4, 3))
        self.assertEqual(sheet.empty_rows, [3])
        self.assertEqual(sheet.null_counts, [1, 2, 1])
        self.assertEqual(sheet.sample[1].values, ["Ana", "10", "NA"])
        self.assertEqual(sheet.sample[-1].values[-1], "001")

    def test_csv_supports_utf8_bom_semicolon_and_multiline_values(self) -> None:
        content = '\ufeffnombre;detalle\nJosé;"dos\nlíneas"\n'.encode()
        summary = self.processor.inspect("datos.CSV", content)
        self.assertEqual(summary.csv_delimiter, ";")
        self.assertEqual(summary.sheets[0].rows, 2)
        self.assertEqual(summary.sheets[0].sample[1].values, ["José", "dos\nlíneas"])

    def test_csv_cp1252_fallback_is_reported(self) -> None:
        summary = self.processor.inspect("test.csv", "nombre\nJosé\n".encode("cp1252"))
        self.assertEqual(summary.csv_encoding, "cp1252")
        self.assertTrue(summary.warnings)

    def test_csv_uneven_rows_are_reported_without_losing_values(self) -> None:
        summary = self.processor.inspect("test.csv", b"a,b\n1\n2,3,4\n")
        self.assertEqual(summary.sheets[0].columns, 3)
        self.assertEqual(summary.sheets[0].sample[-1].values, ["2", "3", "4"])
        self.assertTrue(summary.warnings)

    def test_samples_cover_beginning_and_end_and_remain_bounded(self) -> None:
        content = ("value\n" + "\n".join(str(value) for value in range(100))).encode()
        sheet = self.processor.inspect("test.csv", content).sheets[0]
        self.assertEqual(len(sheet.sample), 12)
        self.assertEqual(sheet.sample[0].row_number, 1)
        self.assertEqual(sheet.sample[-1].row_number, 101)

    def test_wide_summary_maps_columns_and_preserves_complete_dimensions(self) -> None:
        headers = [f"field_{index}" for index in range(1, 81)]
        values = [str(index) for index in range(1, 81)]
        content = (",".join(headers) + "\n" + ",".join(values) + "\n").encode()
        sheet = self.processor.inspect("wide.csv", content).sheets[0]
        indices = list(range(1, 13)) + list(range(77, 81))
        self.assertEqual((sheet.rows, sheet.columns), (2, 80))
        self.assertEqual(sheet.column_indices, indices)
        self.assertEqual(sheet.initial_columns, [headers[index - 1] for index in indices])
        self.assertEqual(sheet.sample[1].values, [values[index - 1] for index in indices])
        self.assertEqual(len(sheet.inferred_types), 16)
        self.assertEqual(sheet.null_counts, [0] * 16)
        self.assertTrue(sheet.warnings)

    def test_wide_summary_ignores_styled_tail_for_representative_columns(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append([f"field_{index}" for index in range(1, 21)])
        sheet.append(list(range(20)))
        sheet.cell(4, 100).number_format = "0.00"
        summary = self.processor.inspect("wide.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual((summary.rows, summary.columns), (4, 100))
        self.assertEqual(summary.column_indices, list(range(1, 13)) + list(range(17, 21)))
        self.assertEqual(summary.empty_column_count, 80)
        self.assertEqual(summary.empty_columns, list(range(21, 101)))

    def test_formula_sample_covers_end_and_keeps_uncached_columns_visible(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["value"])
        for row in range(2, 32):
            sheet.cell(row, 40, f"=A{row}+1")
        summary = self.processor.inspect("wide.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual(summary.formula_count, 30)
        self.assertEqual(len(summary.formula_sample), 12)
        self.assertEqual(summary.formula_sample[0].cell, "AN2")
        self.assertEqual(summary.formula_sample[-1].cell, "AN31")
        self.assertIn(40, summary.column_indices)
        self.assertNotIn(31, summary.empty_rows)

    def test_verified_empty_sheet_is_recorded_without_generating_data_tables(self) -> None:
        for styled in (False, True):
            with self.subTest(styled=styled):
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "Auxiliary"
                if styled:
                    sheet.cell(176, 26).number_format = "0.00"
                content = workbook_bytes(workbook)
                preparation = PreparationRecipe.model_validate({
                    "status": "review_required", "requires_human_review": True,
                    "warnings": ["Resolver el rol de la hoja auxiliar."],
                    "sheets": [
                        {
                            "sheet_name": "Auxiliary", "block_id": identifier,
                            "role": role, "header_rows": [],
                            "data_start_row": 1, "data_end_row": 176,
                            "data_start_column": start_column, "data_end_column": end_column,
                            "transformations": [],
                        }
                        for identifier, role, start_column, end_column in (
                            ("unresolved", "unknown", 1, 13), ("planned", "observations", 14, 26),
                        )
                    ],
                })
                original_recipe = preparation.model_dump_json()
                summary = self.processor.inspect("auxiliary.xlsx", content)
                evidence = summary.sheets[0].content_evidence
                self.assertTrue(evidence.is_empty)
                self.assertEqual((evidence.value_cells, evidence.formula_cells), (0, 0))
                self.assertEqual(evidence.inspected_cells, 176 * 26 if styled else 0)
                self.assertEqual(evidence.verification, "full_sheet_values_and_formulas")
                execution = self.processor.execute("auxiliary.xlsx", content, preparation, summary)
                self.assertEqual(execution.sheets, [])
                self.assertEqual(execution.findings, [])
                self.assertEqual(execution.sheet_content_evidence, [evidence])
                treatment = execution.empty_sheets[0]
                self.assertEqual(treatment.sheet_name, "Auxiliary")
                self.assertEqual(treatment.action, "verified_empty_no_table")
                self.assertEqual(treatment.evidence, evidence)
                self.assertEqual(treatment.skipped_block_ids, ["unresolved", "planned"])
                self.assertEqual(execution.coverage[0].nonempty_cells, 0)
                self.assertEqual(preparation.model_dump_json(), original_recipe)

    def test_unselected_empty_sheet_is_still_registered_separately(self) -> None:
        workbook = Workbook()
        workbook.active.title = "Measurements"
        workbook.active.append(["value"])
        workbook.active.append([5])
        workbook.create_sheet("Unused").cell(20, 15).number_format = "0.00"
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("measurements.xlsx", content)
        execution = self.processor.execute("measurements.xlsx", content, recipe("Measurements"), summary)
        self.assertEqual([block.sheet_name for block in execution.sheets], ["Measurements"])
        self.assertEqual(execution.sheets[0].sample[0].values, [5])
        self.assertEqual(execution.skipped_sheets, ["Unused"])
        self.assertEqual([item.sheet_name for item in execution.empty_sheets], ["Unused"])
        self.assertEqual(execution.empty_sheets[0].skipped_block_ids, [])
        self.assertEqual(len(execution.sheet_content_evidence), 2)
        self.assertEqual(len(execution.coverage), 2)

    def test_sheet_emptiness_scans_values_outside_all_sampled_rows(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.cell(300, 50).number_format = "0.00"
        sheet.cell(157, 30, "exclusive observation")
        summary = self.processor.inspect("sparse.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertNotIn(157, [sample.row_number for sample in summary.sample])
        self.assertTrue(all(value is None for sample in summary.sample for value in sample.values))
        self.assertEqual(summary.content_evidence.inspected_cells, 300 * 50)
        self.assertEqual(summary.content_evidence.value_cells, 1)
        self.assertFalse(summary.content_evidence.is_empty)

    def test_formula_only_sheet_without_any_cached_values_is_not_empty(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.cell(200, 30).number_format = "0.00"
        sheet.cell(157, 17, "=1+1")
        content = workbook_bytes(workbook)
        preparation = PreparationRecipe.model_validate({
            "status": "review_required", "requires_human_review": True,
            "warnings": ["Resolver el rol."], "sheets": [{
                "sheet_name": "Sheet", "block_id": "unresolved", "role": "unknown",
                "header_rows": [], "data_start_row": 1, "data_end_row": 200,
                "data_start_column": 1, "data_end_column": 30, "transformations": [],
            }],
        })
        summary = self.processor.inspect("formula-only.xlsx", content)
        evidence = summary.sheets[0].content_evidence
        self.assertEqual((evidence.value_cells, evidence.formula_cells), (0, 1))
        self.assertFalse(evidence.is_empty)
        self.assertFalse(any("está vacía" in warning for warning in summary.sheets[0].warnings))
        execution = self.processor.execute("formula-only.xlsx", content, preparation, summary)
        self.assertEqual(execution.empty_sheets, [])
        self.assertEqual(len(execution.sheets), 1)
        self.assertEqual(execution.sheets[0].unresolved_formula_cells, ["Q157"])
        self.assertEqual(execution.sheets[0].formula_sample[0].formula, "=1+1")
        self.assertEqual(execution.sheet_content_evidence, [evidence])
        self.assertIn("unresolved_block_role", [finding.code for finding in execution.findings])

    def test_zero_false_and_whitespace_are_values_when_verifying_sheet_emptiness(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append([0])
        sheet.append([False])
        sheet.append(["   "])
        summary = self.processor.inspect("literal-values.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual(summary.content_evidence.value_cells, 3)
        self.assertFalse(summary.content_evidence.is_empty)
        self.assertFalse(any("está vacía" in warning for warning in summary.warnings))

    def test_execution_rechecks_content_instead_of_trusting_empty_summary_evidence(self) -> None:
        empty_workbook = Workbook()
        empty_content = workbook_bytes(empty_workbook)
        empty_summary = self.processor.inspect("same.xlsx", empty_content)
        workbook = Workbook()
        workbook.active.append(["value"])
        workbook.active.append([9])
        content = workbook_bytes(workbook)
        execution = self.processor.execute("same.xlsx", content, recipe("Sheet"), empty_summary)
        self.assertEqual(execution.empty_sheets, [])
        self.assertFalse(execution.sheet_content_evidence[0].is_empty)
        self.assertEqual(execution.sheet_content_evidence[0].value_cells, 2)
        self.assertEqual(execution.sheets[0].sample[0].values, [9])

        populated_summary = self.processor.inspect("same.xlsx", content)
        execution = self.processor.execute("same.xlsx", empty_content, recipe("Sheet"), populated_summary)
        self.assertEqual(execution.sheets, [])
        self.assertTrue(execution.sheet_content_evidence[0].is_empty)
        self.assertEqual(execution.empty_sheets[0].skipped_block_ids, ["main"])

    def test_zero_values_survive_empty_row_and_column_operations(self) -> None:
        result = self.execute(b"a,b\n0,0\n,,\n", recipe(data_end_row=3, data_end_column=3, transformations=[
            {"operation": "drop_empty_rows"},
            {"operation": "drop_empty_columns"},
        ])).sheets[0]
        self.assertEqual((result.rows_before, result.rows_after), (2, 1))
        self.assertEqual(result.dropped_empty_columns, ["column_3"])
        self.assertEqual(result.sample[0].values, ["0", "0"])

    def test_invalid_and_empty_files_are_rejected(self) -> None:
        for filename, content in (("test.pdf", b"pdf"), ("test.csv", b""), ("test.csv", b" \n"), ("test.csv", b"a\x00b"), ("test.xlsx", b"broken")):
            with self.subTest(filename=filename, content=content):
                with self.assertRaises(TabularInputError):
                    self.processor.inspect(filename, content)

    def test_cell_and_archive_limits_are_enforced(self) -> None:
        with self.assertRaises(FileTooLargeError):
            PandasTabularProcessor(max_cells=3).inspect("test.csv", b"a,b\n1,2\n")
        workbook = Workbook()
        workbook.active.append(["value"])
        content = workbook_bytes(workbook)
        with self.assertRaises(FileTooLargeError):
            PandasTabularProcessor(max_expanded_bytes=10).inspect("test.xlsx", content)
        workbook = Workbook()
        workbook.create_sheet("Second")
        with self.assertRaises(FileTooLargeError):
            PandasTabularProcessor(max_sheets=1).inspect("test.xlsx", workbook_bytes(workbook))

    def test_merged_header_expansion_is_bounded_before_openpyxl(self) -> None:
        workbook = Workbook()
        workbook.active.append(["value"])
        original = workbook_bytes(workbook)
        output = io.BytesIO()
        with ZipFile(io.BytesIO(original)) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
            for entry in source.infolist():
                body = source.read(entry.filename)
                if entry.filename == "xl/worksheets/sheet1.xml":
                    body = body.replace(b"</worksheet>", b'<mergeCells count="1"><mergeCell ref="A1:A1048576"/></mergeCells></worksheet>')
                target.writestr(entry, body)
        with self.assertRaises(FileTooLargeError):
            PandasTabularProcessor(max_cells=100).inspect("test.xlsx", output.getvalue())

    def test_operations_trim_rename_convert_and_drop_empty(self) -> None:
        content = b"name,score,unused\n Ana ,10,\n,,\nLuis,bad,\n"
        result = self.execute(content, recipe(data_end_row=4, data_end_column=3, transformations=[
            {"operation": "strip_strings"},
            {"operation": "drop_empty_rows"},
            {"operation": "drop_empty_columns"},
            {"operation": "rename_columns", "mapping": {"score": "points"}},
            {"operation": "convert_type", "columns": ["points"], "target_type": "integer"},
        ])).sheets[0]
        self.assertEqual((result.rows_before, result.rows_after), (3, 2))
        self.assertEqual(result.dropped_empty_rows, 1)
        self.assertEqual(result.dropped_empty_columns, ["unused"])
        self.assertEqual(result.columns_after, ["name", "points"])
        self.assertEqual(result.conversion_failures, {"points": 1})
        self.assertEqual(result.sample[0].values, ["Ana", 10])
        self.assertEqual(result.sample[-1].values, ["Luis", "bad"])
        self.assertEqual(result.sample[-1].row_number, 4)

    def test_range_exclusion_and_numeric_headers_are_reported(self) -> None:
        content = b"title\n10\n20\n30\n40\n"
        result = self.execute(content, recipe(header_rows=[2], data_start_row=3, data_end_row=4)).sheets[0]
        self.assertEqual(result.excluded_nonempty_rows, 2)
        self.assertEqual(result.suspicious_header_rows, [2])
        self.assertEqual((result.rows_before, result.rows_after), (2, 2))

    def test_excel_multirow_merged_headers_form_unique_names(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Counts"
        sheet.append(["Group", None, None, "duplicate", "duplicate"])
        sheet.merge_cells("A1:B1")
        sheet.append(["x", "y", None, None, None])
        sheet.append([1, 2, 3, 4, 5])
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("test.xlsx", content)
        self.assertEqual(summary.sheets[0].merged_cells, ["A1:B1"])
        result = self.processor.execute("test.xlsx", content, recipe(
            "Counts", header_rows=[1, 2], data_start_row=3, data_end_row=3, data_end_column=5,
        ), summary).sheets[0]
        self.assertEqual(result.columns_after, ["Group x", "Group y", "column_3", "duplicate", "duplicate__2"])
        self.assertEqual(result.sample[0].values, [1, 2, 3, 4, 5])
        self.assertEqual(len(result.warnings), 2)

    def test_formula_without_cache_is_preserved_by_empty_row_cleanup(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["result"])
        sheet.append(["=1+1"])
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("test.xlsx", content)
        self.assertEqual(summary.sheets[0].formula_count, 1)
        self.assertNotIn(2, summary.sheets[0].empty_rows)
        result = self.processor.execute("test.xlsx", content, recipe("Sheet", transformations=[{"operation": "drop_empty_rows"}]), summary).sheets[0]
        self.assertEqual(result.unresolved_formula_cells, ["A2"])
        self.assertEqual((result.rows_before, result.rows_after), (1, 1))
        self.assertEqual(result.dropped_empty_rows, 0)
        self.assertEqual(result.sample[0].values, [None])
        self.assertTrue(result.warnings)

    def test_formula_protection_survives_rename_and_bounded_diagnostics(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["unused", "result"])
        for row in range(2, 123):
            sheet.append([None, f"={row}+1"])
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("formulas.xlsx", content)
        self.assertEqual(summary.sheets[0].empty_columns, [])
        self.assertEqual(summary.sheets[0].missing_formula_cache_count, 121)
        result = self.processor.execute(
            "formulas.xlsx", content,
            recipe("Sheet", data_end_row=122, data_end_column=2, transformations=[
                {"operation": "rename_columns", "mapping": {"result": "computed"}},
                {"operation": "drop_empty_columns"},
                {"operation": "drop_empty_rows"},
            ]), summary,
        ).sheets[0]
        self.assertEqual(len(result.unresolved_formula_cells), 100)
        self.assertEqual((result.rows_before, result.rows_after), (121, 121))
        self.assertEqual(result.dropped_empty_rows, 0)
        self.assertEqual(result.dropped_empty_columns, ["unused"])
        self.assertEqual(result.columns_after, ["computed"])
        self.assertEqual(result.empty_columns, [])
        self.assertEqual(result.sample[-1].row_number, 122)

    def test_invalid_sheet_range_and_columns_become_sheet_errors(self) -> None:
        content = b"a,b\n1,2\n"
        for preparation in (
            recipe("missing"),
            recipe(data_start_row=10, data_end_row=10),
            recipe(data_end_row=10),
            recipe(data_end_column=2, transformations=[{"operation": "rename_columns", "mapping": {"a": "b"}}]),
            recipe(transformations=[{"operation": "convert_type", "columns": ["missing"], "target_type": "number"}]),
        ):
            with self.subTest(preparation=preparation):
                self.assertTrue(self.execute(content, preparation).sheets[0].errors)

    def test_numeric_excel_values_are_not_guessed_as_timestamp_units(self) -> None:
        workbook = Workbook()
        workbook.active.append(["date"])
        workbook.active.append([45200])
        result = self.execute(workbook_bytes(workbook), recipe("Sheet", transformations=[
            {"operation": "convert_type", "columns": ["date"], "target_type": "datetime"},
        ]), "test.xlsx").sheets[0]
        self.assertEqual(result.sample[0].values, [45200])
        self.assertEqual(result.conversion_failures, {"date": 1})

    def test_uneven_semicolon_rows_keep_the_separator(self) -> None:
        summary = self.processor.inspect("test.csv", b"a;b\n1\n2;3;4\n")
        self.assertEqual(summary.csv_delimiter, ";")
        self.assertEqual(summary.sheets[0].sample[-1].values, ["2", "3", "4"])

    def test_uneven_semicolon_csv_cannot_silently_become_one_column(self) -> None:
        content = b"name;age\nAna;10\nBob;20;extra\n"
        summary = self.processor.inspect("test.csv", content)
        self.assertEqual(summary.csv_delimiter, ";")
        self.assertEqual(summary.sheets[0].columns, 3)
        self.assertEqual(summary.sheets[0].sample[-1].values, ["Bob", "20", "extra"])
        self.assertTrue(summary.warnings)

    def test_numeric_precision_loss_is_a_conversion_failure(self) -> None:
        result = self.execute(b"value\n9007199254740993\n1.5\n", recipe(data_end_row=3, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])).sheets[0]
        self.assertEqual(result.conversion_failures, {"value": 1})
        self.assertEqual([row.values[0] for row in result.sample], ["9007199254740993", 1.5])
        exact = self.execute(b"value\n9007199254740993\n\n", recipe(data_end_row=3, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])).sheets[0]
        self.assertEqual(exact.sample[0].values, [9007199254740993])
        self.assertEqual(exact.conversion_failures, {})

    def test_cached_excel_errors_are_distinguished_from_literal_strings(self) -> None:
        workbook = Workbook()
        workbook.active.append(["literal", "formula"])
        workbook.active.append(["text", "=1/0"])
        original = workbook_bytes(workbook)
        output = io.BytesIO()
        with ZipFile(io.BytesIO(original)) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
            for entry in source.infolist():
                body = source.read(entry.filename)
                if entry.filename == "xl/worksheets/sheet1.xml":
                    body = body.replace(b'<c r="B2"><f>1/0</f><v /></c>', b'<c r="B2" t="e"><f>1/0</f><v>#DIV/0!</v></c>')
                    body = body.replace(b">text<", b">#DIV/0!<")
                target.writestr(entry, body)
        content = output.getvalue()
        summary = self.processor.inspect("test.xlsx", content)
        self.assertEqual(summary.sheets[0].error_cell_count, 1)
        self.assertEqual(summary.sheets[0].error_cells, ["B2"])
        result = self.processor.execute("test.xlsx", content, recipe("Sheet", data_end_column=2), summary).sheets[0]
        self.assertEqual(result.error_cells, ["B2"])
        self.assertEqual(result.unresolved_formula_cells, [])
        self.assertEqual(result.sample[0].values, ["#DIV/0!", "#DIV/0!"])

    def test_custom_worksheet_paths_are_checked_before_openpyxl(self) -> None:
        workbook = Workbook()
        workbook.active.append(["value"])
        original = workbook_bytes(workbook)
        output = io.BytesIO()
        with ZipFile(io.BytesIO(original)) as source, ZipFile(output, "w", ZIP_DEFLATED) as target:
            for entry in source.infolist():
                body = source.read(entry.filename)
                name = entry.filename
                if name == "xl/worksheets/sheet1.xml":
                    name = "xl/custom/data.xml"
                    body = body.replace(b"</worksheet>", b'<mergeCells count="1"><mergeCell ref="A1:A300"/></mergeCells></worksheet>')
                elif name in {"xl/_rels/workbook.xml.rels", "[Content_Types].xml"}:
                    body = body.replace(b"/xl/worksheets/sheet1.xml", b"/xl/custom/data.xml")
                target.writestr(name, body)
        with patch("aquai.infrastructure.processing.tabular.reader.load_workbook") as load:
            with self.assertRaises(FileTooLargeError):
                PandasTabularProcessor(max_cells=100).inspect("test.xlsx", output.getvalue())
            load.assert_not_called()

    def test_each_conversion_reports_invalid_values_without_code_execution(self) -> None:
        cases = (
            ("integer", "2\n2.5\n9223372036854775808\n", [2, "2.5", "9223372036854775808"], 2),
            ("number", "2.5\ninf\nbad\n", [2.5, "inf", "bad"], 2),
            ("boolean", "true\nsí\nwrong\n", [True, "sí", "wrong"], 2),
            ("datetime", "2026-10-10\nbad\n", ["2026-10-10T00:00:00+00:00", "bad"], 1),
        )
        for target_type, rows, expected, failures in cases:
            with self.subTest(target_type=target_type):
                result = self.execute(("value\n" + rows).encode(), recipe(
                    data_end_row=1 + len(expected), transformations=[
                        {"operation": "convert_type", "columns": ["value"], "target_type": target_type},
                    ],
                )).sheets[0]
                self.assertEqual([row.values[0] for row in result.sample], expected)
                self.assertEqual(result.conversion_failures, {"value": failures})

    def test_block_coverage_reports_gap_and_overlap_by_original_coordinates(self) -> None:
        preparation = PreparationRecipe.model_construct(sheets=[
            SheetRecipe.model_validate({
                "sheet_name": "data", "block_id": identifier, "role": "observations", "header_rows": [1],
                "data_start_row": 2, "data_end_row": 2,
                "data_start_column": 2, "data_end_column": 2, "transformations": []})
            for identifier in ("first", "second")
        ], status="ready", warnings=[], requires_human_review=False)
        result = self.execute(b"taxon,sample\nA,0\nB,3\n", preparation)
        self.assertEqual(result.sheets[0].columns_after, ["sample"])
        self.assertEqual(result.coverage[0].overlapping_nonempty_cells, 1)
        self.assertEqual(result.coverage[0].uncovered_nonempty_cells, 4)
        self.assertEqual(result.coverage[0].uncovered_sample, ["A1", "A2", "A3", "B3"])

    def test_blocks_share_headers_and_separate_total_without_data_loss(self) -> None:
        preparation = PreparationRecipe.model_validate({
            "status": "ready", "warnings": [], "requires_human_review": False,
            "sheets": [
                {"sheet_name": "data", "block_id": "observations", "role": "observations",
                 "header_rows": [1], "data_start_row": 2, "data_end_row": 3,
                 "data_start_column": 1, "data_end_column": 2, "transformations": []},
                {"sheet_name": "data", "block_id": "total", "role": "total",
                 "header_rows": [1], "data_start_row": 4, "data_end_row": 4,
                 "data_start_column": 1, "data_end_column": 2, "transformations": []},
            ],
        })
        content = b"taxon,sample\nA,0\nB,3\nTotal,3\n"
        result = self.execute(content, preparation)
        self.assertEqual([block.rows_after for block in result.sheets], [2, 1])
        self.assertEqual([block.role for block in result.sheets], ["observations", "total"])
        self.assertEqual(result.sheets[1].sample[0].row_number, 4)
        self.assertEqual(result.coverage[0].nonempty_cells, 8)
        self.assertEqual(result.coverage[0].uncovered_nonempty_cells, 0)
        self.assertEqual(result.coverage[0].overlapping_nonempty_cells, 0)
        self.assertEqual(result.sheets[0].source_nonempty_cells, result.sheets[0].preserved_nonempty_cells)

    def test_region_inspection_exposes_totals_derived_summary_and_legend(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        header = ["taxon"] + [f"sample_{index}" for index in range(1, 9)] + ["computed", "presencias", "%"]
        sheet.append(header)
        sheet.append(["A"] + [0] * 8 + ["=B2+1", "A", 0.5])
        sheet.append(["B"] + [2] * 8 + ["=B3+1", "B", 1])
        sheet.append(["Total"] + [2] * 8)
        sheet.append([])
        sheet.append(header)
        sheet.append([])
        sheet.append(["C1", "mar-abr 2006"])
        sheet.append(["C2", "nov 2007"])
        summary = self.processor.inspect("test.xlsx", workbook_bytes(workbook)).sheets[0]
        regions = {(region.start_row, region.end_row, region.start_column, region.end_column): region.kind
                   for region in summary.regions}
        self.assertEqual(regions[(2, 3, 1, 9)], "observations")
        self.assertEqual(regions[(2, 3, 10, 10)], "derived")
        self.assertEqual(regions[(2, 3, 11, 12)], "summary")
        self.assertEqual(regions[(4, 4, 1, 9)], "total")
        self.assertEqual(regions[(6, 6, 1, 12)], "repeated_header")
        self.assertEqual(regions[(8, 9, 1, 2)], "legend")

    def test_formula_protection_and_errors_are_scoped_to_selected_rectangle(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["value", "calculation"])
        sheet.append([0, "=1+1"])
        sheet.append([None, "=" + "+".join(["1"] * 150)])
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("test.xlsx", content)
        self.assertGreater(len(summary.sheets[0].formula_sample[-1].formula), 200)
        block = recipe("Sheet", data_end_row=3, data_start_column=1, data_end_column=1,
                       transformations=[{"operation": "drop_empty_rows"}])
        result = self.processor.execute("test.xlsx", content, block, summary).sheets[0]
        self.assertEqual(result.formula_count, 0)
        self.assertEqual(result.unresolved_formula_cells, [])
        self.assertEqual(result.dropped_empty_rows, 1)
        self.assertEqual(result.sample[0].values, [0])

    def test_codes_not_detected_values_and_ambiguous_periods_survive_conversion(self) -> None:
        content = b"value\n00123\n0\nnd\nn.d\n-\n<0.001\n"
        result = self.execute(content, recipe(data_end_row=7, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])).sheets[0]
        self.assertEqual([row.values[0] for row in result.sample], ["00123", 0, "nd", "n.d", "-", "<0.001"])
        self.assertEqual(result.conversion_failures, {"value": 5})
        self.assertEqual(result.findings[0].cell, "A2")
        self.assertEqual(result.source_nonempty_cells, result.preserved_nonempty_cells)
        period = self.execute(b"period\n03/04/2024\nmar-abr 2006\n2024-03\n", recipe(data_end_row=4, transformations=[
            {"operation": "convert_type", "columns": ["period"], "target_type": "datetime"},
        ])).sheets[0]
        self.assertEqual([row.values[0] for row in period.sample], ["03/04/2024", "mar-abr 2006", "2024-03"])
        self.assertEqual(period.conversion_failures, {"period": 3})

    def test_headerless_numeric_repetitions_are_observations(self) -> None:
        content = b"0,1\n0,1\n0,1\n"
        summary = self.processor.inspect("counts.csv", content)
        self.assertEqual(len(summary.sheets[0].regions), 1)
        self.assertEqual(summary.sheets[0].regions[0].kind, "observations")
        result = self.processor.execute("counts.csv", content, recipe(
            header_rows=[], data_start_row=1, data_end_row=3, data_end_column=2,
        ), summary)
        self.assertEqual(result.sheets[0].rows_after, 3)
        self.assertEqual(result.coverage[0].uncovered_nonempty_cells, 0)

    def test_long_text_sampling_is_explicit_and_preserves_original(self) -> None:
        text = "x" * 250
        content = ("note\n" + text + "\n").encode()
        summary = self.processor.inspect("notes.csv", content)
        self.assertTrue(any("200 caracteres" in warning and "A2" in warning
                            for warning in summary.sheets[0].warnings))
        self.assertEqual(len(summary.sheets[0].sample[-1].values[0]), 200)
        result = self.processor.execute("notes.csv", content, recipe(), summary)
        self.assertEqual(result.sheets[0].source_nonempty_cells, 1)
        self.assertEqual(result.sheets[0].preserved_nonempty_cells, 1)

    def test_repeated_header_region_includes_entire_merged_range(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        header = ["taxon", "1A", "1B", "group"]
        sheet.append(header)
        sheet.append(["A", 0, 0, 0])
        sheet.append([])
        sheet.append(header)
        sheet.merge_cells("D4:G4")
        summary = self.processor.inspect("merged.xlsx", workbook_bytes(workbook)).sheets[0]
        repeated = next(region for region in summary.regions if region.kind == "repeated_header")
        self.assertEqual((repeated.start_column, repeated.end_column), (1, 7))
        self.assertEqual(repeated.nonempty_cells, 4)
        self.assertEqual(summary.merged_cells, ["D4:G4"])

    def test_merged_header_samples_include_every_caption_with_original_columns(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        for row in (1, 4):
            for column, value in ((2, "taxon"), (3, "sample_A"), (4, "sample_B")):
                sheet.cell(row, column, value)
            for column in (10, 20, 30, 40):
                sheet.cell(row, column, f"campaign_{column}")
                sheet.merge_cells(start_row=row, end_row=row, start_column=column, end_column=column + 4)
        for column in range(2, 45):
            sheet.cell(2, column, 0)
        content = workbook_bytes(workbook)
        source = self.processor.reader.read("merged.xlsx", content)
        original = source.sheets[0].frame.copy(deep=True)
        with patch.object(self.processor.reader, "read", return_value=source):
            summary = self.processor.inspect("merged.xlsx", content).sheets[0]
        repeated = next(region for region in summary.regions if region.kind == "repeated_header")
        self.assertEqual((repeated.start_column, repeated.end_column), (2, 44))
        self.assertEqual(repeated.column_indices, [2, 3, 4, 10, 20, 30, 40])
        captions = dict(zip(repeated.column_indices, repeated.sample[0].values, strict=True))
        self.assertEqual({column: captions[column] for column in (10, 20, 30, 40)}, {
            column: f"campaign_{column}" for column in (10, 20, 30, 40)
        })
        self.assertEqual(repeated.sample[0].row_number, 4)
        self.assertTrue(source.sheets[0].frame.equals(original))

    def test_many_merged_header_captions_are_bounded_with_explicit_sampling_warning(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        for column in range(1, 101, 5):
            sheet.cell(1, column, f"campaign_{column}")
            sheet.merge_cells(start_row=1, end_row=1, start_column=column, end_column=column + 4)
        sheet.append([0] * 100)
        summary = self.processor.inspect("many_headers.xlsx", workbook_bytes(workbook)).sheets[0]
        header = next(region for region in summary.regions if region.kind == "header")
        self.assertEqual(len(header.column_indices), 16)
        self.assertEqual((header.column_indices[0], header.column_indices[-1]), (1, 96))
        self.assertEqual(header.sample[0].values, [f"campaign_{column}" for column in header.column_indices])
        self.assertTrue(any("A1:CV1" in warning and "20 textos" in warning and "16 anclas" in warning
                            for warning in summary.warnings))

    def test_uncached_header_formula_remains_visible_outside_data_rectangle(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["=1+1", "value"])
        sheet.append([0, 3])
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("header.xlsx", content)
        self.assertEqual(summary.sheets[0].missing_formula_cache_count, 1)
        self.assertEqual(summary.sheets[0].formula_sample[0].cell, "A1")
        self.assertEqual(summary.sheets[0].formula_sample[0].formula, "=1+1")
        self.assertFalse(summary.sheets[0].formula_sample[0].cache_available)
        execution = self.processor.execute("header.xlsx", content, recipe("Sheet", data_end_column=2), summary)
        self.assertEqual(execution.sheets[0].unresolved_formula_count, 0)
        self.assertEqual(execution.sheets[0].sample[0].values, [0, 3])
        self.assertEqual(execution.coverage[0].uncovered_nonempty_cells, 0)

    def test_many_missing_headers_have_one_warning_and_complete_column_mapping(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["name"])
        sheet.append(["A"] + [0] * 29)
        result = self.execute(workbook_bytes(workbook), recipe("Sheet", data_end_column=30), "wide.xlsx").sheets[0]
        self.assertEqual(result.columns_after, ["name"] + [f"column_{index}" for index in range(2, 31)])
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("29 columnas sin encabezado", result.warnings[0])
        self.assertEqual(result.sample[0].values, ["A"] + [0] * 29)

    def test_original_inclusive_coordinates_retain_values_and_conversion_locations(self) -> None:
        content = b"unused,code,value\nskip,x,9\nkeep,00123,0\nkeep,9007199254740993,2\n"
        preparation = recipe(
            data_start_row=3, data_end_row=4, data_start_column=2, data_end_column=3,
            transformations=[
                {"operation": "rename_columns", "mapping": {"code": "identifier"}},
                {"operation": "convert_type", "columns": ["identifier"], "target_type": "integer"},
            ],
        )
        block = self.execute(content, preparation).sheets[0]
        self.assertEqual((block.data_start_row, block.data_end_row), (3, 4))
        self.assertEqual((block.data_start_column, block.data_end_column), (2, 3))
        self.assertEqual([(sample.row_number, sample.values) for sample in block.sample], [
            (3, ["00123", "0"]), (4, [9007199254740993, "2"]),
        ])
        self.assertEqual(block.findings[0].cell, "B3")
        self.assertEqual(block.findings[0].block_id, "main")
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)

    def test_each_failed_conversion_retains_its_original_coordinate(self) -> None:
        content = ("value\n" + "\n".join(f"invalid_{index}" for index in range(25))).encode()
        block = self.execute(content, recipe(data_end_row=26, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])).sheets[0]
        self.assertEqual(block.conversion_failures, {"value": 25})
        self.assertEqual([finding.cell for finding in block.findings], [f"A{row}" for row in range(2, 27)])
        self.assertEqual(block.sample[-1].values, ["invalid_24"])
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)

    def test_conversion_preserves_blank_whitespace_without_an_implicit_null_conversion(self) -> None:
        content = b"label,value\nA,   \nB, \nC,\nD,0\n"
        preparation = recipe(data_end_row=5, data_end_column=2, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])
        table = self.processor.reader.read("test.csv", content)
        table.sheets[0].frame.iat[3, 1] = ""
        with patch.object(self.processor.reader, "read", return_value=table):
            with patch.object(self.processor, "_sample", wraps=self.processor._sample) as sample:
                block = self.execute(content, preparation).sheets[0]
                prepared = sample.call_args.args[0]
        self.assertEqual(prepared.loc[1, "value"], "   ")
        self.assertEqual(prepared.loc[2, "value"], " ")
        self.assertEqual(prepared.loc[3, "value"], "")
        self.assertEqual(prepared.loc[4, "value"], 0)
        self.assertEqual(block.conversion_failures, {})
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)

    def test_number_conversion_preserves_boolean_values_codes_and_scientific_precision(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["value"])
        values = [True, False, 0, "1.2500", "1e-3", "00123e0", "001.23", "1.234567890123456789e-4"]
        for value in values:
            sheet.append([value])
        block = self.execute(workbook_bytes(workbook), recipe("Sheet", data_end_row=9, transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ]), "test.xlsx").sheets[0]
        self.assertEqual([row.values[0] for row in block.sample], [
            True, False, 0, 1.25, 0.001, "00123e0", "001.23", "1.234567890123456789e-4",
        ])
        self.assertEqual(block.conversion_failures, {"value": 5})
        self.assertEqual([finding.cell for finding in block.findings], ["A2", "A3", "A7", "A8", "A9"])
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)

    def test_datetime_conversion_preserves_fractional_precision_beyond_nanoseconds(self) -> None:
        precise = "2024-01-01T12:00:00.123456789123"
        supported = "2024-01-01T12:00:00.123456789"
        content = f"record,observed_at\nprecise,{precise}\nsupported,{supported}\n".encode()
        block = self.execute(content, recipe(data_end_row=3, data_end_column=2, transformations=[
            {"operation": "rename_columns", "mapping": {"observed_at": "timestamp"}},
            {"operation": "convert_type", "columns": ["timestamp"], "target_type": "datetime"},
        ])).sheets[0]
        self.assertEqual([(row.row_number, row.values) for row in block.sample], [
            (2, ["precise", precise]), (3, ["supported", supported + "+00:00"]),
        ])
        self.assertEqual(block.conversion_failures, {"timestamp": 1})
        self.assertEqual(len(block.findings), 1)
        self.assertEqual(block.findings[0].code, "conversion_preserved")
        self.assertEqual(block.findings[0].cell, "B2")
        self.assertEqual(block.findings[0].block_id, "main")
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)

    def test_initial_unit_rows_and_percent_measurements_are_header_and_observations(self) -> None:
        for unit in ("Cel mL-1", "cells/mL"):
            with self.subTest(unit=unit):
                workbook = Workbook()
                sheet = workbook.active
                sheet.append(["site", "period", "count", "% O2"])
                sheet.append([None, None, unit, "%"])
                sheet.append(["C1", "1/2018", 0, 95])
                sheet.append(["C2", "03/04/2024", "nd", 100])
                sheet.cell(100, 8).number_format = "0.00"
                summary = self.processor.inspect("measurements.xlsx", workbook_bytes(workbook)).sheets[0]
                self.assertEqual([(region.kind, region.start_row, region.end_row,
                                   region.start_column, region.end_column) for region in summary.regions], [
                    ("header", 1, 2, 1, 4), ("observations", 3, 4, 1, 4),
                ])

    def test_not_detected_codes_and_periods_are_not_unit_headers(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["status", "code", "period"])
        sheet.append(["nd", "00123", "1/2018"])
        sheet.append(["n.d", "C1", "03/04/2024"])
        summary = self.processor.inspect("codes.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual([(region.kind, region.start_row, region.end_row) for region in summary.regions], [
            ("header", 1, 1), ("observations", 2, 3),
        ])

    def test_percent_magnitudes_stay_observations_next_to_derived_and_presence_summaries(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["taxon", "sample", "% O2", "calculation", "presencias", "%", "ratio"])
        sheet.append(["A", 0, 95, "=B2+1", "A", 0.5, "=SUM(B2:B3)*100/SUM(B2:B3)"])
        sheet.append(["B", 2, 100, "=B3+1", "B", 1, "=SUM(B2:B3)/SUM(B2:B3)*100"])
        summary = self.processor.inspect("roles.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.regions if region.start_row == 2], [
            ("observations", 1, 3), ("derived", 4, 4), ("summary", 5, 7),
        ])

    def test_percentage_formula_for_individual_measurements_does_not_prove_a_summary(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["site", "oxygen", "capacity", "% saturation"])
        sheet.append(["A", 4, 8, "=B2/C2*100"])
        sheet.append(["B", 5, 10, "=B3/C3*100"])
        summary = self.processor.inspect("saturation.xlsx", workbook_bytes(workbook)).sheets[0]
        self.assertFalse(any(region.kind == "summary" for region in summary.regions))
        self.assertEqual([(region.kind, region.start_column, region.end_column)
                          for region in summary.regions if region.start_row == 2], [
            ("observations", 1, 3), ("derived", 4, 4),
        ])

    def test_full_record_overlap_preserves_reordered_matches_exclusives_and_original_bounds(self) -> None:
        workbook = Workbook()
        left = workbook.active
        left.title = "Original"
        right = workbook.create_sheet("Ordered")
        headers = ["site", "period", "value", "code"]
        units = [None, None, "mg/L", None]
        records = [["A", "1/2018", 0, "00123"], ["B", "03/04/2024", 2, "nd"],
                   ["C", "2024-03", 3, "<0,001"]]
        for row in [headers, units] + records:
            left.append(row)
        for row in [headers, units, records[2], records[0], ["D", "2024-04", 4, "n.d"]]:
            right.append([None] + row)
        left.cell(100, 8).number_format = "0.00"
        content = workbook_bytes(workbook)
        summary = self.processor.inspect("views.xlsx", content)
        self.assertEqual(len(summary.record_overlaps), 1)
        overlap = summary.record_overlaps[0]
        self.assertEqual((overlap.left_header_rows, overlap.right_header_rows), ([1, 2], [1, 2]))
        self.assertEqual((overlap.left_start_column, overlap.left_end_column,
                          overlap.right_start_column, overlap.right_end_column), (1, 4, 2, 5))
        self.assertEqual((overlap.left_record_count, overlap.right_record_count), (3, 3))
        self.assertEqual([(match.left_row, match.right_row) for match in overlap.matches], [(3, 4), (5, 3)])
        self.assertEqual((overlap.left_only_rows, overlap.right_only_rows), ([4], [5]))
        self.assertFalse(overlap.ambiguous_repeated_rows)
        self.assertEqual(overlap.policy, "preserve_sources_do_not_add_matches")
        preparation = PreparationRecipe.model_validate({
            "status": "ready", "warnings": [], "requires_human_review": False,
            "sheets": [{
                "sheet_name": name, "block_id": name.casefold(), "role": "observations", "header_rows": [1, 2],
                "data_start_row": 3, "data_end_row": 5, "data_start_column": first,
                "data_end_column": first + 3, "transformations": [],
            } for name, first in (("Original", 1), ("Ordered", 2))],
        })
        execution = self.processor.execute("views.xlsx", content, preparation, summary)
        self.assertEqual(execution.record_overlaps, summary.record_overlaps)
        self.assertEqual([block.rows_after for block in execution.sheets], [3, 3])
        self.assertEqual([row.values for row in execution.sheets[0].sample], records)
        self.assertEqual(execution.sheets[1].sample[-1].values, ["D", "2024-04", 4, "n.d"])

    def test_overlap_compares_full_unsampled_values_types_codes_and_units(self) -> None:
        cases = (
            ("mg/L", "mg/L", "x" * 250, "x" * 249 + "y", 0),
            ("mg/L", "mg/L", True, 1, 0),
            ("mg/L", "mg/L", "00123", "123", 0),
            ("mg/L", "ug/L", "same", "same", None),
        )
        for left_unit, right_unit, left_value, right_value, expected_matches in cases:
            with self.subTest(left=left_value, right=right_value, units=(left_unit, right_unit)):
                workbook = Workbook()
                for index, (unit, value) in enumerate(((left_unit, left_value), (right_unit, right_value))):
                    sheet = workbook.active if index == 0 else workbook.create_sheet("Other")
                    sheet.append(["site", "measurement", "note"])
                    sheet.append([None, unit, None])
                    sheet.append(["A", 0, value])
                summary = self.processor.inspect("compare.xlsx", workbook_bytes(workbook))
                if expected_matches is None:
                    self.assertEqual(summary.record_overlaps, [])
                else:
                    self.assertEqual(len(summary.record_overlaps), 1)
                    overlap = summary.record_overlaps[0]
                    self.assertEqual(len(overlap.matches), expected_matches)
                    self.assertEqual((overlap.left_only_rows, overlap.right_only_rows), ([3], [3]))

    def test_overlap_compares_original_formula_expressions_not_just_missing_caches(self) -> None:
        workbook = Workbook()
        for index, expression in enumerate(("=1+1", "=1+2")):
            sheet = workbook.active if index == 0 else workbook.create_sheet("Other")
            sheet.append(["site", "measurement", "calculation"])
            sheet.append([None, "mg/L", None])
            sheet.append(["A", 0, expression])
            sheet.append(["B", 1, "literal"])
            sheet.append(["C", 2, "literal"])
        summary = self.processor.inspect("formulas.xlsx", workbook_bytes(workbook))
        overlap = summary.record_overlaps[0]
        self.assertEqual([(match.left_row, match.right_row) for match in overlap.matches], [(4, 4), (5, 5)])
        self.assertEqual((overlap.left_only_rows, overlap.right_only_rows), ([3], [3]))

    def test_overlap_preserves_internal_repetitions_as_an_ambiguous_multiset(self) -> None:
        workbook = Workbook()
        for index, records in enumerate(([["A", 0], ["A", 0], ["B", 1]], [["A", 0], ["B", 1]])):
            sheet = workbook.active if index == 0 else workbook.create_sheet("Other")
            sheet.append(["site", "measurement"])
            sheet.append([None, "mg/L"])
            for row in records:
                sheet.append(row)
        summary = self.processor.inspect("repeated.xlsx", workbook_bytes(workbook))
        overlap = summary.record_overlaps[0]
        self.assertEqual((overlap.left_record_count, overlap.right_record_count), (3, 2))
        self.assertEqual([(match.left_row, match.right_row) for match in overlap.matches], [(3, 3), (5, 4)])
        self.assertEqual((overlap.left_only_rows, overlap.right_only_rows), ([4], []))
        self.assertTrue(overlap.ambiguous_repeated_rows)

    def test_repeated_header_units_do_not_create_overlap_with_the_same_sheet(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        for row in (["site", "measurement"], [None, "mg/L"], ["A", 0], [],
                    ["site", "measurement"], [None, "mg/L"], ["A", 0]):
            sheet.append(row)
        summary = self.processor.inspect("bands.xlsx", workbook_bytes(workbook))
        self.assertEqual(summary.record_overlaps, [])
        self.assertEqual([(region.kind, region.start_row, region.end_row) for region in summary.sheets[0].regions], [
            ("header", 1, 2), ("observations", 3, 3),
            ("repeated_header", 5, 6), ("observations", 7, 7),
        ])

    def test_unknown_preserves_original_values_and_requires_review(self) -> None:
        content = b"code,value\n00123,0\nnd,<0.001\n"
        preparation = PreparationRecipe.model_validate({
            "status": "review_required", "warnings": ["No se conoce la semántica de estos códigos."],
            "requires_human_review": True,
            "sheets": [{
                "sheet_name": "data", "block_id": "unresolved", "role": "unknown", "header_rows": [1],
                "data_start_row": 2, "data_end_row": 3,
                "data_start_column": 1, "data_end_column": 2, "transformations": [],
            }],
        })
        summary = self.processor.inspect("test.csv", content)
        execution = self.processor.execute("test.csv", content, preparation, summary)
        block = execution.sheets[0]
        self.assertEqual([sample.values for sample in block.sample], [["00123", "0"], ["nd", "<0.001"]])
        self.assertEqual(block.source_nonempty_cells, block.preserved_nonempty_cells)
        self.assertEqual(block.findings[0].range, "A2:B3")
        self.assertTrue(block.findings[0].requires_review)
        validation = TabularValidationService().validate(summary, preparation, execution)
        self.assertEqual(validation.status, EvaluationStatus.REVIEW_REQUIRED)

    def test_executor_does_not_complete_missing_or_impossible_bounds(self) -> None:
        content = b"name,value\nA,0\n"
        base = recipe(data_end_column=2).sheets[0].model_dump()
        for overrides in (
            {"data_end_row": None}, {"data_end_column": None},
            {"data_start_row": 3, "data_end_row": 2},
            {"data_start_row": 3, "data_end_row": 3},
            {"data_start_column": 2, "data_end_column": 1},
            {"data_start_column": 0}, {"header_rows": [2]},
        ):
            with self.subTest(overrides=overrides):
                block = SheetRecipe.model_construct(**(base | overrides))
                preparation = PreparationRecipe.model_construct(
                    status="ready", sheets=[block], warnings=[], requires_human_review=False,
                )
                execution = self.execute(content, preparation)
                self.assertTrue(execution.sheets[0].errors)
                self.assertEqual(execution.sheets[0].sample, [])
                self.assertEqual(execution.coverage[0].covered_nonempty_cells, 0)

    def test_observations_total_repeated_header_and_legend_stay_separate(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Counts"
        header = ["taxon"] + [f"sample_{index}" for index in range(1, 9)]
        sheet.append(header)
        sheet.append(["A"] + [0] * 8)
        sheet.append(["B"] + [2] * 8)
        sheet.append(["Total"] + [2] * 8)
        sheet.append([])
        sheet.append(header)
        sheet.append([])
        sheet.append(["C1", "mar-abr 2006"])
        sheet.append(["C2", "nov 2007"])
        content = workbook_bytes(workbook)
        blocks = [
            {"sheet_name": "Counts", "block_id": identifier, "role": role, "header_rows": headers,
             "data_start_row": start, "data_end_row": end,
             "data_start_column": 1, "data_end_column": width, "transformations": []}
            for identifier, role, headers, start, end, width in (
                ("observations", "observations", [1], 2, 3, 9),
                ("total", "total", [1], 4, 4, 9),
                ("repeated", "header", [], 6, 6, 9),
                ("legend", "legend", [], 8, 9, 2),
            )
        ]
        preparation = PreparationRecipe.model_validate({
            "status": "ready", "sheets": blocks, "warnings": [], "requires_human_review": False,
        })
        summary = self.processor.inspect("counts.xlsx", content)
        execution = self.processor.execute("counts.xlsx", content, preparation, summary)
        self.assertEqual([block.role for block in execution.sheets], ["observations", "total", "header", "legend"])
        self.assertEqual([row.row_number for row in execution.sheets[0].sample], [2, 3])
        self.assertEqual(execution.sheets[0].sample[0].values, ["A"] + [0] * 8)
        self.assertEqual(execution.sheets[1].sample[0].values, ["Total"] + [2] * 8)
        self.assertEqual(execution.sheets[2].sample[0].values, header)
        self.assertEqual([row.values for row in execution.sheets[3].sample], [
            ["C1", "mar-abr 2006"], ["C2", "nov 2007"],
        ])
        coverage = execution.coverage[0]
        self.assertEqual((coverage.nonempty_cells, coverage.covered_nonempty_cells), (49, 49))
        self.assertEqual((coverage.uncovered_nonempty_cells, coverage.overlapping_nonempty_cells), (0, 0))
        self.assertTrue(all(block.source_nonempty_cells == block.preserved_nonempty_cells for block in execution.sheets))
        validation = TabularValidationService().validate(summary, preparation, execution)
        self.assertIn(validation.status, (EvaluationStatus.READY, EvaluationStatus.READY_WITH_WARNINGS))

        mixed = recipe("Counts", data_end_row=9, data_end_column=9)
        mixed_execution = self.processor.execute("counts.xlsx", content, mixed, summary)
        self.assertEqual(mixed_execution.coverage[0].uncovered_nonempty_cells, 0)
        mixed_validation = TabularValidationService().validate(summary, mixed, mixed_execution)
        self.assertNotIn(mixed_validation.status, (EvaluationStatus.READY, EvaluationStatus.READY_WITH_WARNINGS))
        self.assertTrue(mixed_validation.findings)


if __name__ == "__main__":
    unittest.main()
