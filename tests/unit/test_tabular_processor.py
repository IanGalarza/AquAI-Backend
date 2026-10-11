"""Pruebas del lector y de las operaciones declarativas sin servicios externos."""

import io
import unittest
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook

from aquai.application.dtos.tabular import PreparationRecipe
from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.infrastructure.processing.tabular import PandasTabularProcessor


def workbook_bytes(workbook: Workbook) -> bytes:
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def recipe(sheet_name: str = "data", **overrides: object) -> PreparationRecipe:
    sheet = {
        "sheet_name": sheet_name, "header_rows": [1], "data_start_row": 2,
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

    def test_zero_values_survive_empty_row_and_column_operations(self) -> None:
        result = self.execute(b"a,b\n0,0\n,,\n", recipe(transformations=[
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
        result = self.execute(content, recipe(transformations=[
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
        self.assertEqual(result.sample[-1].values, ["Luis", None])
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
        result = self.processor.execute("test.xlsx", content, recipe("Counts", header_rows=[1, 2], data_start_row=3), summary).sheets[0]
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
        result = self.processor.execute(
            "formulas.xlsx", content,
            recipe("Sheet", transformations=[
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
            recipe(data_start_row=10),
            recipe(data_end_row=10),
            recipe(transformations=[{"operation": "rename_columns", "mapping": {"a": "b"}}]),
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
        self.assertEqual(result.sample[0].values, [None])
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
        result = self.execute(b"value\n9007199254740993\n1.5\n", recipe(transformations=[
            {"operation": "convert_type", "columns": ["value"], "target_type": "number"},
        ])).sheets[0]
        self.assertEqual(result.conversion_failures, {"value": 1})
        self.assertEqual([row.values[0] for row in result.sample], [None, 1.5])
        exact = self.execute(b"value\n9007199254740993\n\n", recipe(transformations=[
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
        result = self.processor.execute("test.xlsx", content, recipe("Sheet"), summary).sheets[0]
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
            ("integer", "2\n2.5\n9223372036854775808\n", [2, None, None], 2),
            ("number", "2.5\ninf\nbad\n", [2.5, None, None], 2),
            ("boolean", "true\nsí\nwrong\n", [True, True, None], 1),
            ("datetime", "2026-10-10\nbad\n", ["2026-10-10T00:00:00+00:00", None], 1),
        )
        for target_type, rows, expected, failures in cases:
            with self.subTest(target_type=target_type):
                result = self.execute(("value\n" + rows).encode(), recipe(transformations=[{"operation": "convert_type", "columns": ["value"], "target_type": target_type}])).sheets[0]
                self.assertEqual([row.values[0] for row in result.sample], expected)
                self.assertEqual(result.conversion_failures, {"value": failures})


if __name__ == "__main__":
    unittest.main()
