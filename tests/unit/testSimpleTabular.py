import io
import unittest
from datetime import date, time
from decimal import Decimal
from zipfile import ZIP_DEFLATED, ZipFile

from openpyxl import Workbook
from openpyxl.worksheet.table import Table

from aquai.application.contracts.ingestion import StructureSuggestion
from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.processing.tabular import SimpleTabularProcessor


#
# Serializa un libro de Excel a bytes.
#
def workbook_bytes(workbook: Workbook) -> bytes:
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


class SimpleTabularTests(unittest.TestCase):
    #
    # Crea el procesador tabular con la configuración por defecto.
    #
    def setUp(self) -> None:
        self.processor = SimpleTabularProcessor(TabularSettings(_env_file=None))

    #
    # Verifica que el CSV conserva registros, códigos, marcadores, ceros, falsos y nulos.
    #
    def test_csv_preserves_all_records_codes_markers_zero_false_and_null(self) -> None:
        content = (
            b"codigo,,valor,marca,activo\n001,,0,NA,false\n002,,,nd,true\n,,,,\n002,,,nd,true\n"
        )
        result = self.processor.prepare("sample.csv", content)
        self.assertEqual(len(result.tables), 1)
        table = result.tables[0]
        self.assertEqual(len(table.rows), 3)
        self.assertEqual([row["_aquai_source_row"] for row in table.rows], [2, 3, 5])
        self.assertEqual([row["_aquai_row"] for row in table.rows], [1, 2, 3])
        self.assertEqual(table.rows[0]["codigo"], "001")
        self.assertEqual(table.rows[0]["valor"], 0)
        self.assertIs(table.rows[0]["activo"], False)
        self.assertEqual([row["marca"] for row in table.rows], ["NA", "nd", "nd"])
        self.assertIsNone(table.rows[1]["valor"])
        self.assertEqual(table.columns[1]["source_column"], 3)

    #
    # Verifica que los tipos se comprueban sobre la columna completa y los nombres son seguros.
    #
    def test_types_are_verified_over_full_column_and_names_are_safe(self) -> None:
        content = (
            "valor,valor,fecha,medida [mg/L],_aquai_row,decimal\n"
            "1.234,2,03/04/2024,0.50,a,0.12345678901234567890123456789\n"
            "1.235,nd,04/05/2024,1.25,b,0.20\n"
        ).encode()
        table = self.processor.prepare("mixed.csv", content).tables[0]
        self.assertEqual(
            [column["sql_type"] for column in table.columns],
            ["TEXT", "TEXT", "TEXT", "NUMERIC", "TEXT", "NUMERIC"],
        )
        self.assertEqual(
            [column["original_name"] for column in table.columns][:2], ["valor", "valor"]
        )
        self.assertEqual(table.columns[1]["sql_name"], "valor_2")
        self.assertEqual(table.columns[4]["sql_name"], "column_aquai_row")
        self.assertEqual(table.rows[0]["decimal"], Decimal("0.12345678901234567890123456789"))
        self.assertEqual(table.columns[3]["unit"], "mg/L")
        self.assertFalse(table.columns[0]["numeric_eligible"])
        self.assertTrue(table.warnings)

    #
    # Verifica que Excel con varias regiones y hojas conserva encabezados combinados y procedencia.
    #
    def test_excel_multiple_regions_sheets_merged_headers_and_provenance(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "mediciones"
        sheet.append(["sitio", "lecturas", None])
        sheet.append([None, "minimo", "maximo"])
        sheet.append(["A", 1, 2])
        sheet.append(["A", 1, 2])
        sheet.merge_cells("A1:A2")
        sheet.merge_cells("B1:C1")
        sheet.append([])
        sheet.append(["categoria", "cantidad"])
        sheet.append(["X", 4])
        other = workbook.create_sheet("otros")
        other.append(["sitio", "fecha"])
        other.append(["B", date(2024, 4, 3)])
        unit_sheet = workbook.create_sheet("unidades")
        unit_sheet.append(["lugar", "medida", "caudal", None])
        unit_sheet.append([None, "mg L-1", "L/s", "categoria"])
        unit_sheet.append(["A", 2, 3, "X"])
        workbook.create_sheet("vacia")
        workbook.create_sheet("documentacion").append(["Notas del experimento"])
        result = self.processor.prepare("sample.xlsx", workbook_bytes(workbook))
        self.assertEqual(len(result.tables), 4)
        self.assertEqual([len(table.rows) for table in result.tables], [2, 1, 1, 1])
        self.assertEqual(result.tables[0].columns[1]["original_name"], "lecturas | minimo")
        self.assertEqual(result.tables[0].source_locator["header_rows"], [1, 2])
        self.assertEqual(result.tables[1].rows[0]["_aquai_source_row"], 7)
        self.assertEqual(result.tables[2].columns[1]["sql_type"], "DATE")
        self.assertEqual(result.tables[3].source_locator["header_rows"], [1, 2])
        self.assertEqual(result.tables[3].columns[1]["unit"], "mg L-1")
        self.assertEqual(result.tables[3].columns[3]["original_name"], "categoria")
        self.assertTrue(any("vacía" in warning for warning in result.warnings))

    #
    # Verifica que encabezados parciales, tablas anchas y laterales conservan sus registros.
    #
    def test_partial_headers_wide_tables_and_lateral_tables_keep_records(self) -> None:
        workbook = Workbook()
        wide = workbook.active
        wide.append(["id", None] + [f"Variable {index}" for index in range(2, 300)])
        wide.append(["001", "nd"] + list(range(2, 300)))
        wide.append(["002", 5] + list(range(2, 300)))
        lateral = workbook.create_sheet("independientes")
        lateral.append(["id", "valor", None, "categoria", "cantidad"])
        lateral.append(["A", 1, None, "X", 2])
        lateral.append(["B", None, None, "Y", 3])
        levels = workbook.create_sheet("niveles")
        levels.append(["Sitio", "Mediciones", None, None])
        levels.append([None, "Sensor A", None, "Sensor B"])
        levels.append([None, "minimo", "maximo", None])
        levels.append(["A", 1, 2, 3])
        levels.merge_cells("A1:A3")
        levels.merge_cells("B1:D1")
        levels.merge_cells("B2:C2")
        levels.merge_cells("D2:D3")
        result = self.processor.prepare("general.xlsx", workbook_bytes(workbook))
        self.assertEqual([len(table.rows) for table in result.tables], [2, 2, 2, 1])
        self.assertEqual(len(result.tables[0].columns), 300)
        unnamed = result.tables[0].columns[1]
        self.assertTrue(unnamed["header_missing"])
        self.assertEqual(unnamed["header_values"], [None])
        self.assertEqual(unnamed["source_column"], 2)
        self.assertEqual(result.tables[0].rows[0][unnamed["sql_name"]], "nd")
        self.assertEqual(result.tables[-1].source_locator["header_rows"], [1, 2, 3])
        self.assertEqual(
            result.tables[-1].columns[1]["original_name"], "Mediciones | Sensor A | minimo"
        )

    #
    # Verifica que la proyección numérica conserva los marcadores originales y cuenta los excluidos.
    #
    def test_numeric_projection_keeps_original_markers_and_excluded_count(self) -> None:
        content = b"Fecha,Abundancia,codigo,periodo\n2024-03-12,120,001,03/04/2024\n2024-03-13,135,002,04/05/2024\n2024-03-14,nd,nd,05/06/2024\n2024-03-15,180,004,06/07/2024\n"
        table = self.processor.prepare("markers.csv", content).tables[0]
        original, auxiliary = table.columns[1], table.columns[-1]
        self.assertEqual(original["sql_type"], "TEXT")
        self.assertFalse(original["numeric_eligible"])
        self.assertEqual(
            [row[original["sql_name"]] for row in table.rows], ["120", "135", "nd", "180"]
        )
        self.assertEqual(auxiliary["derived_from"], original["sql_name"])
        self.assertEqual(auxiliary["excluded_count"], 1)
        self.assertEqual(auxiliary["nonempty_count"], 4)
        self.assertEqual(
            [row[auxiliary["sql_name"]] for row in table.rows],
            [Decimal(120), Decimal(135), None, Decimal(180)],
        )
        self.assertEqual(len(table.columns), 5)
        self.assertTrue(any("excluidos" in warning for warning in table.warnings))

    #
    # Verifica que la transposición estructural comprueba coordenadas y rechaza pérdida de registros.
    #
    def test_structural_transpose_verifies_coordinates_and_rejects_record_loss(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["variable", date(2024, 1, 1), date(2024, 1, 2)])
        sheet.append(["sitio", "A", "B"])
        sheet.append(["masa [mg]", 10, "nd"])
        sheet.append(["volumen [L]", 2, 4])
        content = workbook_bytes(workbook)
        initial = self.processor.prepare("matrix.xlsx", content)
        self.assertFalse(initial.tables)
        candidate = initial.structure_candidates[0]
        hint = StructureSuggestion(candidate["region_id"], "columns", 1, 1, 2)
        table = self.processor.prepare("matrix.xlsx", content, [hint]).tables[0]
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.source_locator["orientation"], "columns")
        self.assertEqual(table.source_locator["header_columns"], [1])
        self.assertEqual([column["source_row"] for column in table.columns], [1, 2, 3, 4])
        self.assertEqual([row["_aquai_source_column"] for row in table.rows], [2, 3])
        self.assertEqual(table.rows[1]["masa_mg"], "nd")
        corrected = self.processor.prepare(
            "matrix.xlsx",
            content,
            [StructureSuggestion(candidate["region_id"], "columns", 1, 2, 3)],
        ).tables[0]
        self.assertEqual(len(corrected.rows), 2)
        self.assertEqual(corrected.source_locator["structure_assistance"]["header_depth"], 1)
        self.assertEqual(
            corrected.source_locator["structure_assistance"]["suggested_header_depth"], 2
        )
        rejected = self.processor.prepare(
            "matrix.xlsx",
            content,
            [StructureSuggestion(candidate["region_id"], "columns", 2, 1, 3)],
        )
        self.assertFalse(rejected.tables)
        self.assertTrue(any("verificarse" in warning for warning in rejected.warnings))

    #
    # Verifica que Excel conserva fórmulas, códigos con formato y fechas parciales.
    #
    def test_excel_preserves_formulas_formatted_codes_and_partial_dates(self) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["etiqueta", "resultado", "numero", "periodo", "horario"])
        sheet.append(["A", "=1+2", 12, date(2024, 4, 3), time(14, 0)])
        sheet["C2"].number_format = "0000"
        sheet["D2"].number_format = "mmm-yy"
        sheet.append(["=A2", None, None, None])
        table = self.processor.prepare("formula.xlsx", workbook_bytes(workbook)).tables[0]
        self.assertEqual(table.rows[0]["resultado"], "=1+2")
        self.assertEqual(table.rows[0]["numero"], "0012")
        self.assertEqual(table.columns[3]["sql_type"], "TEXT")
        self.assertIsInstance(table.rows[0]["periodo"], str)
        self.assertEqual(table.columns[4]["sql_type"], "TEXT")
        self.assertEqual(len(table.rows), 1)
        self.assertTrue(any("fórmulas" in warning for warning in table.warnings))
        self.assertTrue(any("parcial" in warning for warning in table.warnings))

    #
    # Verifica que las regiones poco fiables no se publican y los totales declarados se excluyen.
    #
    def test_unreliable_regions_are_not_published_and_declared_totals_are_excluded(self) -> None:
        workbook = Workbook()
        notes = workbook.active
        notes.title = "notas"
        notes.append(["Una nota sobre el archivo"])
        notes.append(["Una segunda explicación"])
        matrix = workbook.create_sheet("matriz")
        matrix.append([1, 2])
        matrix.append([3, 4])
        table_sheet = workbook.create_sheet("tabla")
        table_sheet.append(["categoria", "cantidad"])
        table_sheet.append(["A", 1])
        table_sheet.append(["Total", "=SUM(B2:B2)"])
        table_sheet.add_table(Table(displayName="Tabla1", ref="A1:B3", totalsRowCount=1))
        result = self.processor.prepare("regions.xlsx", workbook_bytes(workbook))
        self.assertEqual(len(result.tables), 1)
        self.assertEqual(len(result.tables[0].rows), 1)
        self.assertGreaterEqual(len(result.warnings), 3)

    #
    # Verifica que la codificación, el delimitador y las filas físicas sobreviven a celdas multilínea.
    #
    def test_csv_encoding_delimiter_and_physical_rows_survive_multiline_cells(self) -> None:
        content = 'nombre;comentario;cantidad\nPeña;"primera\nsegunda";2\nOtro;normal;3\n'.encode(
            "cp1252"
        )
        result = self.processor.prepare("text.csv", content)
        table = result.tables[0]
        self.assertEqual(table.source_locator["encoding"], "cp1252")
        self.assertEqual(table.source_locator["delimiter"], ";")
        self.assertEqual(table.rows[0]["nombre"], "Peña")
        self.assertEqual(table.rows[0]["comentario"], "primera\nsegunda")
        self.assertEqual(table.rows[1]["_aquai_source_row"], 4)

    #
    # Verifica que la validación rechaza CSV binario y Excel inseguro o demasiado grande.
    #
    def test_validation_rejects_binary_csv_and_unsafe_or_oversized_excel(self) -> None:
        self.assertEqual(
            self.processor.validate_upload("../folder\\sample.csv", b"a,b\n1,2\n"), "sample.csv"
        )
        with self.assertRaises(TabularInputError):
            self.processor.validate_upload(None, b"a,b\n1,2\n")
        with self.assertRaises(TabularInputError):
            self.processor.validate_upload("x.csv", b"name,value\nA,\x00")
        with self.assertRaises(TabularInputError):
            self.processor.validate_upload("x.xlsx", b"not a zip")
        workbook = Workbook()
        workbook.active.append(["a", "b"])
        workbook.active.append([1, 2])
        workbook.active.cell(99999, 1000).number_format = "0.00"
        content = workbook_bytes(workbook)
        self.assertEqual(
            self.processor.validate_upload("formatted.xlsx", content), "formatted.xlsx"
        )
        self.assertEqual(len(self.processor.prepare("formatted.xlsx", content).tables[0].rows), 1)
        small = SimpleTabularProcessor(TabularSettings(_env_file=None, max_cells=3))
        with self.assertRaises(FileTooLargeError):
            small.validate_upload("x.xlsx", content)
        expanded = SimpleTabularProcessor(TabularSettings(_env_file=None, max_expanded_bytes=10))
        with self.assertRaises(FileTooLargeError):
            expanded.validate_upload("x.xlsx", content)
        unsafe = io.BytesIO()
        with (
            ZipFile(io.BytesIO(content)) as original,
            ZipFile(unsafe, "w", ZIP_DEFLATED) as changed,
        ):
            for entry in original.infolist():
                data = original.read(entry.filename)
                if entry.filename == "xl/worksheets/sheet1.xml":
                    data = b'<!DOCTYPE worksheet [<!ENTITY x "data">]>' + data
                changed.writestr(entry.filename, data)
        with self.assertRaises(TabularInputError):
            self.processor.validate_upload("x.xlsx", unsafe.getvalue())


if __name__ == "__main__":
    unittest.main()
