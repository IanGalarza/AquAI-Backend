import unittest
from collections import Counter
from math import radians
from types import SimpleNamespace
from unittest.mock import patch

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableData,
)

from aquai.application.contracts.documents import PreparedElement
from aquai.application.Rules.documents import (
    SUSPECT_MICRO_UNIT,
    UNMAPPED_CHARACTER,
    caption_label,
    loose_caption_label,
)
from aquai.infrastructure.processing.documents.chunking import _printed_text
from aquai.infrastructure.processing.documents.pageText import (
    Glyph,
    Segment,
    _glyph_text,
    _visible_box,
    _visible_turn,
    aligned_rows,
    glyphs_in_box,
    is_value_row,
    line_segments,
    page_glyphs,
    upright,
    visual_lines,
)
from aquai.infrastructure.processing.documents.structure import (
    _caption_gap,
    attach_loose_captions,
    build_outline,
    clean_text,
    detach_merged_captions,
    document_doi,
    join_split_ranges,
    release_unnamed_captions,
    restore_headings,
)
from aquai.infrastructure.processing.documents.symbols import (
    MICRO_SIGN,
    PageSymbols,
    SymbolWord,
    _agreed_symbol,
    _read_symbol,
    _replaced,
    _standard_symbol,
    _unit_marks,
    _unit_words,
)
from aquai.infrastructure.processing.documents.tableCheck import (
    TableCheck,
    _add_missing_lines,
    _is_rebuildable,
    _read_from_image,
    _rebuild,
    _review_against_image,
    check_table,
)
from aquai.infrastructure.processing.documents.textOrder import _same_text

_CHAR_WIDTH = 5.0
_LINE_HEIGHT = 10.0
_BOX = [0.0, 0.0, 400.0, 100.0]
_PAGE_TEXT = "aquai.infrastructure.processing.documents.pageText.pdfium_raw"
_RENDER_BOX = "aquai.infrastructure.processing.documents.tableCheck.render_box"


#
# Ubica un texto en una línea de la página, carácter por carácter.
#
def glyphs(text: str, left: float, top: float = 0.0, gap: float = 0.0) -> list[Glyph]:
    placed = []
    for char in text:
        placed.append(
            Glyph(char, left, top, left + _CHAR_WIDTH, top + _LINE_HEIGHT, "Font")
        )
        left += _CHAR_WIDTH + gap

    return placed


#
# Ubica un elemento en la primera página, con origen arriba a la izquierda.
#
def located(left: float, top: float, right: float, bottom: float) -> SimpleNamespace:
    box = BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    return SimpleNamespace(prov=[SimpleNamespace(page_no=1, bbox=box)])


#
# Arma la procedencia de un texto en la primera página.
#
def provenance(top: float = 100.0) -> ProvenanceItem:
    box = BoundingBox(l=50, t=top, r=300, b=top + 20, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=1, bbox=box, charspan=(0, 1))


#
# Arma un documento de una página.
#
def document() -> DoclingDocument:
    doc = DoclingDocument(name="prueba")
    doc.add_page(page_no=1, size=Size(width=600, height=800))
    return doc


#
# Arma una palabra sospechosa con un único carácter dudoso.
#
def word(text: str, position: int, before: str = "", after: str = "") -> SymbolWord:
    key = ("Font", text[position])
    return SymbolWord(1, text, {position: key}, 0, 0, 10, 10, before, after)


class LineSegmentsTests(unittest.TestCase):
    #
    # Verifica que las letras contiguas siguen juntas y que una distancia chica es un espacio.
    #
    def test_spaces_come_from_the_distance_between_characters(self) -> None:
        line = [*glyphs("20", 0), *glyphs("000", 12), *glyphs("cells", 28)]

        self.assertEqual([item.text for item in line_segments(line)], ["20 000 cells"])

    #
    # Verifica que un espacio del PDF fuera de lugar no parte una palabra.
    #
    def test_misplaced_pdf_space_does_not_split_a_word(self) -> None:
        stray = Glyph(" ", 2, 0, 4, _LINE_HEIGHT, "Font")
        inside = glyphs_in_box([*glyphs("Evidence", 0), stray], _BOX)
        lines = [line_segments(line) for line in visual_lines(inside)]

        self.assertEqual([[item.text for item in line] for line in lines], [["Evidence"]])

    #
    # Verifica que una distancia grande separa celdas.
    #
    def test_wide_gap_starts_a_new_segment(self) -> None:
        line = [*glyphs("pH", 0), *glyphs("7,8", 60)]

        self.assertEqual([item.text for item in line_segments(line)], ["pH", "7,8"])

    #
    # Verifica que un exponente no se pega a su base: 10⁵ no puede leerse 105.
    #
    def test_superscript_does_not_join_its_base(self) -> None:
        line = [*glyphs("10", 0), *glyphs("5", 10, top=-3.0)]

        self.assertEqual(line_segments(line)[0].text, "10 5")


class PageGeometryTests(unittest.TestCase):
    #
    # Verifica que las coordenadas son relativas al área visible cuando el recorte no empieza en 0.
    #
    def test_glyphs_are_relative_to_the_visible_page(self) -> None:
        textpage = SimpleNamespace(
            raw=None, count_chars=lambda: 1, get_charbox=lambda *_a, **_k: (50, 700, 55, 712)
        )
        page = SimpleNamespace(
            get_bbox=lambda: (40, 40, 652, 832),
            get_rotation=lambda: 0,
            get_textpage=lambda: textpage,
        )
        raw = SimpleNamespace(
            FPDFText_GetFontInfo=lambda *_a: 0,
            FPDFText_GetUnicode=lambda *_a: ord("m"),
            FPDFText_GetCharAngle=lambda *_a: 0,
        )

        with patch(_PAGE_TEXT, raw):
            glyph = page_glyphs(page)[0]

        self.assertEqual((glyph.left, glyph.top), (10, 120))

    #
    # Verifica que un recuadro cae donde se ve en una página girada: la esquina inferior
    # izquierda de una hoja de 100 x 200 queda arriba a la izquierda al girarla 90 grados.
    #
    def test_boxes_follow_the_page_rotation(self) -> None:
        box = (0, 0, 10, 5)
        page = (0, 0, 100, 200)

        self.assertEqual(_visible_box(box, page, 0), (0, 195, 10, 200))
        self.assertEqual(_visible_box(box, page, 90), (0, 0, 5, 10))
        self.assertEqual(_visible_box(box, page, 180), (90, 0, 100, 5))
        self.assertEqual(_visible_box(box, page, 270), (195, 90, 200, 100))

    #
    # Verifica que el giro de la página se suma al del texto y que un escaneo torcido no cuenta.
    #
    def test_turn_is_measured_on_the_visible_page(self) -> None:
        self.assertEqual(_visible_turn(radians(270), 90), 0)
        self.assertEqual(_visible_turn(radians(270), 0), 270)
        self.assertEqual(_visible_turn(radians(359), 0), 0)
        self.assertEqual(_visible_turn(radians(6), 0), 0)

    #
    # Verifica que lo que no es texto válido no llega a JSON ni a PostgreSQL.
    #
    def test_null_and_surrogates_are_not_kept_as_text(self) -> None:
        self.assertEqual(_glyph_text(0), "�")
        self.assertEqual(_glyph_text(0xD835), "�")
        self.assertEqual(_glyph_text(0xD835, 0xDC37), chr(0x1D437))
        self.assertEqual(_glyph_text(ord("µ")), "µ")


class RotatedTextTests(unittest.TestCase):
    #
    # Ubica un texto que se lee de abajo hacia arriba, en una columna de la página.
    #
    def sideways(self, text: str, left: float, bottom: float) -> list[Glyph]:
        placed = []
        for char in text:
            top = bottom - _CHAR_WIDTH
            placed.append(Glyph(char, left, top, left + _LINE_HEIGHT, bottom, "Font", 270))
            bottom = top

        return placed

    #
    # Verifica que un texto girado se lee en orden, con cada columna de la hoja como una línea.
    #
    def test_upright_turns_columns_into_lines(self) -> None:
        page = [*self.sideways("Sitio", 0, 300), *self.sideways("DL", 20, 300)]

        lines = [line_segments(line) for line in visual_lines(upright(page))]

        self.assertEqual([[item.text for item in line] for line in lines], [["Sitio"], ["DL"]])

    #
    # Verifica que una tabla de costado se contrasta y siempre queda en duda: Docling la leyó sin
    # enderezar, aunque sus letras alcancen para que cada línea encuentre una fila.
    #
    def test_rotated_table_is_checked_and_never_trusted(self) -> None:
        page = [
            *self.sideways("Coelastrum", 0, 300),
            *self.sideways("207", 0, 200),
            *self.sideways("Dolichospermum", 20, 300),
            *self.sideways("649", 20, 200),
        ]
        box = [-5, 0, 60, 310]
        right = [["Coelastrum", "207"], ["Dolichospermum", "649"]]
        crossed = [["Coelastrum", "649"], ["Dolichospermum", "207"]]

        self.assertEqual(check_table(page, box, right).broken_lines, 0)
        self.assertTrue(check_table(page, box, right).misaligned)
        self.assertEqual(check_table(page, box, crossed).broken_lines, 2)

    #
    # Verifica que un texto mayormente girado se endereza por su giro, sin importar lo demás.
    #
    def test_upright_ignores_horizontal_text(self) -> None:
        page = [*self.sideways("AB", 0, 300), *glyphs("x", 0), *glyphs("y", 0)]

        self.assertLess(upright(page)[0].left, upright(page)[1].left)


class AlignedRowsTests(unittest.TestCase):
    #
    # Verifica que una marca conserva su columna: en estas tablas la posición es el dato.
    #
    def test_marks_keep_their_column(self) -> None:
        lines = [
            [Segment("Organism", 0, 40), Segment("Mild", 100, 120), Segment("Severe", 200, 230)],
            [Segment("Sharks", 0, 30), Segment("x", 105, 110), Segment("x", 205, 210)],
            [Segment("Barracudas", 0, 50), Segment("x", 205, 210)],
            [Segment("Urchins", 0, 35), Segment("x", 105, 110)],
        ]

        rows = aligned_rows(lines)

        self.assertEqual(rows[2], ["Barracudas", "", "x"])
        self.assertEqual(rows[3], ["Urchins", "x", ""])

    #
    # Verifica que un nombre partido en dos líneas vuelve a la fila de sus valores.
    #
    def test_wrapped_name_joins_its_row(self) -> None:
        lines = [
            [Segment("Sitio", 0, 40), Segment("Valor", 100, 120), Segment("Otro", 200, 230)],
            [Segment("Dictyosphaerium", 0, 60)],
            [Segment("7,61", 100, 120), Segment("2,49", 200, 220)],
            [Segment("ehrenbergianum", 0, 60)],
            [Segment("Cyclotella", 0, 40), Segment("6,98", 100, 120), Segment("2,29", 200, 220)],
        ]

        rows = aligned_rows(lines)

        self.assertEqual(rows[1], ["Dictyosphaerium ehrenbergianum", "7,61", "2,49"])
        self.assertEqual(len(rows), 3)

    #
    # Verifica que un título que cruza columnas no las une.
    #
    def test_spanning_title_does_not_merge_columns(self) -> None:
        lines = [
            [Segment("Water quality category", 100, 230)],
            [Segment("Site", 0, 20), Segment("A", 100, 110), Segment("B", 200, 210)],
            [Segment("One", 0, 20), Segment("7", 100, 110), Segment("9", 200, 210)],
        ]

        self.assertEqual(aligned_rows(lines)[2], ["One", "7", "9"])


class TableCheckTests(unittest.TestCase):
    #
    # Verifica que una tabla que coincide con su página no se marca como desalineada.
    #
    def test_matching_rows_are_not_misaligned(self) -> None:
        page = [
            *glyphs("Sitio", 0, 0),
            *glyphs("Valor", 100, 0),
            *glyphs("DL", 0, 20),
            *glyphs("533", 100, 20),
        ]
        cells = [["Sitio", "Valor"], ["DL", "533"]]

        self.assertFalse(check_table(page, _BOX, cells).misaligned)

    #
    # Verifica que basta una línea con valores en otra fila para desconfiar de la tabla.
    #
    def test_one_value_on_the_wrong_row_is_misaligned(self) -> None:
        page = [
            *glyphs("Coelastrum", 0, 0),
            *glyphs("207", 100, 0),
            *glyphs("Dolichospermum", 0, 20),
            *glyphs("649", 100, 20),
        ]
        cells = [["Coelastrum", "649"], ["Dolichospermum", "207"]]

        self.assertTrue(check_table(page, _BOX, cells).misaligned)

    #
    # Verifica que un nombre largo no tapa un valor que pertenece a otra fila.
    #
    def test_long_name_does_not_hide_a_crossed_value(self) -> None:
        page = [
            *glyphs("Microcystis aeruginosa", 0, 0),
            *glyphs("7", 200, 0),
            *glyphs("Dolichospermum circinale", 0, 20),
            *glyphs("12", 200, 20),
        ]
        cells = [["Microcystis aeruginosa", "12"], ["Dolichospermum circinale", "7"]]

        self.assertTrue(check_table(page, _BOX, cells).misaligned)

    #
    # Verifica que se detectan valores con los mismos dígitos en filas distintas.
    #
    def test_values_with_the_same_digits_are_told_apart(self) -> None:
        page = [
            *glyphs("Coelastrum", 0, 0),
            *glyphs("207", 100, 0),
            *glyphs("Dolichospermum", 0, 20),
            *glyphs("702", 100, 20),
        ]
        cells = [["Coelastrum", "702"], ["Dolichospermum", "207"]]

        self.assertTrue(check_table(page, _BOX, cells).misaligned)

    #
    # Verifica que se detectan dos valores intercambiados dentro de una misma fila.
    #
    def test_values_swapped_within_a_row_are_misaligned(self) -> None:
        page = [*glyphs("DL", 0, 0), *glyphs("1.2", 100, 0), *glyphs("7.5", 200, 0)]

        self.assertTrue(check_table(page, _BOX, [["DL", "7.5", "1.2"]]).misaligned)

    #
    # Verifica que se detecta un decimal movido, aunque los dígitos sean los mismos.
    #
    def test_moved_decimal_point_is_misaligned(self) -> None:
        page = [*glyphs("DL", 0, 0), *glyphs("1.325", 100, 0)]

        self.assertTrue(check_table(page, _BOX, [["DL", "1325"]]).misaligned)

    #
    # Verifica que el orden de subíndices en un rótulo no cuenta como un valor intercambiado.
    #
    def test_digits_in_a_label_are_not_values(self) -> None:
        page = [*glyphs("PO 3 4", 0, 0), *glyphs("0.3", 100, 0)]

        self.assertFalse(check_table(page, _BOX, [["PO4 3", "0.3"]]).misaligned)

    #
    # Verifica que una fila de varias líneas sigue coincidiendo línea por línea.
    #
    def test_row_split_in_two_lines_keeps_matching(self) -> None:
        page = [
            *glyphs("Sitio", 0, 0),
            *glyphs("7", 100, 0),
            *glyphs("9", 200, 0),
            *glyphs("Norte", 0, 20),
            *glyphs("8", 100, 20),
        ]
        cells = [["Sitio Norte", "7 8", "9"]]

        self.assertEqual(check_table(page, _BOX, cells).broken_lines, 0)

    #
    # Verifica que un valor repetido en una fila no la hace parecer desalineada.
    #
    def test_repeated_values_in_a_row_are_not_misaligned(self) -> None:
        page = [
            *glyphs("DL", 0, 0),
            *glyphs("1.2", 100, 0),
            *glyphs("1.2", 200, 0),
            *glyphs("1.3", 300, 0),
        ]

        self.assertFalse(check_table(page, _BOX, [["DL", "1.2", "1.2", "1.3"]]).misaligned)

    #
    # Verifica que un encabezado con una celda combinada no tapa lo que le falta a la grilla.
    #
    def test_spanned_header_does_not_hide_a_missing_label(self) -> None:
        page = [*glyphs("Feature", 0, 0), *glyphs("Performance index", 100, 0)]
        cells = [["Performance index"] * 5]

        self.assertEqual(check_table(page, _BOX, cells, header_rows=1).broken_lines, 1)

    #
    # Verifica que sin ninguna fila encontrada ninguna línea pasa por título.
    #
    def test_lines_are_not_titles_if_no_row_matches(self) -> None:
        page = [*glyphs("zzzzzz", 0, 0), *glyphs("qqqqqq", 0, 20)]

        self.assertEqual(check_table(page, _BOX, [["Sitio", "533"]]).broken_lines, 2)

    #
    # Verifica que un título sobre la tabla no cuenta como una línea sin fila.
    #
    def test_title_above_the_table_is_not_a_broken_line(self) -> None:
        page = [*glyphs("Anexo continuacion", 0, 0), *glyphs("DL", 0, 20), *glyphs("533", 100, 20)]

        self.assertEqual(check_table(page, _BOX, [["DL", "533"]]).broken_lines, 0)

    #
    # Verifica que un encabezado de varias líneas no se toma por filas colapsadas.
    #
    def test_multiline_header_is_not_a_collapsed_row(self) -> None:
        page = []
        for top in (0, 15, 30):
            page += [*glyphs("Co", 0, top), *glyphs("Te", 100, top), *glyphs("Tu", 200, top)]
        cells = [["CoCoCo", "TeTeTe", "TuTuTu"]]

        self.assertEqual(check_table(page, _BOX, cells, header_rows=1).collapsed_rows, 0)
        self.assertEqual(check_table(page, _BOX, cells, header_rows=0).collapsed_rows, 1)

    #
    # Verifica que una fila que se quedó con media tabla deja la tabla en duda.
    #
    def test_row_absorbing_half_the_table_is_misaligned(self) -> None:
        page = []
        for top, order in enumerate(("Chroococcales", "Nostocales", "Oscillatoriales", "Synechoco")):
            page += glyphs(order, 0, top * 15)
        cells = [["Orden"], ["Chroococcales Nostocales Oscillatoriales Synechoco"]]

        self.assertTrue(check_table(page, _BOX, cells, header_rows=1).misaligned)

    #
    # Verifica que una fila que la grilla perdió se distingue de una línea cruzada.
    #
    def test_lost_row_is_reported_as_missing(self) -> None:
        page = [
            *glyphs("WHO", 0, 0),
            *glyphs("World Health", 100, 0),
            *glyphs("WTO", 0, 20),
            *glyphs("World Tourism", 100, 20),
        ]

        check = check_table(page, _BOX, [["WHO", "World Health"]])

        self.assertEqual(check.missing_lines, [["WTO", "World Tourism"]])

    #
    # Verifica que sin caracteres en la región no se afirma nada sobre la tabla.
    #
    def test_table_without_page_text_is_not_verifiable(self) -> None:
        self.assertFalse(check_table([], _BOX, [["a"]]).verifiable)

    #
    # Verifica que sólo se rehace una tabla si una fila entera la contradice, con celdas breves
    # y líneas que cubren la grilla.
    #
    def test_only_contradicted_brief_and_covered_tables_are_rebuilt(self) -> None:
        values = [[Segment("DL (7)", 0, 20), Segment("533", 50, 60)]]
        prose = [[Segment("Short-term adverse health outcomes", 0, 90)]]

        self.assertTrue(_is_rebuildable(TableCheck(values, 1, 0, 1, covered_share=1.0)))
        self.assertFalse(_is_rebuildable(TableCheck(values, 1, 0, 0, covered_share=1.0)))
        fewer_columns = TableCheck(values, 1, 0, 1, covered_share=1.0, grid_columns=4)
        self.assertFalse(_is_rebuildable(fewer_columns))
        self.assertFalse(_is_rebuildable(TableCheck(prose, 1, 0, 1, covered_share=1.0)))
        self.assertFalse(_is_rebuildable(TableCheck(values, 1, 0, 1, covered_share=0.5)))

    #
    # Verifica que rehacer una tabla conserva sus notas y su encabezado.
    #
    def test_rebuild_keeps_notes_and_header(self) -> None:
        element = PreparedElement(
            position=0, kind="table", cells=[["a", "b"]], content="| a | b |\nnota al pie"
        )
        lines = [
            [Segment("Sitio", 0, 20), Segment("Valor", 100, 120)],
            [Segment("DL", 0, 10), Segment("533", 100, 115)],
        ]

        _rebuild(element, TableCheck(lines, 1, 0, header_lines=1))

        self.assertEqual(element.cells, [["Sitio", "Valor"], ["DL", "533"]])
        self.assertEqual(element.header_rows, 1)
        self.assertTrue(element.content.endswith("nota al pie"))

    #
    # Verifica que una tabla dudosa recupera sus líneas perdidas sin perder sus notas.
    #
    def test_uncertain_table_recovers_lost_lines(self) -> None:
        element = PreparedElement(
            position=0, kind="table", cells=[["a", "b"]], content="| a | b |\nnota"
        )

        _add_missing_lines(element, TableCheck([], 1, 0, missing_lines=[["WTO", "World"]]))

        self.assertEqual(element.cells, [["a", "b"], ["WTO", "World"]])
        self.assertEqual(element.content, "| a | b |\n| WTO | World |\nnota")

    #
    # Verifica que una línea recuperada conserva el ancho de la grilla.
    #
    def test_recovered_line_keeps_the_grid_width(self) -> None:
        element = PreparedElement(position=0, kind="table", cells=[["a", "b", "c"]])

        _add_missing_lines(element, TableCheck([], 1, 0, missing_lines=[["WTO"]]))

        self.assertEqual(element.cells[1], ["WTO", "", ""])


class ScannedTableTests(unittest.TestCase):
    #
    # Prepara una tabla cuya capa de texto tiene una sola línea.
    #
    def setUp(self) -> None:
        self.element = PreparedElement(
            position=0, kind="table", cells=[["a"]], content="| a |",
            locations=[{"page": 1, "bbox": [0, 0, 100, 100]}],
        )
        self.check = TableCheck([[Segment("a", 0, 5)]], 0, 0)

    #
    # Verifica que la imagen reemplaza a la capa de texto si muestra muchas más líneas.
    #
    def test_image_with_more_lines_replaces_the_table(self) -> None:
        boxes = [*glyphs("a", 0, 0), *glyphs("2", 0, 20), *glyphs("3", 0, 40)]

        with patch(_RENDER_BOX, return_value=None):
            read = _read_from_image(None, self.element, self.check, lambda _: boxes)

        self.assertTrue(read)
        self.assertEqual(self.element.cells, [["a"], ["2"], ["3"]])

    #
    # Verifica que una lectura que no contiene el texto de la grilla no la reemplaza.
    #
    def test_foreign_reading_keeps_the_table(self) -> None:
        boxes = [*glyphs("x", 0, 0), *glyphs("y", 0, 20), *glyphs("z", 0, 40)]

        with patch(_RENDER_BOX, return_value=None):
            read = _read_from_image(None, self.element, self.check, lambda _: boxes)

        self.assertFalse(read)

    #
    # Revisa una tabla sin capa de texto contra una lectura de su imagen.
    #
    def reviewed(self, boxes: list[Glyph]) -> dict[str, list]:
        element = PreparedElement(
            position=0, kind="table", cells=[["Sitio DL", "533"]],
            locations=[{"page": 1, "bbox": [0, 0, 100, 100]}],
        )
        review: dict[str, list] = {"uncertain": [], "unverifiable": []}

        with patch(_RENDER_BOX, return_value=None):
            _review_against_image(None, element, {"page": 1}, review, lambda _: boxes)

        self.assertEqual(element.cells, [["Sitio DL", "533"]])
        return review

    #
    # Verifica que una lectura que coincide no deja avisos y que nunca se reescribe la grilla.
    #
    def test_matching_image_reading_leaves_no_doubt(self) -> None:
        review = self.reviewed([*glyphs("Sitio DL", 0), *glyphs("533", 100)])

        self.assertEqual(review, {"uncertain": [], "unverifiable": []})

    #
    # Verifica que un valor distinto en la imagen deja la duda, sin elegir entre dos lecturas.
    #
    def test_differing_image_reading_keeps_the_doubt(self) -> None:
        review = self.reviewed([*glyphs("Sitio DL", 0), *glyphs("538", 100)])

        self.assertEqual(len(review["uncertain"]), 1)

    #
    # Verifica que una lectura ajena o vacía no cuenta como contraste.
    #
    def test_foreign_or_empty_reading_is_not_a_check(self) -> None:
        foreign = [*glyphs("zzzzzz", 0, 0), *glyphs("qqqqqq", 0, 20)]

        self.assertEqual(len(self.reviewed(foreign)["unverifiable"]), 1)
        self.assertEqual(len(self.reviewed([])["unverifiable"]), 1)

    #
    # Verifica que una imagen sin líneas de más deja la tabla tal como está.
    #
    def test_image_without_extra_lines_keeps_the_table(self) -> None:
        with patch(_RENDER_BOX, return_value=None):
            read = _read_from_image(None, self.element, self.check, lambda _: glyphs("1", 0))

        self.assertFalse(read)
        self.assertEqual(self.element.cells, [["a"]])


class SymbolTests(unittest.TestCase):
    #
    # Verifica que las fuentes de símbolos siguen su estándar, también en el rango de uso privado.
    #
    def test_symbol_fonts_follow_their_standard_encoding(self) -> None:
        self.assertEqual(_standard_symbol("ABCDEF+Symbol", "m"), MICRO_SIGN)
        self.assertEqual(_standard_symbol("SymbolMT", chr(0xF06D)), MICRO_SIGN)
        self.assertEqual(_standard_symbol("ABCDEF+AdvPSMSAM10", "6"), "≤")

    #
    # Verifica que una fuente común nunca se reinterpreta, aunque su nombre mencione símbolos.
    #
    def test_other_fonts_are_not_decoded(self) -> None:
        self.assertIsNone(_standard_symbol("AGaramond-Regular", "m"))
        self.assertIsNone(_standard_symbol("TeX_CM_Maths_Symbols", "m"))
        self.assertIsNone(_standard_symbol("NotoSansSymbols2", "m"))

    #
    # Verifica que la lectura de la página resuelve el carácter si la palabra coincide.
    #
    def test_reading_resolves_the_symbol_of_a_matching_word(self) -> None:
        suspect = word("lm", 0, "of1.2", "was")

        self.assertEqual(_read_symbol(suspect, 0, "a pore of 1.2 µm was used"), MICRO_SIGN)

    #
    # Verifica que una lectura ambigua o ajena no resuelve nada: el símbolo queda en duda.
    #
    def test_ambiguous_or_foreign_reading_keeps_the_doubt(self) -> None:
        self.assertIsNone(_read_symbol(word("lg", 0), 0, "10 µg and 20 µg"))
        self.assertIsNone(_read_symbol(word("lg", 0), 0, "ten grams"))

    #
    # Verifica que cambiar una letra declarada exige lecturas unánimes: puede ser una m real.
    #
    def test_declared_letter_needs_unanimous_readings(self) -> None:
        key = ("Font", "m")

        self.assertEqual(_agreed_symbol(key, Counter("µµµ"), 5), MICRO_SIGN)
        self.assertIsNone(_agreed_symbol(key, Counter("µµµµm"), 5))
        self.assertIsNone(_agreed_symbol(key, Counter("µ"), 1))

    #
    # Verifica que un carácter sin equivalente se resuelve con un acuerdo amplio.
    #
    def test_unmapped_character_needs_broad_agreement(self) -> None:
        key = ("Font", "\x03")

        self.assertEqual(_agreed_symbol(key, Counter("−−−−−−x"), 7), "−")
        self.assertIsNone(_agreed_symbol(key, Counter("−x"), 2))

    #
    # Verifica que el reemplazo respeta los límites de palabra.
    #
    def test_replacement_does_not_reach_inside_another_word(self) -> None:
        self.assertIsNone(_replaced("The algae grew", word("lg", 0), "µg"))
        self.assertEqual(_replaced("of 10 lg per L", word("lg", 0), "µg"), "of 10 µg per L")

    #
    # Verifica que un carácter sin equivalente nunca se confunde con un valor legítimo.
    #
    def test_unmapped_character_never_matches_a_real_value(self) -> None:
        self.assertEqual(_replaced("7", word("\ufffd", 0), "≤"), "7")
        self.assertEqual(_replaced("valores 25 y 35", word("\x035", 0), "−5"), "valores 25 y 35")
        self.assertEqual(_replaced("pH \ufffd 7", word("\ufffd", 0), "≤"), "pH ≤ 7")

    #
    # Verifica que el entorno se acorta cuando ya tiene otra palabra corregida.
    #
    def test_replacement_tolerates_a_corrected_neighbour(self) -> None:
        suspect = word("6", 0, "(>5", "20lgL")

        self.assertEqual(_replaced("a (>5 620 µgL) 6", suspect, "≤"), "a (>5 ≤20 µgL) 6")

    #
    # Verifica que las unidades dudosas se buscan sólo junto a un número o un paréntesis.
    #
    def test_unit_candidates_need_a_number_or_parenthesis(self) -> None:
        self.assertEqual(
            [item.text for item in _unit_words(1, glyphs("0.22 pm and the pg", 0))], ["pm"]
        )

    #
    # Verifica que la lectura distingue la µ de la letra en la imagen.
    #
    def test_unit_reading_tells_micro_from_the_letter(self) -> None:
        self.assertEqual(_unit_marks(word("pg", 0, "("), "(μg L-1) (NTU)"), {"μ"})
        self.assertEqual(_unit_marks(word("pg", 0, "0"), "10 pg per litre"), {"p"})
        self.assertEqual(_unit_marks(word("pg", 0, "0"), "illegible"), set())

    #
    # Verifica que la lectura de una unidad vecina no decide sobre la unidad dudosa.
    #
    def test_neighbouring_unit_does_not_decide(self) -> None:
        self.assertEqual(_unit_marks(word("pm", 0, "2"), "0.22 ?? and 5 µm"), set())

    #
    # Verifica que una corrección puntual se aplica a ese carácter y no a su fuente.
    #
    def test_placed_symbol_only_changes_that_glyph(self) -> None:
        first, second = glyphs("pp", 0)
        symbols = PageSymbols(by_place={(1, first.left, first.top): MICRO_SIGN})

        self.assertEqual([item.text for item in symbols.applied(1, [first, second])], ["µ", "p"])

    #
    # Verifica que las unidades con µ mal leída quedan señaladas, sin tocar palabras comunes.
    #
    def test_suspect_units_are_flagged(self) -> None:
        self.assertTrue(SUSPECT_MICRO_UNIT.search("0.22 pm"))
        self.assertTrue(SUSPECT_MICRO_UNIT.search("(pg L-1)"))
        self.assertTrue(SUSPECT_MICRO_UNIT.search("10 lg"))
        self.assertIsNone(SUSPECT_MICRO_UNIT.search("the pm shift at 5 p.m."))

    #
    # Verifica que los controles sin equivalente cuentan, pero no el corte de línea de PDFium.
    #
    def test_unmapped_characters_include_controls(self) -> None:
        self.assertTrue(UNMAPPED_CHARACTER.search("a\x18b"))
        self.assertIsNone(UNMAPPED_CHARACTER.search("a\x02b"))


class FigureTextTests(unittest.TestCase):
    #
    # Verifica que el texto de una figura se indexa entero, con sus valores sueltos.
    #
    def test_printed_values_are_kept(self) -> None:
        content = "Punta Piedras\npH: 8,4 ± 0,9\nTurbidez [UNT]: 772\n184.48\n202.85"

        self.assertEqual(_printed_text(content), content)

    #
    # Verifica que una figura con sólo un par de rótulos no genera un fragmento.
    #
    def test_figure_with_few_words_is_not_indexed(self) -> None:
        self.assertEqual(_printed_text("Eje\n10\n20"), "")


class CaptionGapTests(unittest.TestCase):
    #
    # Prepara un documento de una página con una figura en la mitad izquierda.
    #
    def setUp(self) -> None:
        page = SimpleNamespace(size=SimpleNamespace(height=800))
        self.doc = SimpleNamespace(pages={1: page})
        self.figure = located(50, 100, 250, 300)

    #
    # Verifica que una leyenda debajo de la figura se mide en vertical.
    #
    def test_caption_below_the_figure(self) -> None:
        caption = located(50, 310, 250, 330).prov[0]

        self.assertEqual(_caption_gap(self.doc, caption, self.figure), 10)

    #
    # Verifica que una leyenda al costado de la figura también se reconoce.
    #
    def test_caption_beside_the_figure(self) -> None:
        caption = located(270, 180, 420, 220).prov[0]

        self.assertEqual(_caption_gap(self.doc, caption, self.figure), 20)

    #
    # Verifica que una leyenda lejana o en diagonal no se atribuye a la figura.
    #
    def test_distant_or_diagonal_caption_is_rejected(self) -> None:
        distant = located(50, 500, 250, 520).prov[0]
        diagonal = located(270, 310, 420, 330).prov[0]

        self.assertIsNone(_caption_gap(self.doc, distant, self.figure))
        self.assertIsNone(_caption_gap(self.doc, diagonal, self.figure))


class CaptionLabelTests(unittest.TestCase):
    #
    # Verifica que los anexos y las numeraciones romanas se reconocen como nombres.
    #
    def test_annexes_and_roman_numbers_are_named_elements(self) -> None:
        self.assertEqual(caption_label("Anexo 3. Lista de especies"), "annex 3")
        self.assertEqual(caption_label("Annex 1 Sampling sites"), "annex 1")
        self.assertEqual(caption_label("Plate III. Mokoan data"), "plate iii")

    #
    # Verifica que una frase que menciona una tabla no se toma por su leyenda.
    #
    def test_sentence_is_not_a_loose_caption(self) -> None:
        self.assertIsNone(loose_caption_label("Table 2 shows the results"))
        self.assertIsNone(loose_caption_label("Fig. 3 shows three such spectra"))
        self.assertEqual(loose_caption_label("Table 2. Results"), "table 2")
        self.assertEqual(loose_caption_label("Annex 1 Sampling sites"), "annex 1")


class DocumentRepairTests(unittest.TestCase):
    #
    # Verifica que un rango partido entre dos textos seguidos se vuelve a unir.
    #
    def test_split_range_is_joined(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "Its depth varies from 2-", prov=provenance(100))
        doc.add_text(DocItemLabel.TEXT, "3 m to 6-16 m", prov=provenance(130))

        join_split_ranges(doc)

        self.assertEqual([item.text for item in doc.texts], ["Its depth varies from 2-3 m to 6-16 m"])
        self.assertEqual(len(doc.texts[0].prov), 2)

    #
    # Verifica que no se une un rango si hay otro elemento entre los dos textos.
    #
    def test_range_is_not_joined_across_another_element(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "Datos desde 1990-", prov=provenance(100))
        doc.add_heading("Resultados", prov=provenance(130))
        doc.add_text(DocItemLabel.TEXT, "12 muestras", prov=provenance(160))

        join_split_ranges(doc)

        self.assertEqual(len(doc.texts), 3)

    #
    # Verifica que el nombre de una tabla pegado a un párrafo de otra página se separa y se enlaza.
    #
    def test_caption_merged_into_a_paragraph_is_detached(self) -> None:
        doc = document()
        paragraph = doc.add_text(DocItemLabel.TEXT, "toxin registers as", prov=provenance(100))
        paragraph.text = "toxin registers as Table 2"
        paragraph.prov[0].charspan = (0, 18)
        paragraph.prov.append(
            ProvenanceItem(page_no=1, bbox=provenance(300).bbox, charspan=(19, 26))
        )
        table = doc.add_table(TableData(num_rows=1, num_cols=1), prov=provenance(330))

        detach_merged_captions(doc)
        attach_loose_captions(doc)

        self.assertEqual(doc.texts[0].text, "toxin registers as")
        self.assertEqual(len(doc.texts[0].prov), 1)
        self.assertEqual(table.caption_text(doc), "Table 2")

    #
    # Verifica que una leyenda en una pieza intermedia se separa y la frase vuelve a quedar entera.
    #
    def test_caption_in_a_middle_piece_is_detached(self) -> None:
        doc = document()
        paragraph = doc.add_text(DocItemLabel.TEXT, "through both", prov=provenance(100))
        paragraph.text = "through both Fig. 5. Exposure routes waters"
        paragraph.prov[0].charspan = (0, 12)
        paragraph.prov += [
            ProvenanceItem(page_no=1, bbox=provenance(300).bbox, charspan=(13, 36)),
            ProvenanceItem(page_no=1, bbox=provenance(600).bbox, charspan=(37, 43)),
        ]
        figure = doc.add_picture(prov=provenance(330))

        detach_merged_captions(doc)
        attach_loose_captions(doc)

        self.assertEqual(doc.texts[0].text, "through both waters")
        self.assertEqual(figure.caption_text(doc), "Fig. 5. Exposure routes")

    #
    # Verifica que una leyenda larga se enlaza si ya se sabe que es una leyenda, pero un párrafo no.
    #
    def test_only_known_captions_may_be_long(self) -> None:
        doc = document()
        text = "Fig. 5. Exposure routes " + "and guidance values " * 50
        doc.add_text(DocItemLabel.CAPTION, text, prov=provenance(310))
        figure = doc.add_picture(prov=provenance(330))

        self.assertEqual(attach_loose_captions(doc), 1)
        self.assertTrue(figure.caption_text(doc).startswith("Fig. 5."))

        other = document()
        other.add_text(DocItemLabel.TEXT, text, prov=provenance(310))
        other.add_picture(prov=provenance(330))

        self.assertEqual(attach_loose_captions(other), 0)

    #
    # Verifica que sin un elemento al lado la pieza se toma como la oración que sigue.
    #
    def test_named_piece_without_a_neighbour_is_kept(self) -> None:
        doc = document()
        paragraph = doc.add_text(DocItemLabel.TEXT, "as shown in", prov=provenance(100))
        paragraph.text = "as shown in Table 2. Overall the bloom ended"
        paragraph.prov.append(
            ProvenanceItem(page_no=1, bbox=provenance(300).bbox, charspan=(12, 44))
        )

        detach_merged_captions(doc)

        self.assertEqual(len(doc.texts), 1)

    #
    # Verifica que un párrafo que sigue en otra página con texto común no se toca.
    #
    def test_paragraph_across_pages_is_kept(self) -> None:
        doc = document()
        paragraph = doc.add_text(DocItemLabel.TEXT, "the bloom lasted", prov=provenance(100))
        paragraph.text = "the bloom lasted two weeks"
        paragraph.prov.append(
            ProvenanceItem(page_no=1, bbox=provenance(300).bbox, charspan=(17, 26))
        )

        detach_merged_captions(doc)

        self.assertEqual(doc.texts[0].text, "the bloom lasted two weeks")

    #
    # Verifica que un guion común al final de un texto no une nada.
    #
    def test_text_ending_in_a_word_is_not_joined(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "pre-", prov=provenance(100))
        doc.add_text(DocItemLabel.TEXT, "3 m", prov=provenance(130))

        join_split_ranges(doc)

        self.assertEqual(len(doc.texts), 2)

    #
    # Verifica que los guiones de corte invisibles no parten las palabras.
    #
    def test_soft_hyphens_are_removed(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "cyano\xadbacteria", prov=provenance())

        clean_text(doc)

        self.assertEqual(doc.texts[0].text, "cyanobacteria")

    #
    # Verifica que la bibliografía tomada por encabezado de página abre su sección.
    #
    def test_references_heading_in_the_page_header_opens_a_section(self) -> None:
        doc = document()
        heading = doc.add_text(DocItemLabel.PAGE_HEADER, "BIBLIOGRAFÍA", prov=provenance(50))
        heading.content_layer = ContentLayer.FURNITURE
        doc.add_text(DocItemLabel.TEXT, "ACHA, M. (2004). Coastal fronts.", prov=provenance(100))
        doc.add_text(DocItemLabel.TEXT, "ADAMS, S. (2005). Bioindicators.", prov=provenance(130))

        restore_headings(doc)

        self.assertEqual([item.title for item in build_outline(doc).sections], ["BIBLIOGRAFÍA"])

    #
    # Verifica que la leyenda de un mapa no saca de la búsqueda al texto que la sigue.
    #
    def test_legend_named_references_does_not_open_the_bibliography(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "References", prov=provenance(50))
        doc.add_text(DocItemLabel.TEXT, "El muestreo se hizo en 12 sitios.", prov=provenance(100))
        doc.add_text(DocItemLabel.TEXT, "El pH fue de 7,8 a 8,4.", prov=provenance(130))

        restore_headings(doc)

        self.assertEqual(build_outline(doc).sections, [])

    #
    # Verifica que un encabezado repetido antes del título real no se toma por el título.
    #
    def test_running_header_before_the_real_heading_stays_out(self) -> None:
        doc = document()
        header = doc.add_text(DocItemLabel.PAGE_HEADER, "References", prov=provenance(10))
        header.content_layer = ContentLayer.FURNITURE
        doc.add_text(DocItemLabel.TEXT, "Fin de las conclusiones.", prov=provenance(50))
        doc.add_heading("References", prov=provenance(80))

        restore_headings(doc)

        self.assertEqual(doc.texts[0].content_layer, ContentLayer.FURNITURE)

    #
    # Verifica que un encabezado repetido no abre otra sección si el título real ya existe.
    #
    def test_running_header_does_not_duplicate_the_references_section(self) -> None:
        doc = document()
        doc.add_heading("References", prov=provenance(50))
        header = doc.add_text(DocItemLabel.PAGE_HEADER, "References", prov=provenance(10))
        header.content_layer = ContentLayer.FURNITURE

        restore_headings(doc)

        self.assertEqual([item.title for item in build_outline(doc).sections], ["References"])

    #
    # Verifica que un título en mayúsculas tomado por texto vuelve a abrir su sección.
    #
    def test_capitalized_title_opens_a_section(self) -> None:
        doc = document()
        title = doc.add_text(DocItemLabel.TEXT, "2. METODOLOGÍAS Y ÁREA DE ESTUDIO", prov=provenance(50))
        title.prov[0].page_no = 3
        doc.add_text(DocItemLabel.TEXT, "El área de estudio comprende.", prov=provenance(100))

        restore_headings(doc)

        titles = [item.title for item in build_outline(doc).sections]
        self.assertEqual(titles, ["2. METODOLOGÍAS Y ÁREA DE ESTUDIO"])

    #
    # Verifica que las mayúsculas de la portada no son títulos de sección.
    #
    def test_cover_page_capitals_are_not_titles(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "DIRECTORA: DRA. NORA GÓMEZ", prov=provenance(50))

        restore_headings(doc)

        self.assertEqual(build_outline(doc).sections, [])

    #
    # Verifica que un DOI partido por un salto de línea se recupera entero.
    #
    def test_doi_split_by_a_line_break_is_rejoined(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "https://doi.org/10.3390/blsf 2022014038", prov=provenance())

        self.assertEqual(document_doi(doc), "10.3390/blsf2022014038")

    #
    # Verifica que un encabezado de página repetido, aunque cambie su número, no es un título.
    #
    def test_repeated_page_header_is_not_a_title(self) -> None:
        doc = document()
        for page in (13, 14):
            text = f"CHAPTER 1. INTRODUCTION {page}"
            header = doc.add_text(DocItemLabel.PAGE_HEADER, text, prov=provenance(10))
            header.content_layer = ContentLayer.FURNITURE

        restore_headings(doc)

        self.assertEqual(build_outline(doc).sections, [])

    #
    # Verifica que un acento escrito aparte vuelve a su letra.
    #
    def test_loose_accents_join_their_letter(self) -> None:
        doc = document()
        doc.add_text(DocItemLabel.TEXT, "El Ni˜no, L´opez y d´ıas", prov=provenance())

        clean_text(doc)

        self.assertEqual(doc.texts[0].text, "El Niño, López y días")

    #
    # Verifica que una fila con valores no se toma por encabezado, aunque sus rótulos tengan números.
    #
    def test_value_rows_are_told_from_headers(self) -> None:
        self.assertTrue(is_value_row(["Kirchneriella obtusa", "0 ± 0", "93 ± 162"]))
        self.assertTrue(is_value_row(["Sitio", ".85", "–0.5", "(12)"]))
        self.assertFalse(is_value_row(["OD [mg L-1]", "NID [mg L-1]", "P-PO4 3-"]))

    #
    # Verifica que una leyenda sin nombre se desarma y que una nombrada se enlaza a su figura.
    #
    def test_only_named_captions_stay_linked(self) -> None:
        doc = document()
        axis = doc.add_text(DocItemLabel.CAPTION, "Wavelength (nm)", prov=provenance(320))
        figure = doc.add_picture(prov=provenance(100), caption=axis)
        figure.prov[0].bbox = BoundingBox(
            l=50, t=100, r=300, b=300, coord_origin=CoordOrigin.TOPLEFT
        )
        doc.add_text(DocItemLabel.TEXT, "Fig. 7. Spectral data", prov=provenance(310))

        release_unnamed_captions(doc)
        linked = attach_loose_captions(doc)

        self.assertEqual(linked, 1)
        self.assertEqual(figure.caption_text(doc), "Fig. 7. Spectral data")


class ReleasedCaptionTests(unittest.TestCase):
    #
    # Verifica que la leyenda sin nombre de una tabla vuelve al cuerpo y no se pierde.
    #
    def test_table_caption_without_name_returns_to_the_body(self) -> None:
        doc = document()
        caption = doc.add_text(DocItemLabel.CAPTION, "Parámetros medidos", prov=provenance(90))
        table = doc.add_table(
            TableData(num_rows=1, num_cols=1), caption=caption, prov=provenance(120)
        )
        doc.body.children = [child for child in doc.body.children if child.cref != caption.self_ref]
        caption.parent = table.get_ref()
        table.children.append(caption.get_ref())

        release_unnamed_captions(doc)

        self.assertEqual(table.captions, [])
        self.assertIn(caption.self_ref, [item.self_ref for item, _ in doc.iterate_items()])
        self.assertEqual(caption.parent.cref, doc.body.self_ref)


class TextOrderTests(unittest.TestCase):
    #
    # Verifica que el orden de las líneas no cambia la comparación, pero una pérdida sí.
    #
    def test_same_text_ignores_order_but_not_loss(self) -> None:
        self.assertTrue(_same_text("a la DBO5 de 3 mg", "de 3 mg a la DBO5"))
        self.assertFalse(_same_text("a la DBO5 de 3 mg", "de 3 mg"))

    #
    # Verifica que un dígito de más o de menos descarta el rearmado, aunque sea poco texto.
    #
    def test_digits_must_match_exactly(self) -> None:
        long_text = "el valor medido en la estación central del estuario fue "

        self.assertFalse(_same_text(long_text + "42", long_text + "4"))


if __name__ == "__main__":
    unittest.main()
