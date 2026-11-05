import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from docling_core.types.doc import (
    ContentLayer,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from aquai.application.contracts.documents import PreparedElement, PreparedSection
from aquai.application.Rules.documents import (
    DOI,
    WHITESPACE,
    caption_label,
    is_references_heading,
    loose_caption_label,
    normalized_text,
)
from aquai.domain.enums.documentContent import DocumentElementKind
from aquai.infrastructure.processing.documents.pageText import is_value_row

_TITLE_PAGES = 2
_TITLE_MIN_WORDS = 4
_DOI_PAGES = 2
_DECORATIVE_WIDTH = 25
_MAX_CAPTION_CHARS = 700
_MAX_CAPTION_GAP = 60
_MIN_CAPTION_OVERLAP = 0.3
_CAPTION_LABELS = (DocItemLabel.CAPTION, DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER)
_RANGE_BREAK = re.compile(r"\d[-–]$")
_SOFT_HYPHEN = "\xad"
_LOOSE_ACCENT = re.compile("([´`˜¨ˆ])([A-Za-zı])")
_COMBINING_ACCENTS = {"´": "\u0301", "`": "\u0300", "˜": "\u0303", "¨": "\u0308", "ˆ": "\u0302"}
_MIN_TITLE_LETTERS = 6
_CITATION_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CITATION_WINDOW = 3
_MIN_CITATIONS = 2
_MAX_TITLE_CHARS = 80
_OPENING_ITEMS = 40


@dataclass
class DocumentOutline:
    sections: list[PreparedSection] = field(default_factory=list)
    section_by_ref: dict[str, int] = field(default_factory=dict)
    order_by_ref: dict[str, int] = field(default_factory=dict)


#
# Recorre el documento en orden de lectura y arma sus secciones con su jerarquía y sus páginas.
#
def build_outline(doc: DoclingDocument) -> DocumentOutline:
    outline = DocumentOutline()
    stack: list[PreparedSection] = []
    captions = {ref.cref for item in [*doc.tables, *doc.pictures] for ref in item.captions}

    for order, (item, _) in enumerate(doc.iterate_items()):
        outline.order_by_ref[item.self_ref] = order
        if _opens_section(item, captions, stack):
            stack = _open_section(outline, stack, item)

        if stack:
            outline.section_by_ref[item.self_ref] = stack[-1].position
        _extend_pages(stack, item)

    return outline


#
# Indica si un elemento abre una sección: un título que no sea la leyenda de una tabla o figura.
# Un anexo nuevo sí la abre aunque nombre a su tabla, porque lo que sigue ya no es lo anterior.
#
def _opens_section(item: Any, captions: set[str], stack: list[PreparedSection]) -> bool:
    if not isinstance(item, (TitleItem, SectionHeaderItem)):
        return False

    label = caption_label(item.text) or ""
    is_new_annex = label.startswith("annex") and not (
        stack and caption_label(stack[-1].title) == label
    )
    return item.self_ref not in captions or is_new_annex


#
# Abre una sección nueva cerrando las de nivel igual o inferior y devuelve la pila vigente.
#
def _open_section(
    outline: DocumentOutline, stack: list[PreparedSection], item: TitleItem | SectionHeaderItem
) -> list[PreparedSection]:
    level = item.level if isinstance(item, SectionHeaderItem) else 0
    parents = [section for section in stack if section.level < level]
    section = PreparedSection(
        position=len(outline.sections),
        level=level,
        title=item.text.strip(),
        path=[*(parent.title for parent in parents), item.text.strip()],
    )
    outline.sections.append(section)
    return [*parents, section]


#
# Extiende el rango de páginas de las secciones abiertas con las páginas de un elemento.
#
def _extend_pages(stack: list[PreparedSection], item: Any) -> None:
    pages = [prov.page_no for prov in getattr(item, "prov", [])]
    if not pages:
        return

    for section in stack:
        section.page_start = min(pages[0], section.page_start or pages[0])
        section.page_end = max(max(pages), section.page_end or 0)


#
# Devuelve la procedencia de un elemento: página y recuadro, con origen arriba a la izquierda.
#
def item_locations(doc: DoclingDocument, item: DocItem) -> list[dict[str, Any]]:
    locations = []
    for prov in item.prov:
        page = doc.pages.get(prov.page_no)
        box = prov.bbox.to_top_left_origin(page.size.height) if page else prov.bbox
        locations.append(
            {
                "page": prov.page_no,
                "bbox": [round(value, 1) for value in (box.l, box.t, box.r, box.b)],
            }
        )

    return locations


#
# Devuelve la primera y la última página de una lista de ubicaciones, si las hay.
#
def page_span(locations: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    pages = [location["page"] for location in locations]
    return (min(pages), max(pages)) if pages else (None, None)


#
# Arma las tablas y figuras en orden de lectura, sin interpretar el contenido de los gráficos.
#
def build_elements(
    doc: DoclingDocument, outline: DocumentOutline
) -> tuple[list[PreparedElement], dict[str, PreparedElement]]:
    floating = [
        item
        for item in [*doc.tables, *doc.pictures]
        if item.self_ref in outline.order_by_ref and not is_decorative(item)
    ]
    floating.sort(key=lambda item: outline.order_by_ref[item.self_ref])

    by_ref = {}
    for position, item in enumerate(floating):
        by_ref[item.self_ref] = _element(doc, outline, item, position)

    return list(by_ref.values()), by_ref


#
# Arma una tabla o figura con su leyenda, su ubicación y, en las tablas, sus celdas.
#
def _element(
    doc: DoclingDocument,
    outline: DocumentOutline,
    item: TableItem | PictureItem,
    position: int,
) -> PreparedElement:
    caption = item.caption_text(doc).strip() or None
    locations = item_locations(doc, item)
    page_start, page_end = page_span(locations)
    is_table = isinstance(item, TableItem)
    cells = table_cells(item) if is_table else None
    notes = item_notes(doc, item)
    body = table_lines(cells) if is_table else figure_text(doc, item)

    return PreparedElement(
        position=position,
        kind=DocumentElementKind.TABLE if is_table else DocumentElementKind.FIGURE,
        section_position=outline.section_by_ref.get(item.self_ref),
        label=caption_label(caption) if caption else None,
        caption=caption,
        content="\n".join([*body, *notes]) or None,
        cells=cells,
        page_start=page_start,
        page_end=page_end,
        locations=locations,
        header_rows=header_row_count(item) if is_table else 0,
    )


#
# Devuelve las notas al pie de una tabla o figura.
#
def item_notes(doc: DoclingDocument, item: TableItem | PictureItem) -> list[str]:
    notes = [getattr(ref.resolve(doc), "text", "").strip() for ref in item.footnotes]
    return [note for note in notes if note]


#
# Devuelve el texto impreso dentro de una figura, tal como está: no se interpreta el gráfico.
#
def figure_text(doc: DoclingDocument, picture: PictureItem) -> list[str]:
    attached = {ref.cref for ref in [*picture.captions, *picture.footnotes]}
    return [
        child.text.strip()
        for child, _ in doc.iterate_items(root=picture, traverse_pictures=True)
        if isinstance(child, TextItem) and child.self_ref not in attached and child.text.strip()
    ]


#
# Indica si una figura es un adorno, como una barra del margen, y no contenido del documento.
#
def is_decorative(item: Any) -> bool:
    return (
        isinstance(item, PictureItem)
        and bool(item.prov)
        and item.prov[0].bbox.width < _DECORATIVE_WIDTH
    )


#
# Desarma las leyendas de los adornos y las que no se nombran como tal: suelen ser un rótulo del
# eje, y una leyenda ajena es peor que ninguna.
#
def release_unnamed_captions(doc: DoclingDocument) -> None:
    for item in [*doc.tables, *doc.pictures]:
        if is_decorative(item) or not caption_label(item.caption_text(doc)):
            _release_captions(doc, item)


#
# Enlaza con su tabla o figura las leyendas que quedaron sueltas; sólo si la leyenda se nombra
# como tal y toca a un elemento del mismo tipo sin leyenda.
#
def attach_loose_captions(doc: DoclingDocument) -> int:
    taken = {ref.cref for item in [*doc.tables, *doc.pictures] for ref in item.captions}
    linked = 0
    for item in doc.texts:
        label = loose_caption_label(item.text) if _may_be_caption(item, taken) else None
        target = _nearest_uncaptioned(doc, item.prov[0], label) if label else None
        if target is not None:
            target.captions.append(item.get_ref())
            linked += 1

    return linked


#
# Quita las leyendas de un elemento sin perder su texto: una figura lo conserva como texto
# impreso; de una tabla o un adorno vuelven al cuerpo, porque sus hijos no se fragmentan.
#
def _release_captions(doc: DoclingDocument, item: Any) -> None:
    released = list(item.captions)
    item.captions.clear()
    if isinstance(item, PictureItem) and not is_decorative(item):
        return

    owner = item.parent.resolve(doc)
    place = next(
        position
        for position, child in enumerate(owner.children)
        if child.cref == item.self_ref
    )
    for ref in released:
        caption = ref.resolve(doc)
        if caption.parent.cref == item.self_ref:
            item.children = [child for child in item.children if child.cref != ref.cref]
            caption.parent = item.parent
            owner.children.insert(place + 1, ref)


#
# Indica si un texto libre puede ser una leyenda: ubicado, sin elemento asignado y breve, salvo
# que ya se sepa que es una leyenda.
#
def _may_be_caption(item: Any, taken: set[str]) -> bool:
    return (
        item.label in _CAPTION_LABELS
        and item.self_ref not in taken
        and bool(item.prov)
        and (item.label == DocItemLabel.CAPTION or len(item.text) <= _MAX_CAPTION_CHARS)
    )


#
# Devuelve el elemento del tipo nombrado por la leyenda más cercano a ella, sin leyenda propia.
#
def _nearest_uncaptioned(doc: DoclingDocument, place: Any, label: str) -> Any | None:
    candidates = doc.pictures if label.startswith(("figure", "plate")) else doc.tables
    distances = [
        (_caption_gap(doc, place, item), position)
        for position, item in enumerate(candidates)
        if item.prov
        and item.prov[0].page_no == place.page_no
        and not item.captions
        and not is_decorative(item)
    ]
    near = [entry for entry in distances if entry[0] is not None]
    return candidates[min(near)[1]] if near else None


#
# Devuelve la distancia entre una leyenda y un elemento enfrentados, arriba, abajo o al costado,
# o None si no se corresponden.
#
def _caption_gap(doc: DoclingDocument, place: Any, item: Any) -> float | None:
    height = doc.pages[place.page_no].size.height
    first = place.bbox.to_top_left_origin(height)
    second = item.prov[0].bbox.to_top_left_origin(height)
    gaps = []
    if _face_each_other(first.l, first.r, second.l, second.r):
        gaps.append(max(second.t - first.b, first.t - second.b, 0))
    if _face_each_other(first.t, first.b, second.t, second.b):
        gaps.append(max(second.l - first.r, first.l - second.r, 0))

    return min((gap for gap in gaps if gap <= _MAX_CAPTION_GAP), default=None)


#
# Indica si dos tramos de un mismo eje se solapan lo suficiente para estar enfrentados.
#
def _face_each_other(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> bool:
    overlap = min(first_end, second_end) - max(first_start, second_start)
    narrower = min(first_end - first_start, second_end - second_start)
    return narrower > 0 and overlap / narrower >= _MIN_CAPTION_OVERLAP


#
# Devuelve la grilla de textos de una tabla, tal como quedó en el documento.
#
def table_cells(table: TableItem) -> list[list[str]]:
    return [[cell.text.strip() for cell in row] for row in table.data.grid]


#
# Devuelve las filas de una tabla como líneas de Markdown.
#
def table_lines(cells: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in row) + " |"
        for row in cells
    ]


#
# Devuelve la cantidad de filas de encabezado que Docling marcó al comienzo de una tabla; en una
# página de continuación puede marcar como encabezado la primera fila de datos.
#
def header_row_count(table: TableItem) -> int:
    count = 0
    for row in table.data.grid:
        is_header = any(cell.column_header for cell in row)
        if not is_header or is_value_row([cell.text for cell in row]):
            break
        count += 1

    return count


#
# Separa de un párrafo la leyenda que Docling le pegó porque la oración seguía más adelante: sin
# ella el elemento queda sin nombre y la frase, cortada. Sólo si la pieza nombra un elemento y
# toca a uno sin leyenda; si no, puede ser la oración siguiente. Va de atrás hacia adelante
# para que las posiciones de las demás piezas sigan valiendo.
#
def detach_merged_captions(doc: DoclingDocument) -> None:
    for item in [text for text in doc.texts if _is_plain_text(text) and len(text.prov) > 1]:
        for part in reversed(item.prov[1:]):
            start, end = part.charspan
            name = item.text[start:end]
            label = loose_caption_label(name)
            if label and _nearest_uncaptioned(doc, part, label) is not None:
                item.text = f"{item.text[:start].rstrip()} {item.text[end:].lstrip()}".strip()
                item.prov.remove(part)
                doc.insert_text(item, DocItemLabel.CAPTION, name, prov=part)


#
# Recupera los títulos que el análisis de página tomó por encabezado o por texto: el primer
# nombre de la bibliografía, sin el cual las citas se indexarían como texto, y los títulos en
# mayúsculas que no se repiten, porque un encabezado de página sí lo hace.
#
def restore_headings(doc: DoclingDocument) -> None:
    layers = set(ContentLayer)
    items = [item for item, _ in doc.iterate_items(included_content_layers=layers)]
    texts = [item for item in items if isinstance(item, TextItem)]
    repeated = Counter(_without_numbers(item.text) for item in texts)
    references = _references_heading(texts)
    for item in texts:
        once = repeated[_without_numbers(item.text)] == 1
        if item is references or (once and _reads_as_title(item)):
            _promote_to_heading(doc, item)


#
# Devuelve el texto que encabeza la bibliografía, si el documento no tiene ya ese título: el que
# la nombra y va seguido de citas, porque lo que queda bajo él sale de la búsqueda.
#
def _references_heading(texts: list[TextItem]) -> TextItem | None:
    named = [item for item in texts if is_references_heading(item.text)]
    if any(isinstance(item, (TitleItem, SectionHeaderItem)) for item in named):
        return None

    for item in named:
        following = texts[texts.index(item) + 1 :][:_CITATION_WINDOW]
        if sum(bool(_CITATION_YEAR.search(text.text)) for text in following) >= _MIN_CITATIONS:
            return item

    return None


#
# Quita los números de un texto, que en un encabezado de página cambian de una a otra.
#
def _without_numbers(text: str) -> str:
    return normalized_text(re.sub(r"\d+", "", text))


#
# Indica si un texto o encabezado de página se lee como un título: breve, en mayúsculas y fuera
# de la portada, donde las mayúsculas son autoridades o instituciones.
#
def _reads_as_title(item: TextItem) -> bool:
    text = item.text.strip()
    letters = sum(char.isalpha() for char in text)
    return (
        item.label in (DocItemLabel.TEXT, DocItemLabel.PAGE_HEADER)
        and _MIN_TITLE_LETTERS <= letters
        and len(text) <= _MAX_TITLE_CHARS
        and text == text.upper()
        and not caption_label(text)
        and not _on_opening_pages(item, _TITLE_PAGES)
    )


#
# Reemplaza un texto por un título en el mismo lugar del documento.
#
def _promote_to_heading(doc: DoclingDocument, item: TextItem) -> None:
    if isinstance(item, (TitleItem, SectionHeaderItem)):
        return

    position = int(item.self_ref.rsplit("/", 1)[-1])
    doc.texts[position] = SectionHeaderItem(
        self_ref=item.self_ref,
        parent=item.parent,
        label=DocItemLabel.SECTION_HEADER,
        prov=item.prov,
        orig=item.orig,
        text=item.text,
        level=1,
    )


#
# Limpia lo que parte las palabras para la búsqueda: guiones de corte invisibles y acentos que
# el PDF escribe como un carácter aparte, como Ni˜no o L´opez.
#
def clean_text(doc: DoclingDocument) -> None:
    cells = [cell for table in doc.tables for cell in table.data.table_cells]
    for item in [*doc.texts, *cells]:
        text = item.text.replace(_SOFT_HYPHEN, "")
        item.text = _LOOSE_ACCENT.sub(_accented, text)


#
# Devuelve la letra con su acento, como un solo carácter.
#
def _accented(match: re.Match) -> str:
    letter = match.group(2).replace("ı", "i")
    return unicodedata.normalize("NFC", letter + _COMBINING_ACCENTS[match.group(1)])


#
# Une dos textos seguidos que quedaron partidos en medio de un rango: 2- y 3 m son 2-3 m.
#
def join_split_ranges(doc: DoclingDocument) -> None:
    items = [item for item, _ in doc.iterate_items()]
    joined = []
    for first, second in zip(items, items[1:]):
        if _is_plain_text(first) and _is_plain_text(second) and _splits_range(first, second):
            second.text = first.text + second.text
            second.prov = [*first.prov, *second.prov]
            joined.append(first)

    if joined:
        doc.delete_items(node_items=joined)


#
# Indica si un elemento es un párrafo común.
#
def _is_plain_text(item: Any) -> bool:
    return isinstance(item, TextItem) and item.label == DocItemLabel.TEXT


#
# Indica si un párrafo termina en medio de un rango que sigue en el párrafo siguiente.
#
def _splits_range(first: TextItem, second: TextItem) -> bool:
    return bool(_RANGE_BREAK.search(first.text)) and second.text[:1].isdigit()


#
# Conserva el texto crudo de las fórmulas no decodificadas y devuelve cuántas son.
#
def restore_raw_formulas(doc: DoclingDocument) -> int:
    undecoded = [
        item
        for item in doc.texts
        if isinstance(item, FormulaItem) and not item.text.strip() and item.orig.strip()
    ]
    for item in undecoded:
        item.text = item.orig

    return len(undecoded)


#
# Devuelve el título y su origen, sólo si hay evidencia en el propio documento.
#
def document_title(doc: DoclingDocument, pdf_title: str | None) -> tuple[str | None, str | None]:
    for item, _ in doc.iterate_items():
        if isinstance(item, TitleItem) and item.text.strip():
            return item.text.strip(), "document_title"

    if pdf_title and normalized_text(pdf_title) in normalized_text(_opening_text(doc)):
        return pdf_title.strip(), "pdf_metadata"

    headings = [
        item.text.strip()
        for item, _ in doc.iterate_items()
        if isinstance(item, SectionHeaderItem)
        and _on_opening_pages(item, _TITLE_PAGES)
        and len(item.text.split()) >= _TITLE_MIN_WORDS
    ]
    if headings:
        return max(headings, key=len), "opening_heading"

    return None, None


#
# Devuelve el DOI que figura en las primeras páginas del documento, si lo hay.
#
def document_doi(doc: DoclingDocument) -> str | None:
    items = doc.iterate_items(included_content_layers=set(ContentLayer))
    texts = [item for item, _ in items if getattr(item, "text", "")]
    opening = [item for item in texts if _on_opening_pages(item, _DOI_PAGES)]
    match = DOI.search(" ".join(item.text for item in opening or texts[:_OPENING_ITEMS]))
    return WHITESPACE.sub("", match.group(0)).rstrip(".)]") if match else None


#
# Devuelve el texto de las primeras páginas, o el inicio del documento si no tiene páginas.
#
def _opening_text(doc: DoclingDocument) -> str:
    items = [item for item, _ in doc.iterate_items() if getattr(item, "text", "")]
    opening = [item for item in items if _on_opening_pages(item, _TITLE_PAGES)]
    return " ".join(item.text for item in opening or items[:_OPENING_ITEMS])


#
# Indica si un elemento aparece en las primeras páginas del documento.
#
def _on_opening_pages(item: Any, pages: int) -> bool:
    return any(prov.page_no <= pages for prov in getattr(item, "prov", []))
