import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pypdfium2
from docling_core.types.doc import DocItemLabel, DoclingDocument, TextItem

from aquai.infrastructure.processing.documents.pageText import (
    glyphs_in_box,
    is_rotated,
    line_segments,
    page_glyphs,
    visual_lines,
)

_NOT_ALPHANUMERIC = re.compile(r"[\W_]+")
_BROKEN_WORD = re.compile("[^\\W\\d_][-\\x02]$")
_BROKEN_RANGE = re.compile(r"\d[-–]$")
_MIN_SHARED_HEIGHT = 8
_MAX_LEFT_SHIFT = 15
_MIN_SHARED_CHARS = 0.98


#
# Rearma los párrafos con líneas entrelazadas: si dos recuadros de una columna se solapan, Docling
# reparte sus líneas entre ambos y un valor puede terminar en otra oración. Se los une en el orden
# de la página, sólo si contienen el mismo texto. Devuelve cuántos rearmó.
#
def restore_line_order(path: Path, doc: DoclingDocument) -> int:
    pairs = _overlapping_pairs(doc)
    if not pairs:
        return 0

    pdf = pypdfium2.PdfDocument(str(path))
    merged: dict[str, TextItem] = {}
    try:
        for first, second in pairs:
            if first.self_ref in merged:
                continue

            page = pdf[first.prov[0].page_no - 1]
            text = _page_text(page, _joint_box(doc, first, second))
            if _same_text(text, first.text + second.text):
                first.text = text
                first.prov = [*first.prov, *second.prov]
                merged[second.self_ref] = second
    finally:
        pdf.close()

    if merged:
        doc.delete_items(node_items=list(merged.values()))

    return len(merged)


#
# Devuelve los textos seguidos de una misma columna cuyos recuadros comparten líneas.
#
def _overlapping_pairs(doc: DoclingDocument) -> list[tuple[TextItem, TextItem]]:
    texts = [
        item
        for item, _ in doc.iterate_items()
        if isinstance(item, TextItem) and item.label == DocItemLabel.TEXT and len(item.prov) == 1
    ]
    return [pair for pair in zip(texts, texts[1:]) if _share_lines(doc, *pair)]


#
# Indica si dos textos de una misma página y columna se solapan en más de una línea parcial.
#
def _share_lines(doc: DoclingDocument, first: TextItem, second: TextItem) -> bool:
    if first.prov[0].page_no != second.prov[0].page_no:
        return False

    upper, lower = _boxes(doc, first, second)
    shared = min(upper.b, lower.b) - max(upper.t, lower.t)
    return shared >= _MIN_SHARED_HEIGHT and abs(upper.l - lower.l) <= _MAX_LEFT_SHIFT


#
# Devuelve los recuadros de dos textos de una página, con origen arriba a la izquierda.
#
def _boxes(doc: DoclingDocument, first: TextItem, second: TextItem) -> tuple[Any, Any]:
    height = doc.pages[first.prov[0].page_no].size.height
    return (
        first.prov[0].bbox.to_top_left_origin(height),
        second.prov[0].bbox.to_top_left_origin(height),
    )


#
# Devuelve el recuadro que abarca a dos textos.
#
def _joint_box(doc: DoclingDocument, first: TextItem, second: TextItem) -> list[float]:
    upper, lower = _boxes(doc, first, second)
    return [
        min(upper.l, lower.l),
        min(upper.t, lower.t),
        max(upper.r, lower.r),
        max(upper.b, lower.b),
    ]


#
# Lee el texto de una región en el orden en que la página muestra sus líneas. Devuelve vacío si
# no es una sola columna de texto horizontal: ahí las líneas visuales no son renglones.
#
def _page_text(page, box: list[float]) -> str:
    glyphs = glyphs_in_box(page_glyphs(page), box)
    lines = [line_segments(line) for line in visual_lines(glyphs)]
    if not glyphs or is_rotated(glyphs) or any(len(line) > 1 for line in lines):
        return ""

    text = ""
    for line in lines:
        words = line[0].text.replace("\x02", "-")
        if _BROKEN_WORD.search(text) and words[:1].islower():
            text = text[:-1] + words
        elif _BROKEN_RANGE.search(text) and words[:1].isdigit():
            text += words
        else:
            text = f"{text} {words}".strip()

    return text


#
# Indica si dos textos tienen los mismos caracteres, sin importar su orden; los dígitos deben
# coincidir todos, para que ningún valor se pierda ni se duplique.
#
def _same_text(first: str, second: str) -> bool:
    one, other = Counter(_compact(first)), Counter(_compact(second))
    same_digits = _digits(one) == _digits(other)
    shared = (one & other).total()
    return same_digits and shared >= _MIN_SHARED_CHARS * max(one.total(), other.total(), 1)


#
# Devuelve la cuenta de dígitos de un texto.
#
def _digits(letters: Counter) -> dict[str, int]:
    return {char: count for char, count in letters.items() if char.isdigit()}


#
# Reduce un texto a sus letras y dígitos, con las ligaduras separadas.
#
def _compact(text: str) -> str:
    return _NOT_ALPHANUMERIC.sub("", unicodedata.normalize("NFKC", text)).casefold()
