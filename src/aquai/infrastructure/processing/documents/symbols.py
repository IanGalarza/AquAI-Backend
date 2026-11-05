import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pypdfium2
from docling_core.types.doc import DoclingDocument

from aquai.application.Rules.documents import UNMAPPED_CHARACTER
from aquai.infrastructure.processing.documents.pageText import Glyph, page_glyphs, render_box

MICRO_SIGN = "µ"
MINUS_SIGN = "−"
SYMBOL_ENCODING_LETTERS = {
    "a": "α",
    "b": "β",
    "c": "χ",
    "d": "δ",
    "e": "ε",
    "f": "φ",
    "g": "γ",
    "h": "η",
    "i": "ι",
    "k": "κ",
    "l": "λ",
    "m": MICRO_SIGN,
    "n": "ν",
    "p": "π",
    "q": "θ",
    "r": "ρ",
    "s": "σ",
    "t": "τ",
    "u": "υ",
    "w": "ω",
    "x": "ξ",
    "y": "ψ",
    "z": "ζ",
    "D": "Δ",
    "F": "Φ",
    "G": "Γ",
    "L": "Λ",
    "P": "Π",
    "Q": "Θ",
    "S": "Σ",
    "W": "Ω",
    "Y": "Ψ",
    "\xa2": "′",
    "\xa3": "≤",
    "\xb0": "°",
    "\xb1": "±",
    "\xb3": "≥",
    "\xb4": "×",
    "\xb7": "•",
    "~": "∼",
}
AMS_SYMBOLS = {"6": "≤", ">": "≥"}
READABLE_SYMBOLS = frozenset(MICRO_SIGN + "−–-≤≥×±°‰αβγδεηθλπρστφχψωΔΣΩ²³⁻¹")

_SYMBOL_FONT = re.compile(r"(?:^|\+)symbol", re.IGNORECASE)
_AMS_FONT = re.compile(r"msam10", re.IGNORECASE)
_STANDARD_ENCODINGS = ((_SYMBOL_FONT, SYMBOL_ENCODING_LETTERS), (_AMS_FONT, AMS_SYMBOLS))
_LOOKALIKES = "¥£"
_LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"
_NO_TEXT = "\ufffd"
_MICRO_READ_AS_LETTER = re.compile(
    r"(?:(?<=\d)|(?<=\d\s)|(?<=\()|(?<=\(\s))([pPl])\s?(?:g|m|mol|M|S)L?(?![^\W\d_])"
)
_MU_VARIANTS = "μµ"
_DASHES = "-–"
_ANY_DASH = "[-–—−]"
_PRIVATE_SYMBOLS = range(0xF000, 0xF100)
_BULLET = "•"
_SHORT_WORD = 2
_READING_CONTEXT = 2
_RARE_FONT_SHARE = 0.005
_CROP_SCALE = 5
_CROP_MARGIN_X = 40
_CROP_MARGIN_Y = 4
_SIDEWAYS_TURNS = (90, 270)
_CONTEXT_CHARS = 6
_SAMPLES_PER_GLYPH = 5
_MIN_AGREEMENT = 0.75
_MAX_RECORDED = 200
_BOX_TOLERANCE = 2.0
_MAX_READABLE_SUSPECTS = 8
_MIN_DECLARED_READINGS = 2

GlyphKey = tuple[str, str]
GlyphPlace = tuple[int, float, float]
ReadImage = Callable[[Any], str]


@dataclass
class SymbolWord:
    page: int
    text: str
    keys: dict[int, GlyphKey]
    left: float
    top: float
    right: float
    bottom: float
    before: str = ""
    after: str = ""
    place: GlyphPlace | None = None
    turn: int = 0

    #
    # Devuelve el centro de la palabra, con origen arriba a la izquierda.
    #
    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2


@dataclass(frozen=True)
class ResolvedGlyph:
    symbol: str
    method: str


@dataclass
class PageSymbols:
    by_glyph: dict[GlyphKey, str] = field(default_factory=dict)
    by_place: dict[GlyphPlace, str] = field(default_factory=dict)

    #
    # Aplica a los caracteres de una página los símbolos ya resueltos en el documento.
    #
    def applied(self, number: int, glyphs: list[Glyph]) -> list[Glyph]:
        return [replace(glyph, text=self._text(number, glyph)) for glyph in glyphs]

    #
    # Devuelve lo que dibuja un carácter, por su lugar en la página o por su fuente y código.
    #
    def _text(self, number: int, glyph: Glyph) -> str:
        placed = self.by_place.get(_place(number, glyph))
        return placed or self.by_glyph.get((glyph.font, glyph.text), glyph.text)


@dataclass
class SymbolReport:
    decoded: int = 0
    verified: int = 0
    readings: int = 0
    unresolved: int = 0
    unresolved_pages: list[int] = field(default_factory=list)
    unresolved_words: list[str] = field(default_factory=list)
    corrections: list[dict[str, Any]] = field(default_factory=list)
    glyphs: list[dict[str, Any]] = field(default_factory=list)

    #
    # Registra una corrección con su texto de origen, para poder auditarla.
    #
    def record(self, word: SymbolWord, corrected: str, methods: set[str]) -> None:
        if methods == {"symbol_encoding"}:
            self.decoded += 1
        else:
            self.verified += 1

        if len(self.corrections) < _MAX_RECORDED:
            self.corrections.append(
                {
                    "page": word.page,
                    "original": word.text,
                    "corrected": corrected,
                    "method": "+".join(sorted(methods)),
                }
            )

    #
    # Registra una palabra que no pudo resolverse: se conserva tal como está en el PDF.
    #
    def keep_doubt(self, word: SymbolWord) -> None:
        self.unresolved += 1
        if word.page not in self.unresolved_pages:
            self.unresolved_pages.append(word.page)
        if len(self.unresolved_words) < _MAX_RECORDED:
            self.unresolved_words.append(word.text)


#
# Devuelve el lugar de un carácter en el documento.
#
def _place(number: int, glyph: Glyph) -> GlyphPlace:
    return number, glyph.left, glyph.top


#
# Indica si un carácter debe resolverse sí o sí porque el PDF no dice qué dibuja.
#
def _requires_resolution(key: GlyphKey) -> bool:
    font, char = key
    known_font = any(pattern.search(font) for pattern, _ in _STANDARD_ENCODINGS)
    return bool(UNMAPPED_CHARACTER.fullmatch(char) or known_font)


#
# Indica si un carácter debe contrastarse: no tiene equivalente, o es una letra en una fuente
# rara dentro de una palabra de otra fuente, el rastro de un símbolo matemático.
#
def _is_suspect(glyph: Glyph, part: list[Glyph], rare_fonts: set[str]) -> bool:
    if UNMAPPED_CHARACTER.fullmatch(glyph.text):
        return True
    if _AMS_FONT.search(glyph.font):
        return glyph.text in AMS_SYMBOLS
    if _SYMBOL_FONT.search(glyph.font):
        known = glyph.text in _LOOKALIKES or glyph.text in SYMBOL_ENCODING_LETTERS
        return (glyph.text.isascii() and glyph.text.isalpha()) or known

    is_latin_letter = glyph.text.isascii() and glyph.text.isalpha()
    mixed_word = any(other.font != glyph.font for other in part)
    return is_latin_letter and glyph.font in rare_fonts and mixed_word


#
# Devuelve las fuentes con letras que aparecen muy poco frente al texto del documento.
#
def _rare_fonts(glyphs: list[Glyph]) -> set[str]:
    letters = Counter(glyph.font for glyph in glyphs if glyph.text.isalpha())
    total = sum(letters.values())
    return {font for font, count in letters.items() if font and count <= total * _RARE_FONT_SHARE}


#
# Agrupa los caracteres sospechosos de una página en las palabras que los contienen.
#
def _suspect_words(number: int, glyphs: list[Glyph], rare_fonts: set[str]) -> list[SymbolWord]:
    words = []
    start = 0
    for index in range(len(glyphs) + 1):
        if index < len(glyphs) and not _is_blank(glyphs[index].text):
            continue

        part = glyphs[start:index]
        keys = {
            position: (glyph.font, glyph.text)
            for position, glyph in enumerate(part)
            if _is_suspect(glyph, part, rare_fonts)
        }
        if keys:
            words.append(_word(number, glyphs, start, index, keys))
        start = index + 1

    return words


#
# Indica si un carácter separa palabras; un control sin equivalente no es un espacio.
#
def _is_blank(char: str) -> bool:
    return not char.strip() and not UNMAPPED_CHARACTER.fullmatch(char)


#
# Devuelve las unidades de una página cuya letra inicial pudo ser una µ mal reconocida.
#
def _unit_words(number: int, glyphs: list[Glyph]) -> list[SymbolWord]:
    text = "".join(glyph.text for glyph in glyphs)
    words = []
    for match in _MICRO_READ_AS_LETTER.finditer(text):
        mark = glyphs[match.start(1)]
        word = _word(number, glyphs, match.start(1), match.end(), {0: (mark.font, mark.text)})
        word.place = _place(number, mark)
        words.append(word)

    return words


#
# Construye una palabra sospechosa con su ubicación y el texto que la rodea.
#
def _word(
    number: int, glyphs: list[Glyph], start: int, end: int, keys: dict[int, GlyphKey]
) -> SymbolWord:
    part = glyphs[start:end]
    return SymbolWord(
        page=number,
        text="".join(glyph.text for glyph in part),
        keys=keys,
        left=min(glyph.left for glyph in part),
        top=min(glyph.top for glyph in part),
        right=max(glyph.right for glyph in part),
        bottom=max(glyph.bottom for glyph in part),
        before=_visible_text(glyphs[max(0, start - 3 * _CONTEXT_CHARS) : start])[-_CONTEXT_CHARS:],
        after=_visible_text(glyphs[end : end + 3 * _CONTEXT_CHARS])[:_CONTEXT_CHARS],
        turn=part[0].turn,
    )


#
# Devuelve los caracteres visibles de un tramo, sin espacios.
#
def _visible_text(glyphs: list[Glyph]) -> str:
    return "".join(glyph.text for glyph in glyphs if glyph.text.strip())


#
# Renderiza la región de una palabra en la página original, con margen, para leerla.
#
def _crop(page, word: SymbolWord) -> Any:
    sideways = word.turn in _SIDEWAYS_TURNS
    margin_x = _CROP_MARGIN_Y if sideways else _CROP_MARGIN_X
    margin_y = _CROP_MARGIN_X if sideways else _CROP_MARGIN_Y
    box = [word.left - margin_x, word.top - margin_y, word.right + margin_x, word.bottom + margin_y]
    return render_box(page, box, _CROP_SCALE, word.turn)


#
# Busca la palabra en lo leído y devuelve el símbolo en una posición, sólo si la lectura es
# inequívoca: el resto de la palabra debe coincidir y el símbolo debe ser uno conocido.
#
def _read_symbol(word: SymbolWord, target: int, reading: str) -> str | None:
    if len(word.keys) > _MAX_READABLE_SUSPECTS:
        return None

    matches = list(re.finditer(_reading_pattern(word), reading))
    if len(matches) != 1:
        return None

    read = dict(zip(sorted(word.keys), matches[0].groups())).get(target)
    symbol = _canonical_symbol(read or "")
    return symbol if symbol == word.text[target] or symbol in READABLE_SYMBOLS else None


#
# Arma la expresión que ubica la palabra en lo leído; si casi no tiene letras propias, usa las
# vecinas, porque un símbolo aislado no se puede encontrar por sí solo.
#
def _reading_pattern(word: SymbolWord) -> str:
    letters = r"\s*".join(
        r"(\S{1,2}?)" if position in word.keys else re.escape(char)
        for position, char in enumerate(word.text)
        if not _is_blank(char)
    )
    if len(word.text) - len(word.keys) >= _SHORT_WORD:
        return rf"(?<![^\W\d_]){letters}(?![^\W\d_])"

    before = r"\s*".join(re.escape(char) for char in word.before[-_READING_CONTEXT:])
    after = r"\s*".join(re.escape(char) for char in word.after[:_READING_CONTEXT])
    return rf"{before}\s*{letters}\s*{after}"


#
# Unifica las variantes de un símbolo leído: las dos µ de Unicode y los guiones como signo menos.
#
def _canonical_symbol(read: str) -> str:
    if read and read in _MU_VARIANTS:
        return MICRO_SIGN

    return MINUS_SIGN if read and read in _DASHES else read


#
# Elige hasta una cantidad fija de palabras, repartidas entre todas las apariciones de un glifo.
#
def _samples(words: list[SymbolWord]) -> list[SymbolWord]:
    step = max(len(words) // _SAMPLES_PER_GLYPH, 1)
    return words[::step][:_SAMPLES_PER_GLYPH]


#
# Lee un glifo en varias apariciones: un código de una fuente siempre dibuja lo mismo, así que
# basta que las lecturas coincidan. Devuelve el conteo de lo leído en cada aparición.
#
def _read_glyph(
    pdf,
    key: GlyphKey,
    words: list[SymbolWord],
    read_image: ReadImage,
    report: SymbolReport,
    limit: int,
) -> Counter[str]:
    votes: Counter[str] = Counter()
    for word in _samples(words):
        if report.readings >= limit:
            break

        report.readings += 1
        target = next(position for position, value in word.keys.items() if value == key)
        symbol = _read_symbol(word, target, read_image(_crop(pdf[word.page - 1], word)))
        if symbol:
            votes[symbol] += 1

    return votes


#
# Decide qué dibuja un glifo. Si el PDF declara una letra, cambiarla exige lecturas unánimes,
# porque puede ser una letra real; si no declara nada, alcanza con un acuerdo amplio.
#
def _agreed_symbol(key: GlyphKey, votes: Counter[str], occurrences: int) -> str | None:
    if not votes:
        return None

    symbol, count = votes.most_common(1)[0]
    total = sum(votes.values())
    if UNMAPPED_CHARACTER.fullmatch(key[1]):
        enough = count >= min(2, occurrences) and count >= _MIN_AGREEMENT * total
    else:
        enough = count == total and count >= _MIN_DECLARED_READINGS

    return symbol if enough else None


#
# Decodifica un carácter de una fuente de símbolos por su estándar publicado; algunos PDF lo
# escriben en el rango de uso privado, donde U+F0xx equivale al código xx de la fuente.
#
def _standard_symbol(font: str, char: str) -> str | None:
    code = ord(char)
    plain = chr(code - 0xF000) if code in _PRIVATE_SYMBOLS else char
    for pattern, encoding in _STANDARD_ENCODINGS:
        if pattern.search(font):
            return encoding.get(plain)

    return None


#
# Resuelve cada glifo: por la codificación estándar o contrastándolo con la página. Devuelve
# además los glifos en disputa: la página muestra otra cosa, pero sin el acuerdo necesario.
#
def _resolve_glyphs(
    pdf, words: list[SymbolWord], read_image: ReadImage, report: SymbolReport, limit: int
) -> tuple[dict[GlyphKey, ResolvedGlyph], set[GlyphKey]]:
    by_key: dict[GlyphKey, list[SymbolWord]] = defaultdict(list)
    for word in words:
        for key in set(word.keys.values()):
            by_key[key].append(word)

    resolved = {}
    contested = set()
    for key, occurrences in by_key.items():
        font, char = key
        standard = _standard_symbol(font, char)
        if standard:
            resolved[key] = ResolvedGlyph(standard, "symbol_encoding")
            continue
        if char == _NO_TEXT:
            continue

        votes = _read_glyph(pdf, key, occurrences, read_image, report, limit)
        symbol = _agreed_symbol(key, votes, len(occurrences))
        if symbol:
            resolved[key] = ResolvedGlyph(symbol, "page_reading")
            report.glyphs.append(
                {"font": font, "code": ord(char), "symbol": symbol, "occurrences": len(occurrences)}
            )
        elif set(votes) - {char}:
            contested.add(key)

    return resolved, contested


#
# Devuelve las celdas y textos del documento que contienen físicamente a una palabra.
#
def _holders(doc: DoclingDocument, word: SymbolWord) -> Iterator[Any]:
    page = doc.pages.get(word.page)
    if page is None:
        return

    height = page.size.height
    for item in doc.texts:
        if _covers(item.prov, word, height):
            yield item
    for table in doc.tables:
        if _covers(table.prov, word, height):
            yield from _table_holders(table, word, height)


#
# Devuelve las celdas de una tabla que contienen a la palabra; sin ubicación no se toca ninguna,
# porque otra celda puede tener esa misma palabra y ser legítima.
#
def _table_holders(table, word: SymbolWord, height: float) -> list[Any]:
    cells = table.data.table_cells
    return [cell for cell in cells if cell.bbox and _inside(cell.bbox, word, height)]


#
# Indica si alguna procedencia del elemento, en la página de la palabra, la contiene.
#
def _covers(provenance: list, word: SymbolWord, height: float) -> bool:
    return any(
        prov.page_no == word.page and _inside(prov.bbox, word, height) for prov in provenance
    )


#
# Indica si un recuadro de Docling contiene el centro de la palabra.
#
def _inside(box, word: SymbolWord, height: float) -> bool:
    area = box.to_top_left_origin(height)
    x, y = word.center
    return (
        area.l - _BOX_TOLERANCE <= x <= area.r + _BOX_TOLERANCE
        and area.t - _BOX_TOLERANCE <= y <= area.b + _BOX_TOLERANCE
    )


#
# Arma la expresión que ubica la palabra dentro del texto extraído, con parte de su entorno.
#
def _item_pattern(word: SymbolWord, context: int) -> str:
    letters = [
        _unmapped_or_literal(char, position in word.keys)
        for position, char in enumerate(word.text)
        if not _is_blank(char)
    ]
    body = r"(?<![^\W\d_])(" + r"\s*".join(letters) + r")(?![^\W\d_])"
    if not context:
        return body

    before = r"\s*".join(_literal(char) for char in word.before[-context:])
    after = r"\s*".join(_literal(char) for char in word.after[:context])
    return rf"{before}\s*{body}\s*{after}"


#
# Devuelve la expresión de un carácter de la palabra; uno sin equivalente puede quedar en el
# texto como cualquier otro sin equivalente, pero nunca como un carácter legítimo.
#
def _unmapped_or_literal(char: str, is_suspect: bool) -> str:
    if is_suspect and UNMAPPED_CHARACTER.fullmatch(char):
        return UNMAPPED_CHARACTER.pattern

    return _literal(char)


#
# Devuelve la expresión de un carácter tal como Docling lo normaliza: guiones y ligaduras.
#
def _literal(char: str) -> str:
    if char in _DASHES or char == "—":
        return _ANY_DASH
    if char in _MU_VARIANTS:
        return f"[{_MU_VARIANTS}]"

    return re.escape(unicodedata.normalize("NFKC", char) if char in _LIGATURES else char)


#
# Reemplaza la palabra en un texto sólo si la ubica sin ambigüedad. El entorno se acorta porque
# puede tener otra palabra ya corregida.
#
def _replaced(text: str, word: SymbolWord, corrected: str) -> str | None:
    for context in (_CONTEXT_CHARS, _READING_CONTEXT, 0):
        matches = list(re.finditer(_item_pattern(word, context), text))
        if len(matches) == 1:
            start, end = matches[0].span(1)
            return text[:start] + corrected + text[end:]

    if _only_unmapped(word):
        return re.sub(_item_pattern(word, 0), lambda _: corrected, text)

    return None


#
# Indica si todo lo dudoso de la palabra son caracteres que no pueden ser texto legítimo: en ese
# caso es seguro corregir todas sus apariciones en un mismo elemento.
#
def _only_unmapped(word: SymbolWord) -> bool:
    return all(
        UNMAPPED_CHARACTER.fullmatch(char) or char in _LOOKALIKES
        for _, char in word.keys.values()
    )


#
# Aplica la corrección en el elemento del documento que contiene la palabra.
#
def _apply(doc: DoclingDocument, word: SymbolWord, corrected: str) -> bool:
    done = r"\s*".join(_literal(char) for char in corrected if not _is_blank(char))
    original = _item_pattern(word, 0)
    holders = list(_holders(doc, word))
    for holder in holders:
        text = _replaced(holder.text, word, corrected)
        if text is not None and text != holder.text:
            holder.text = text
            return True

    still_there = any(re.search(original, holder.text) for holder in holders)
    return not still_there and any(re.search(done, holder.text) for holder in holders)


#
# Devuelve la palabra con sus glifos resueltos y los métodos usados.
#
def _corrected(word: SymbolWord, resolved: dict[GlyphKey, ResolvedGlyph]) -> tuple[str, set[str]]:
    chars = list(word.text)
    methods = set()
    for position, key in word.keys.items():
        glyph = resolved.get(key)
        if glyph and glyph.symbol != chars[position]:
            chars[position] = glyph.symbol
            methods.add(glyph.method)

    return "".join(chars), methods


#
# Indica si a la palabra le queda algún carácter que el PDF no explica.
#
def _has_doubt(
    word: SymbolWord, resolved: dict[GlyphKey, ResolvedGlyph], contested: set[GlyphKey]
) -> bool:
    return any(
        key not in resolved and (_requires_resolution(key) or key in contested)
        for key in word.keys.values()
    )


#
# Corrige los símbolos mal declarados en el PDF: decodifica las fuentes de símbolos por su
# estándar y contrasta el resto con la página original. En las páginas escaneadas contrasta
# además las unidades, porque su capa de texto la escribió un OCR. Lo que no se pueda resolver
# queda como está.
#
def restore_symbols(
    path: Path,
    doc: DoclingDocument,
    read_image: ReadImage,
    max_readings: int,
    scanned_pages: set[int],
) -> tuple[SymbolReport, PageSymbols]:
    report = SymbolReport()
    symbols = PageSymbols()
    pdf = pypdfium2.PdfDocument(str(path))
    try:
        pages = [(number, page_glyphs(page)) for number, page in enumerate(pdf, 1)]
        rare_fonts = _rare_fonts([glyph for _, glyphs in pages for glyph in glyphs])
        words = []
        units = []
        for number, glyphs in pages:
            words += _suspect_words(number, glyphs, rare_fonts)
            units += _unit_words(number, glyphs) if number in scanned_pages else []

        resolved, contested = _resolve_glyphs(pdf, words, read_image, report, max_readings)
        _contrast_units(pdf, doc, units, read_image, report, max_readings, symbols)
    finally:
        pdf.close()

    for word in words:
        corrected, methods = _corrected(word, resolved)
        if _has_doubt(word, resolved, contested):
            report.keep_doubt(word)
        elif corrected.strip(_BULLET) and methods:
            _settle(doc, word, corrected, methods, report)

    symbols.by_glyph = {key: glyph.symbol for key, glyph in resolved.items()}
    return report, symbols


#
# Lee en la página cada unidad dudosa de un texto escaneado: corrige si la imagen muestra µ,
# la deja si muestra la letra y conserva la duda si no se la puede leer.
#
def _contrast_units(
    pdf,
    doc: DoclingDocument,
    words: list[SymbolWord],
    read_image: ReadImage,
    report: SymbolReport,
    limit: int,
    symbols: PageSymbols,
) -> None:
    for word in words:
        marks: set[str] = set()
        if report.readings < limit:
            report.readings += 1
            marks = _unit_marks(word, read_image(_crop(pdf[word.page - 1], word)))

        if marks and marks <= set(_MU_VARIANTS):
            symbols.by_place[word.place] = MICRO_SIGN
            _settle(doc, word, MICRO_SIGN + word.text[1:], {"page_reading"}, report)
        elif word.text[0] not in marks:
            report.keep_doubt(word)


#
# Devuelve lo que la lectura muestra delante de la unidad, justo después del número o paréntesis
# que la precede, para no tomar otra unidad del mismo recorte.
#
def _unit_marks(word: SymbolWord, reading: str) -> set[str]:
    unit = r"\s*".join(re.escape(char) for char in word.text[1:] if char.strip())
    return set(re.findall(rf"{re.escape(word.before[-1:])}\s*(\S)\s*{unit}(?![^\W\d_])", reading))


#
# Aplica una corrección y la registra, o conserva la duda si no se la puede ubicar.
#
def _settle(
    doc: DoclingDocument,
    word: SymbolWord,
    corrected: str,
    methods: set[str],
    report: SymbolReport,
) -> None:
    if _apply(doc, word, corrected):
        report.record(word, corrected, methods)
    else:
        report.keep_doubt(word)
