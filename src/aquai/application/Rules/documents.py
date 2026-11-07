import re
import unicodedata
from pathlib import PurePath

TABULAR_FILE_TYPES = ("csv", "xlsx")
DOCUMENT_FILE_TYPES = ("pdf", "docx")
DOCUMENT_MIME_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

REFERENCES_HEADING = re.compile(
    r"^\W*(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(?:references?(?:\s+cited)?|literature\s+cited|cited\s+literature|bibliography|"
    r"bibliograf[ií]a(?:\s+citada)?|referencias(?:\s+bibliogr[aá]ficas)?|"
    r"literatura\s+citada)[\s.]*$",
    re.IGNORECASE,
)
ELEMENT_NAME = re.compile(
    r"\b(fig(?:ure|ura)?s?|tab(?:les?|las?)?|cuadros?|plates?|l[aá]minas?|gr[aá]ficos?|mapas?|"
    r"fotos?|anexos?|annex(?:es)?)\.?\s*(?:n[°ºo]\.?\s*)?"
    r"((?:[A-Z]\.?)?\d+(?:\.\d+)*[a-z]?|(?-i:[IVX]+))\b",
    re.IGNORECASE,
)
_LETTER_PREFIX_DOT = re.compile(r"(?<=^[A-Za-z])\.")
DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>,;]+(?:\s+\d{6,}\b)?", re.IGNORECASE)
SUSPECT_MICRO_UNIT = re.compile(
    r"(?<![\w.])l(?:g|S|mol|m)\b|(?:(?<=\d )|(?<=\())[pP] ?[gm]\b|,ug\b"
)
UNMAPPED_CHARACTER = re.compile("[\\x00\\x01\\x03-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f\\ufffd\\ue000-\\uf8ff]")
WORD = re.compile(r"[^\W\d_]+")
TABLE_DOUBTS = ("uncertain", "rebuilt", "read_by_ocr", "unverifiable")
WHITESPACE = re.compile(r"\s+")

_ELEMENT_FAMILIES = {
    "fig": "figure",
    "tab": "table",
    "cua": "table",
    "pla": "plate",
    "lam": "plate",
    "lám": "plate",
    "gra": "figure",
    "grá": "figure",
    "map": "figure",
    "fot": "figure",
    "ane": "annex",
    "ann": "annex",
}
_LANGUAGE_CUES = {
    "en": {"the", "of", "and", "in", "to", "was", "were", "with", "for", "that", "this", "from"},
    "es": {"el", "la", "los", "las", "de", "del", "y", "en", "que", "para", "con", "una"},
    "pt": {"o", "os", "as", "do", "da", "dos", "das", "em", "para", "com", "uma", "foi"},
}


#
# Devuelve la extensión en minúsculas de un nombre de archivo, sin el punto.
#
def file_suffix(filename: str | None) -> str:
    if not isinstance(filename, str):
        return ""

    return PurePath(filename.strip()).suffix.lower().removeprefix(".")


#
# Normaliza el nombre de una tabla o figura para enlazarla con sus menciones en el texto.
#
def element_key(family: str, number: str) -> str:
    kind = _ELEMENT_FAMILIES.get(family.lower()[:3], family.lower())
    return f"{kind} {_LETTER_PREFIX_DOT.sub('', number).lower()}"


#
# Devuelve el nombre normalizado con el que empieza una leyenda, si lo tiene.
#
def caption_label(caption: str) -> str | None:
    match = ELEMENT_NAME.match(caption.strip())
    return element_key(match.group(1), match.group(2)) if match else None


#
# Devuelve el nombre de un texto suelto que se lee como leyenda; «Table 2 shows» es una frase.
#
def loose_caption_label(text: str) -> str | None:
    match = ELEMENT_NAME.match(text.strip())
    following = text.strip()[match.end() :].lstrip() if match else ""
    return None if following[:1].islower() else caption_label(text)


#
# Devuelve las tablas y figuras mencionadas en un texto, sin repetir y en orden de aparición.
#
def element_mentions(text: str) -> list[str]:
    keys = (element_key(match.group(1), match.group(2)) for match in ELEMENT_NAME.finditer(text))
    return list(dict.fromkeys(keys))


#
# Indica si un título corresponde a la sección de bibliografía de un documento.
#
def is_references_heading(title: str) -> bool:
    return REFERENCES_HEADING.match(title.strip()) is not None


#
# Estima el idioma por las palabras funcionales más frecuentes, o None si no hay evidencia clara.
#
def detect_language(text: str) -> str | None:
    words = [word.lower() for word in WORD.findall(text[:40000])]
    if len(words) < 30:
        return None

    scores = language_scores(text[:40000])
    language, score = max(scores.items(), key=lambda item: item[1])
    return language if score >= 0.1 * len(words) else None


#
# Cuenta las palabras funcionales de cada idioma en un texto.
#
def language_scores(text: str) -> dict[str, int]:
    words = [word.lower() for word in WORD.findall(text)]

    return {
        language: sum(word in cues for word in words) for language, cues in _LANGUAGE_CUES.items()
    }


#
# Devuelve las advertencias de extracción que alcanzan a un fragmento.
#
def chunk_doubts(details, *, element_position, page_start, page_end, text) -> list[str]:
    quality = details.get("quality") or {}
    ocr = details.get("ocr") or {}
    doubts = _table_doubts(quality.get("tables") or {}, element_position)
    pages = set(range(page_start, (page_end or page_start) + 1)) if page_start else set()

    if pages & set(ocr.get("pages_with_prior_ocr") or []):
        doubts.append("prior_ocr_page")
    if pages & set(ocr.get("pages_without_text") or []):
        doubts.append("ocr_read_page")
    if pages & set((quality.get("symbols") or {}).get("unresolved_pages") or []):
        doubts.append("unresolved_symbols")
    if UNMAPPED_CHARACTER.search(text):
        doubts.append("unmapped_characters")
    if SUSPECT_MICRO_UNIT.search(text):
        doubts.append("suspect_micro_unit")

    return doubts


#
# Devuelve las marcas de revisión de la tabla de la que proviene un fragmento.
#
def _table_doubts(tables, element_position) -> list[str]:
    if element_position is None:
        return []

    return [
        f"table_{name}"
        for name in TABLE_DOUBTS
        if any(item.get("element") == element_position for item in tables.get(name) or [])
    ]


#
# Normaliza un texto para comparar el contenido de dos documentos sin importar el formato.
#
def normalized_text(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()
