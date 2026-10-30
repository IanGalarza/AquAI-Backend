"""Comparación literal tolerante a mayúsculas, acentos y puntuación; no infiere sinónimos."""

import re
import unicodedata


_EXTENSIONS = re.compile(r"\.(?:csv|xlsx|xlsm|xls)$", re.IGNORECASE)


def normalize_text(value: str) -> str:
    """'5Punta piedras' -> '5 punta piedras'; 'Tabla_RdlP.xlsx' -> 'tabla rdlp xlsx'."""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", text)
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def normalize_name(value: str) -> str:
    """Nombre de archivo o fuente sin extensión, para reconocer referencias explícitas."""
    return normalize_text(_EXTENSIONS.sub("", str(value).strip()))


def _singular(word: str) -> str:
    # Sólo iguala plurales regulares (totales/total, sitios/sitio); no traduce ni sinonimiza.
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def contains_phrase(text: str, phrase: str, *, plural_insensitive: bool = False) -> bool:
    """La frase aparece como secuencia completa de palabras, nunca como fragmento."""
    words, target = normalize_text(text).split(), normalize_text(phrase).split()
    if plural_insensitive:
        words, target = [_singular(item) for item in words], [_singular(item) for item in target]
    if not target:
        return False
    return any(words[index:index + len(target)] == target for index in range(len(words) - len(target) + 1))
