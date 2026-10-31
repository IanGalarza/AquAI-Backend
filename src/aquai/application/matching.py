"""Comparación literal tolerante a mayúsculas, acentos y puntuación; no infiere sinónimos."""

import re
import unicodedata


_EXTENSIONS = re.compile(r"\.(?:csv|xlsx|xlsm|xls)$", re.IGNORECASE)
# Estadística pedida y conectores que la unen a la variable: «pH promedio» y «valor máximo de pH» nombran pH.
_QUALIFIER_WORDS = {
    "promedio", "promedios", "medio", "media", "medias", "mediana", "maximo", "maximos", "maxima", "maximas",
    "minimo", "minimos", "minima", "minimas", "suma", "cantidad", "conteo", "desvio", "estandar", "valor", "valores",
    "de", "del", "la", "el", "los", "las", "mean", "average", "median", "maximum", "minimum", "max", "min", "sum",
    "count", "value", "values", "of", "the",
}
# Cantidades que se escriben con palabras; las respuestas las usan para redactar y el Router, para leerlas.
NUMBER_WORDS = {"es": ("cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez"),
                "en": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")}
_WORD_COUNTS = {word: value for words in NUMBER_WORDS.values() for value, word in enumerate(words)}
# Entero con separadores de miles opcionales: 5, 1000, 1.000, 1,000.
_INTEGER = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+")


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


def variable_core(phrase: str) -> str:
    """Variable sin la estadística que la acompaña: «pH promedio» → «ph», «valor máximo de turbidez» → «turbidez»."""
    words = normalize_text(phrase).split()
    while words and words[0] in _QUALIFIER_WORDS:
        words.pop(0)
    while words and words[-1] in _QUALIFIER_WORDS:
        words.pop()
    return " ".join(words)


def mentioned_count(phrase: str) -> int | None:
    """Cantidad entera escrita con cifras o palabras: «los primeros 5» → 5, «top cinco» → 5, «1.000 registros» → 1000.

    Un decimal («2,5») no es una cantidad de filas; sin una única cantidad, None."""
    counts = set()
    for token in str(phrase).split():
        token = token.strip("¿?¡!.,;:()[]«»\"'")
        if _INTEGER.fullmatch(token):
            counts.add(int(re.sub(r"[.,]", "", token)))
        elif (word := normalize_text(token)) in _WORD_COUNTS:
            counts.add(_WORD_COUNTS[word])
    counts.discard(0)
    return counts.pop() if len(counts) == 1 else None


def contains_phrase(text: str, phrase: str, *, plural_insensitive: bool = False) -> bool:
    """La frase aparece como secuencia completa de palabras, nunca como fragmento."""
    words, target = normalize_text(text).split(), normalize_text(phrase).split()
    if plural_insensitive:
        words, target = [_singular(item) for item in words], [_singular(item) for item in target]
    if not target:
        return False
    return any(words[index:index + len(target)] == target for index in range(len(words) - len(target) + 1))
