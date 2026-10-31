import re
import unicodedata

from aquai.application.Rules.vocabulary import (
    INTEGER,
    NUMBER_WORDS,
    SPREADSHEET_EXTENSION,
    STATISTIC_WORDS,
    SURROUNDING_PUNCTUATION,
)

_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_SEPARATORS = re.compile(r"[\W_]+")
_THOUSANDS_SEPARATORS = re.compile(r"[.,]")


def _word_counts() -> dict[str, int]:
    counts = {}

    for words in NUMBER_WORDS.values():
        for value, word in enumerate(words):
            counts[word] = value

    return counts


_WORD_COUNTS = _word_counts()


def normalize_text(value: str) -> str:
    """'5Punta piedras' -> '5 punta piedras'; 'Tabla_RdlP.xlsx' -> 'tabla rdlp xlsx'."""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _LETTER_DIGIT_BOUNDARY.sub(" ", text)

    return " ".join(_SEPARATORS.sub(" ", text).split())


def normalize_name(value: str) -> str:
    return normalize_text(SPREADSHEET_EXTENSION.sub("", str(value).strip()))


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

    while words and words[0] in STATISTIC_WORDS:
        words.pop(0)
    while words and words[-1] in STATISTIC_WORDS:
        words.pop()

    return " ".join(words)


def mentioned_count(phrase: str) -> int | None:
    """Cantidad entera escrita con cifras o palabras: «los primeros 5» → 5, «top cinco» → 5, «1.000 registros» → 1000.

    Un decimal («2,5») no es una cantidad de filas; sin una única cantidad, None."""
    counts = set()

    for token in str(phrase).split():
        token = token.strip(SURROUNDING_PUNCTUATION)
        if INTEGER.fullmatch(token):
            counts.add(int(_THOUSANDS_SEPARATORS.sub("", token)))
        elif (word := normalize_text(token)) in _WORD_COUNTS:
            counts.add(_WORD_COUNTS[word])

    counts.discard(0)

    return counts.pop() if len(counts) == 1 else None


def contains_phrase(text: str, phrase: str, *, plural_insensitive: bool = False) -> bool:
    """La frase aparece como secuencia completa de palabras, nunca como fragmento."""
    words = normalize_text(text).split()
    target = normalize_text(phrase).split()

    if plural_insensitive:
        words = [_singular(item) for item in words]
        target = [_singular(item) for item in target]

    if not target:
        return False

    return any(
        words[index : index + len(target)] == target
        for index in range(len(words) - len(target) + 1)
    )
