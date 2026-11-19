import calendar
import re
import unicodedata
from datetime import date, timedelta
from typing import NamedTuple

from aquai.application.Rules.units import UNIT_RATIO_WORD
from aquai.application.Rules.variables import (
    CHEMICAL_FORMULAS,
    DERIVED_SUM,
    OTHER_MEDIUM_WORDS,
    PERCENT_SIGN,
    SECONDARY_QUANTITY_WORDS,
    SUM_WORDS,
    UNIT_CONFIRMED_SYMBOLS,
    VARIABLE_SYNONYMS,
    WATER_WORDS,
)
from aquai.application.Rules.vocabulary import (
    INNER_STATISTIC_WORDS,
    COMPARED_STATISTICS,
    COMPARISON_ARTICLES,
    COMPARISON_REFERENCE,
    COMPARISON_BOUNDARIES,
    COMPARISON_CUES,
    POSTFIX_COMPARISON_CUES,
    COMPARISON_FILLERS,
    FOLLOWING_STATISTIC_FILLERS,
    INTEGER,
    MENTION_CONNECTORS,
    MONTH_ABBREVIATIONS,
    MONTH_NAMES,
    NUMBER,
    OPEN_END_WORDS,
    NUMBER_WORDS,
    PERIOD_BOUND_CUES,
    RANGE_CONNECTORS,
    RANGE_CUES,
    REQUESTED_YEAR,
    SEASON_WORDS,
    SPREADSHEET_EXTENSION,
    STATISTIC_WORDS,
    SURROUNDING_PUNCTUATION,
)
from aquai.application.units import is_ph_name
from aquai.application.Rules.evidence import (
    COMPARISON_DIRECTIONS,
    DOCUMENT_AUTHOR_NAMES,
    INCLUSIVE_BOUND,
    DOCUMENT_YEAR,
    REFERENCE_LINE_CUES,
    REFERENCE_REQUEST_CLAUSE,
    SEARCH_STOPWORDS,
)

_SHORT_TERM = 4
_MIN_TERM_ROOT = 4
_TERM_FLEXION_CHARS = 2
_MIN_PREFIX = 4
_MIN_FUZZY_AUTHOR = 7
_CITED_YEAR_WINDOW = 24
_MAX_FILLER_LETTERS = 2
_LEADING_NUMBER = re.compile(r"(-?\d[\d.,]*)\s*[^\d\s]{0,12}")
_MIN_CONTRACTION = 3
_MAX_CONTRACTION = 4
_MIN_TYPO_WORD = 5
_LONG_WORD = 8
from aquai.application.units import comparable_unit, unit_key, unitless_name

_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_SEPARATORS = re.compile(r"[\W_]+")
_THOUSANDS_SEPARATORS = re.compile(r"[.,]")
_WORD = re.compile(r"\w+")
_MAX_FOLLOWING_WORDS = 5
_EXPLICIT_DAY = re.compile(
    r"\b(?:(?P<iso>\d{4}-\d{1,2}-\d{1,2})|(?P<dmy>\d{1,2}/\d{1,2}/\d{4}))\b"
)
_EXPLICIT_MONTH = re.compile(r"\b(?:(?P<year>\d{4})-(?P<month>\d{1,2})|(?P<m>\d{1,2})/(?P<y>\d{4}))\b")
_DAY_NUMBER = re.compile(r"(?:0?[1-9]|[12]\d|3[01])")
_DAY_RANGE_WORDS = {"al", "a", "y", "hasta", "to", "and", "through", "until"}
_DAY_ARTICLES = {"el", "los", "the"}
_VALUE_CODE = re.compile(r"\d{5,}")
_COORDINATION = re.compile(r"\s*,\s*|\s+(?:y|e|o|u|and|or)\s+", re.IGNORECASE)
_ENUMERATED_CODE = re.compile(r"[A-Za-z]?\d{1,3}[A-Za-z]?|[A-Z]")
_SEGMENT_CONNECTOR = re.compile(r"\s+(?:de|del|en|of|in)\s+", re.IGNORECASE)
_PART_CONNECTORS = {
    "de", "del", "la", "las", "el", "los", "en", "y", "e", "o", "u", "a", "al", "con", "por", "para",
    "of", "the", "in", "and", "or", "at",
}
_FORMULA_NAMES = {
    formula.casefold(): name for name, formulas in CHEMICAL_FORMULAS.items() for formula in formulas
}
_FORMULA = re.compile(
    rf"(?<![^\W\d_])(?:{'|'.join(map(re.escape, _FORMULA_NAMES))})(?!\d)", re.IGNORECASE
)
_MONTH_WORDS = "|".join(MONTH_NAMES)
_DATE_RANGE = re.compile(
    rf"(?:\d|\b(?:{_MONTH_WORDS}))\s*{RANGE_CONNECTORS}\s*(?:el\s+)?(?:\d|(?:{_MONTH_WORDS})\b)",
    re.IGNORECASE,
)
_DAY = timedelta(days=1)
_POSTFIX_COMPARISON = re.compile(
    rf"(?<![\w.,])(?P<number>{NUMBER.pattern})(?:\s*[^\s\d,;:?¿!()]+){{0,3}}?\s+(?:"
    + "|".join(f"(?P<{operator}>{'|'.join(cues)})" for operator, cues in POSTFIX_COMPARISON_CUES.items())
    + r")\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:"
    + "|".join(f"(?P<{operator}>{'|'.join(cues)})" for operator, cues in COMPARISON_CUES.items())
    + rf")\s+(?:{COMPARISON_REFERENCE}\s+)?(?:{COMPARISON_ARTICLES}\s+)?(?P<number>{NUMBER.pattern})",
    re.IGNORECASE,
)


class Comparison(NamedTuple):
    text: str
    operator: str
    value: int | float
    compares_statistic: bool


#
# Devuelve el valor numérico de cada palabra que expresa una cantidad.
#
def _word_counts() -> dict[str, int]:
    counts = {}

    for words in NUMBER_WORDS.values():
        for value, word in enumerate(words):
            counts[word] = value

    return counts


_WORD_COUNTS = _word_counts()


#
# Normaliza un texto: minúsculas, sin acentos, separando letras de dígitos y símbolos.
#
def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _LETTER_DIGIT_BOUNDARY.sub(" ", text)

    return " ".join(_SEPARATORS.sub(" ", text).split())


#
# Normaliza un nombre de archivo o fuente sin su extensión de planilla.
#
def normalize_name(value: str) -> str:
    return normalize_text(SPREADSHEET_EXTENSION.sub("", str(value).strip()))


#
# Devuelve el singular de una palabra para igualar plurales regulares.
#
def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]

    if len(word) > 3 and word.endswith("s"):
        return word[:-1]

    return word


#
# Devuelve las palabras normalizadas de un texto en singular.
#
def singular_words(text: str) -> tuple[str, ...]:
    return tuple(_singular(word) for word in normalize_text(text).split())


#
# Devuelve los grupos de sinónimos de variables, con los nombres más largos primero.
#
def _synonym_groups() -> list[list[tuple[str, ...]]]:
    return [
        sorted((singular_words(term) for term in group), key=len, reverse=True)
        for group in VARIABLE_SYNONYMS
    ]


#
# Reúne las palabras en singular de varios textos.
#
def _word_set(texts) -> set[str]:
    return {word for text in texts for word in singular_words(text)}


_SYNONYM_GROUPS = _synonym_groups()
_CONNECTORS = _word_set(MENTION_CONNECTORS)
_PERIOD_FILLERS = _CONNECTORS | {"ano", "year"}
_WATER = _word_set(WATER_WORDS)
_OTHER_MEDIUM = _word_set(OTHER_MEDIUM_WORDS)
_SUM = _word_set(SUM_WORDS)
_SECONDARY = _word_set(SECONDARY_QUANTITY_WORDS)


#
# Devuelve las palabras que precisan una variable, sin conectores ni palabras del agua.
#
def _content(words) -> tuple[str, ...]:
    return tuple(word for word in words if word not in _CONNECTORS and word not in _WATER)


#
# Devuelve los nombres equivalentes de cada variable registrada.
#
def _variable_terms() -> dict[str, set[tuple[str, ...]]]:
    terms = {name: {_content(singular_words(name))} for name in UNIT_CONFIRMED_SYMBOLS}
    for group in VARIABLE_SYNONYMS:
        terms.setdefault(group[0], set()).update(_content(singular_words(term)) for term in group)

    return terms


_VARIABLE_TERMS = _variable_terms()


#
# Reemplaza las fórmulas químicas de una frase por el nombre del compuesto.
#
def _formula_names(phrase: str) -> str:
    return _FORMULA.sub(lambda match: _FORMULA_NAMES[match.group().casefold()], phrase)


#
# Devuelve la variable y sus nombres equivalentes, incluida su fórmula química.
#
def variable_variants(phrase: str) -> list[str]:
    variants = [phrase, *_synonym_variants(phrase), *_synonym_variants(_formula_names(phrase))]

    return list(dict.fromkeys(variants))


#
# Devuelve los términos con que un texto nombra una variable: sus variantes completas y, si la variable incluye un
# nombre conocido («cianobacterias potencialmente tóxicas»), ese nombre en cada idioma («cyanobacteria»): un texto
# en otro idioma rara vez repite los calificativos palabra por palabra.
#
def subject_terms(phrase: str) -> list[str]:
    known = [" ".join(term) for index in sorted(named_variable_groups(phrase)) for term in _SYNONYM_GROUPS[index]]

    return list(dict.fromkeys([*variable_variants(variable_name(phrase)), *known]))


#
# Devuelve la frase con cada palabra abreviada escrita como en el encabezado de su columna («cianos potencialmente
# tóxicas» con «Recuento total de cianobacterias potencialmente tóxicas» es «cianobacterias potencialmente
# toxicas»): los documentos nombran la variable completa.
#
def spelled_as(phrase: str, header: str) -> str:
    written = normalize_text(header).split()
    words = [
        next((item for item in written if len(item) > len(word) and _same_word(_singular(word), item)), word)
        for word in normalize_text(phrase).split()
    ]

    return " ".join(words)


#
# Devuelve las frases que resultan de sustituir un nombre de variable por sus sinónimos.
#
def _synonym_variants(phrase: str) -> list[str]:
    words = singular_words(phrase)
    variants = []

    for group in _SYNONYM_GROUPS:
        span = next(
            (
                (index, index + len(term))
                for term in group
                for index in range(len(words) - len(term) + 1)
                if tuple(words[index : index + len(term)]) == term
            ),
            None,
        )
        if span is None:
            continue

        start, end = span
        variants += [" ".join([*words[:start], *term, *words[end:]]) for term in group]

    return variants


#
# Devuelve la variable registrada que la frase nombra sin más precisión que el agua.
#
def _exact_variable(phrase: str) -> str | None:
    words = _content(singular_words(_formula_names(phrase)))

    return next((name for name, terms in _VARIABLE_TERMS.items() if words in terms), None)


#
# Devuelve los grupos de variables de calidad de agua que nombra un texto.
#
def named_variable_groups(text: str) -> set[int]:
    words = singular_words(text)

    return {
        index
        for index, group in enumerate(_SYNONYM_GROUPS)
        for term in group
        if any(tuple(words[start : start + len(term)]) == term for start in range(len(words)))
    }


#
# Indica si la frase, sin estadísticas ni artículos, nombra una variable conocida («el pH», «la conductividad
# promedio»): permite separar una lista de variables sin partir un nombre compuesto.
#
def names_known_variable(phrase: str) -> bool:
    core = variable_core(phrase)

    return bool(named_variable_groups(core)) or is_ph_name("".join(_content(singular_words(core))))


#
# Indica si la variable es un nutriente que se expresa respecto de una especie química.
#
def requires_chemical_basis(phrase: str) -> bool:
    return _exact_variable(phrase) in CHEMICAL_FORMULAS


#
# Indica si el encabezado nombra con su fórmula química el nutriente pedido.
#
def formula_named(name: str, phrase: str) -> bool:
    formulas = CHEMICAL_FORMULAS.get(_exact_variable(phrase), ())

    return any(
        re.search(rf"(?<![A-Za-z\d/:]){re.escape(formula)}(?![a-z\d]|\s*[/:])", name)
        for formula in formulas
    )


#
# Indica si un símbolo ambiguo nombra la variable pedida según su unidad documentada.
#
def symbol_named(name: str, unit: str | None, phrase: str) -> bool:
    variable = _exact_variable(phrase)
    if variable not in UNIT_CONFIRMED_SYMBOLS or not unit:
        return False

    symbols, units = UNIT_CONFIRMED_SYMBOLS[variable]
    return normalize_text(unitless_name(name)) in symbols and unit_key(unit) in units


#
# Devuelve la variable que la frase pide en la unidad de una columna, sin esa unidad.
#
def variable_in_unit(phrase: str, name: str, unit: str | None) -> str | None:
    if not unit:
        return None

    measured = comparable_unit(name, unit)
    words = phrase.split()
    for index in range(1, len(words)):
        asked = _without_final_connectors(words[:index])
        stated = UNIT_RATIO_WORD.sub("/", " ".join(words[index:]))
        if comparable_unit(asked, stated.strip(SURROUNDING_PUNCTUATION)) == measured:
            return variable_name(asked)

    return None


#
# Une las palabras dadas sin los conectores con que terminan.
#
def _without_final_connectors(words) -> str:
    kept = list(words)
    while kept and normalize_text(kept[-1]) in _CONNECTORS:
        kept.pop()

    return " ".join(kept)


#
# Quita de la variable el medio que la precisa para buscar sus columnas.
#
def without_medium(phrase: str) -> str:
    words = normalize_text(phrase).split()
    kept = [word for word in words if _singular(word) not in _OTHER_MEDIUM]
    while kept and kept[-1] in _CONNECTORS:
        kept.pop()

    return " ".join(kept) if kept and len(kept) < len(words) else phrase


#
# Devuelve la variable con el calificativo que la sigue en la pregunta y cambia su magnitud.
#
def qualified_variable(mention: str, question: str) -> str:
    words, target = normalize_text(question).split(), normalize_text(mention).split()
    qualifiers = _OTHER_MEDIUM | _SECONDARY

    for index in range(len(words) - len(target) + 1):
        if not target or words[index : index + len(target)] != target:
            continue
        following = words[index + len(target) : index + len(target) + 3]
        for offset, word in enumerate(following):
            if _singular(word) in qualifiers:
                return " ".join([mention, *following[: offset + 1]])
            if word not in _CONNECTORS:
                break

    return mention


#
# Indica si un valor es solo el calificativo que la variable ya incluye.
#
def qualifies_variable(value: str, variable: str) -> bool:
    words = set(singular_words(value))
    return bool(words) and words <= (_OTHER_MEDIUM | _SECONDARY) and words <= set(singular_words(variable))


#
# Indica si un encabezado mide otra magnitud que la pedida: otro medio o una suma.
#
def other_quantity(name: str, phrase: str) -> str | None:
    words, asked = set(singular_words(name)), set(singular_words(phrase))

    if bool(words & _OTHER_MEDIUM) != bool(asked & _OTHER_MEDIUM):
        return "medium"
    if DERIVED_SUM.search(name) and not (DERIVED_SUM.search(phrase) or asked & _SUM):
        return "sum"
    return None


#
# Indica si un encabezado es una variante secundaria que la frase no pide, como un porcentaje.
#
def secondary_quantity(name: str, unit: str | None, phrase: str) -> bool:
    described = f"{name} {unit or ''}"
    marked = PERCENT_SIGN in described or bool(set(singular_words(described)) & _SECONDARY)
    asked = PERCENT_SIGN in phrase or bool(set(singular_words(phrase)) & _SECONDARY)

    return marked and not asked


#
# Devuelve la variable normalizada sin las estadísticas que la acompañan.
#
def variable_core(phrase: str) -> str:
    words = _without_inner_statistics(normalize_text(phrase).split())

    while words and words[0] in STATISTIC_WORDS:
        words.pop(0)
    while words and words[-1] in STATISTIC_WORDS:
        words.pop()

    return " ".join(words)


#
# Quita las estadísticas escritas entre las palabras de la variable: «temp max del agua».
#
def _without_inner_statistics(words: list[str]) -> list[str]:
    inner = [
        word
        for index, word in enumerate(words)
        if not (0 < index < len(words) - 1 and normalize_text(word) in INNER_STATISTIC_WORDS)
    ]
    return inner if inner else words


#
# Devuelve la variable como la escribió el usuario, sin su estadística.
#
def variable_name(phrase: str) -> str:
    words = _without_inner_statistics(phrase.split())

    while words and normalize_text(words[0]) in STATISTIC_WORDS:
        words.pop(0)
    while words and normalize_text(words[-1]) in STATISTIC_WORDS:
        words.pop()

    return " ".join(words) or phrase


#
# Devuelve la cantidad entera mencionada en cifras o palabras, si es única.
#
def mentioned_count(phrase: str) -> int | None:
    counts = set()

    for token in str(phrase).split():
        token = token.strip(SURROUNDING_PUNCTUATION)
        if INTEGER.fullmatch(token):
            counts.add(int(_THOUSANDS_SEPARATORS.sub("", token)))
        elif (word := normalize_text(token)) in _WORD_COUNTS:
            counts.add(_WORD_COUNTS[word])

    counts.discard(0)

    return counts.pop() if len(counts) == 1 else None


#
# Extrae las comparaciones numéricas (umbrales) literales de un texto.
#
def numeric_comparisons(text: str) -> list[Comparison]:
    comparisons = []

    for match in _COMPARISON.finditer(str(text)):
        value = _comparison_number(match.group("number"))
        if value is None:
            continue

        operator = next(name for name in COMPARISON_CUES if match.group(name))
        statistic = _compared_statistic(text, match.start())
        following = None if statistic is not None else _following_statistic(text, match.end())
        start = match.start() if statistic is None else statistic
        end = match.end() if following is None else following
        compares = statistic is not None or following is not None
        comparisons.append(Comparison(text[start:end], operator, value, compares))

    taken = [(item.start(), item.end()) for item in _COMPARISON.finditer(str(text))]
    for match in _POSTFIX_COMPARISON.finditer(str(text)):
        value = _comparison_number(match.group("number"))
        if value is None or any(start < match.end() and match.start() < end for start, end in taken):
            continue

        operator = next(name for name in POSTFIX_COMPARISON_CUES if match.group(name))
        statistic = _compared_statistic(text, match.start())
        start = match.start() if statistic is None else statistic
        comparisons.append(Comparison(text[start : match.end()], operator, value, statistic is not None))

    return comparisons


#
# Devuelve el final de la estadística que sigue a un número comparado, si la hay.
#
def _following_statistic(text: str, start: int) -> int | None:
    skipped = 0

    for word in list(_WORD.finditer(text[start:]))[:_MAX_FOLLOWING_WORDS]:
        normalized = normalize_text(word.group())
        if normalized in COMPARED_STATISTICS:
            return start + word.end()
        if normalized in COMPARISON_BOUNDARIES:
            return None
        if normalized in COMPARISON_FILLERS or normalized in FOLLOWING_STATISTIC_FILLERS:
            continue

        skipped += 1
        if skipped > 2:
            return None

    return None


#
# Devuelve el único intervalo de fechas que nombra el texto, o None si hay varios o ninguno.
#
def period_range(text: str) -> tuple[date | None, date | None] | None:
    ranges = period_ranges(text)
    return ranges[0] if len(ranges) == 1 else None


#
# Devuelve los intervalos de fechas que nombra un texto, abiertos o cerrados.
#
def period_ranges(text: str) -> list[tuple[date | None, date | None]]:
    text = str(text)
    seasonal = contains_any_phrase(text, SEASON_WORDS) and not _month_words(text)
    anchors = [] if seasonal else _period_anchors(text)
    if not anchors:
        return []

    sides = {side for side, cues in PERIOD_BOUND_CUES.items() if contains_any_phrase(text, cues)}
    if sides == {"from", "until"} and contains_any_phrase(text, OPEN_END_WORDS):
        sides = {"from"}
    if len(anchors) == 1:
        return _bounded(anchors[0], sides)

    masked = _EXPLICIT_MONTH.sub("0000", _EXPLICIT_DAY.sub("0000", text))
    ranged = contains_any_phrase(text, RANGE_CUES) or _DATE_RANGE.search(masked)
    if ranged or sides == {"from", "until"}:
        return [(anchors[0][0], anchors[-1][1])]

    return [] if sides else anchors


#
# Aplica los extremos «desde», «hasta», «antes» o «después» a una fecha de referencia.
#
def _bounded(anchor, sides) -> list[tuple[date | None, date | None]]:
    start, end = anchor
    lower = start if "from" in sides else end + _DAY if "after" in sides else None
    upper = end if "until" in sides else start - _DAY if "before" in sides else None
    if lower is None and upper is None:
        return [anchor] if not sides else []
    if lower is not None and upper is not None:
        return []

    return [(lower, upper)]


#
# Devuelve las fechas que nombra el texto como días, meses o años, en orden.
#
def _period_anchors(text) -> list[tuple[date, date]]:
    years = sorted({int(year) for year in REQUESTED_YEAR.findall(text)})
    days = sorted(
        {
            *(day for match in _EXPLICIT_DAY.finditer(text) if (day := _explicit_day(match))),
            *_named_days(text, years),
        }
    )
    if days:
        return [(day, day) for day in days]

    explicit = sorted(
        {
            (int(match.group("year") or match.group("y")), month)
            for match in _EXPLICIT_MONTH.finditer(text)
            if (month := _explicit_month(match))
        }
    )
    named = _named_months(text, years)
    if named is None:
        return []
    explicit = sorted({*explicit, *named})

    if explicit:
        return [
            (date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
            for year, month in explicit
        ]
    return [(date(year, 1, 1), date(year, 12, 31)) for year in years]


#
# Devuelve los días que el texto escribe con el nombre del mes («22 de septiembre de 2021», «September 22, 2021»,
# «22 sep. 2021»), y el otro extremo de un rango de días del mismo mes («del 12 al 28 de enero de 2022»). El año es
# el que sigue al mes o el único del texto; una abreviatura cuenta sólo entre un día y un año.
#
def _named_days(text, years) -> list[date]:
    words = normalize_text(text).split()
    days = []
    for index, word in enumerate(words):
        month = MONTH_NAMES.get(word) or _abbreviated_month(words, index)
        following = next((int(item) for item in words[index + 1 : index + 4] if REQUESTED_YEAR.fullmatch(item)), None)
        year = following or (years[0] if len(years) == 1 else None)
        if month is None or year is None:
            continue

        for day in _month_days(words, index):
            try:
                days.append(date(year, month, day))
            except ValueError:
                continue
    return days


#
# Devuelve el mes de una abreviatura escrita entre un día y un año («22 sep 2021»), o None.
#
def _abbreviated_month(words, index) -> int | None:
    month = MONTH_ABBREVIATIONS.get(words[index])
    dated = 0 < index < len(words) - 1 and _DAY_NUMBER.fullmatch(words[index - 1])
    if month is None or not dated or not any(REQUESTED_YEAR.fullmatch(item) for item in words[index + 1 : index + 3]):
        return None

    return month


#
# Devuelve los números de día que acompañan a un mes: el que lo precede («22 de septiembre»), el que lo sigue
# («September 22») y el primero de un rango de días que termina en el precedente («del 12 al 28 de enero»).
#
def _month_days(words, index) -> list[int]:
    start = index - 2 if index >= 2 and words[index - 1] in {"de", "del"} else index - 1
    if start >= 0 and _DAY_NUMBER.fullmatch(words[start]):
        before = [word for word in words[max(0, start - 4) : start] if word not in _DAY_ARTICLES]
        ranged = len(before) >= 2 and before[-1] in _DAY_RANGE_WORDS and _DAY_NUMBER.fullmatch(before[-2])
        return [int(before[-2]), int(words[start])] if ranged else [int(words[start])]

    after = words[index + 1] if index + 1 < len(words) else ""
    return [int(after)] if _DAY_NUMBER.fullmatch(after) else []


#
# Indica si un texto nombra un período concreto con un año o un mes.
#
def names_period(text: str) -> bool:
    return bool(REQUESTED_YEAR.search(str(text)) or _month_words(text))


#
# Devuelve las palabras de un texto que nombran meses, con su posición.
#
def _month_words(text) -> list[tuple[int, int]]:
    words = normalize_text(text).split()

    return [(index, MONTH_NAMES[word]) for index, word in enumerate(words) if word in MONTH_NAMES]


#
# Devuelve cada mes nombrado con su año: el que lo sigue («diciembre de 2021»), el único año del texto o, en un
# rango de dos meses que cruza de año («verano 2021-2022, de diciembre a febrero»), el primero y el siguiente.
# None si algún mes queda sin un año inequívoco.
#
def _named_months(text, years) -> set[tuple[int, int]] | None:
    words = normalize_text(text).split()
    months = _month_words(text)
    following = [
        next((int(word) for word in words[index + 1 : index + 3] if REQUESTED_YEAR.fullmatch(word)), None)
        for index, _ in months
    ]
    crossing = (
        len(months) == 2 and len(years) == 2 and years[1] == years[0] + 1 and months[0][1] > months[1][1]
    )
    named = set()
    for position, ((_, month), year) in enumerate(zip(months, following, strict=True)):
        if year is None and len(years) == 1:
            year = years[0]
        elif year is None and crossing:
            year = years[position]
        if year is None:
            return None
        named.add((year, month))

    return named


#
# Devuelve el nombre legible de un período cerrado: año, mes, día o rango.
#
def period_text(start: date, end: date) -> str:
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31) and start.year == end.year:
        return str(start.year)
    if start.day == 1 and end == date(start.year, start.month, calendar.monthrange(start.year, start.month)[1]):
        return f"{_SPANISH_MONTHS[start.month - 1]} de {start.year}"
    if start == end:
        return start.strftime("%d/%m/%Y")
    return f"{start.strftime('%d/%m/%Y')} al {end.strftime('%d/%m/%Y')}"


_SPANISH_MONTHS = list(MONTH_NAMES)[:12]


#
# Devuelve la fecha con el extremo o rango que la precede en la pregunta y se omitió.
#
def qualified_period(mention: str, question: str) -> str:
    words, target = normalize_text(question).split(), normalize_text(mention).split()
    cues = [cue.split() for phrases in PERIOD_BOUND_CUES.values() for cue in phrases]
    cues += [cue.split() for cue in RANGE_CUES]
    if not target or any(contains_phrase(mention, " ".join(cue)) for cue in cues):
        return mention

    for index in range(len(words) - len(target) + 1):
        if words[index : index + len(target)] != target:
            continue
        start = index
        while start > 0 and _singular(words[start - 1]) in _PERIOD_FILLERS:
            start -= 1
        cue = next((cue for cue in cues if words[max(0, start - len(cue)) : start] == cue), None)
        if cue is not None:
            before = _original_words(question, mention, index - start + len(cue))
            return " ".join([*(before or words[start - len(cue) : index]), mention])

    return mention


#
# Devuelve las últimas palabras antes de una mención, tal como las escribió el usuario.
#
def _original_words(question: str, mention: str, count: int) -> list[str]:
    position = question.casefold().find(mention.casefold())
    words = [word.strip(SURROUNDING_PUNCTUATION) for word in question[:position].split()]
    words = [word for word in words if word]
    return words[-count:] if position >= 0 and len(words) >= count else []


#
# Devuelve el mes de una fecha en formato AAAA-MM o MM/AAAA, si es válido.
#
def _explicit_month(match) -> int | None:
    month = int(match.group("month") or match.group("m"))
    return month if 1 <= month <= 12 else None


#
# Devuelve el día de una fecha explícita AAAA-MM-DD o DD/MM/AAAA, si es válida.
#
def _explicit_day(match) -> date | None:
    try:
        if match.group("iso"):
            return date(*(int(part) for part in match.group("iso").split("-")))
        day, month, year = (int(part) for part in match.group("dmy").split("/"))
        return date(year, month, day)
    except ValueError:
        return None


#
# Indica si un texto puede buscarse como valor guardado: tiene una palabra de al menos tres letras o es un
# código numérico largo («03122004»); un número suelto («2») figura en casi cualquier registro.
#
def searchable_value(text: str) -> bool:
    words = normalize_text(text).split()

    return any(len(word) >= 3 and word.isalpha() for word in words) or bool(_VALUE_CODE.fullmatch(text.strip()))


#
# Devuelve las partes en que se puede dividir la mención de un valor: los elementos de una enumeración («Pta Lara
# y Balandra»), los tramos entre conectores («La Balandra» y «Berisso») y cada frase de al menos dos palabras,
# menor que la mención, que no empieza ni termina en un conector («cruda superficial», «Punta Lara»).
#
def value_parts(text: str) -> list[str]:
    words = _value_words(text)
    phrases = [
        " ".join(words[start:end])
        for start in range(len(words))
        for end in range(start + 2, len(words) + 1)
        if end - start < len(words)
        and words[start].casefold() not in _PART_CONNECTORS
        and words[end - 1].casefold() not in _PART_CONNECTORS
    ]
    segments = [segment.strip() for segment in _SEGMENT_CONNECTOR.split(" ".join(words))]
    segments = [segment for segment in segments if segment and segment != " ".join(words)]

    return list(dict.fromkeys([*coordinated_parts(text), *segments, *phrases]))


#
# Devuelve los elementos de una enumeración («A y B», «A, B o C»); en una enumeración de códigos con un nombre común
# («estaciones 1, 2 y 3»), cada código con el nombre en singular («estacion 1», «estacion 2», «estacion 3»). Nada si
# no es una enumeración o algún elemento no se puede buscar como valor.
#
def coordinated_parts(text: str) -> list[str]:
    parts = [part.strip(f"{SURROUNDING_PUNCTUATION} ") for part in _COORDINATION.split(text)]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return []
    if all(searchable_value(part) for part in parts):
        return parts

    *head, first = parts[0].split()
    codes = [first, *parts[1:]]
    if not head or not all(_ENUMERATED_CODE.fullmatch(code) for code in codes):
        return []

    noun = " ".join([*head[:-1], _singular(head[-1])])
    return [f"{noun} {code}" for code in codes]


#
# Elige las partes guardadas que reemplazan a la mención de un valor, a partir de dónde figura cada una (columnas
# por dataset): la enumeración entera si todos sus elementos figuran o, si no, las frases guardadas más largas que no
# se superponen. Partes guardadas en columnas distintas se combinan («Cruda Superficial» es un lugar y «Punta Lara»
# una localidad); si comparten columna, la que sigue sólo describe a la primera («La Balandra de Berisso») y se
# deja. None si no hay ninguna o si lo que queda afuera incluye un nombre propio (salvo la mayúscula inicial).
#
def stored_parts(text: str, parts: list[str], stored: dict) -> list[str] | None:
    coordinated = coordinated_parts(text)
    if coordinated and all(part in stored for part in coordinated):
        return coordinated

    words = _value_words(text)
    chosen = []
    for phrase in sorted((part for part in parts if part in stored), key=lambda part: -len(part.split())):
        span = _word_span(words, phrase.split())
        if span is not None and not any(span[0] < end and start < span[1] for start, end in chosen):
            chosen.append(span)
    if not chosen:
        return None

    covered = {index for start, end in chosen for index in range(start, end)}
    left = [
        word
        for index, word in enumerate(words)
        if index not in covered and word.casefold() not in _PART_CONNECTORS
    ]
    if any(word[:1].isupper() for word in left if word is not words[0]):
        return None

    kept, columns = [], set()
    for start, end in sorted(chosen):
        part = " ".join(words[start:end])
        if not stored[part] & columns:
            kept.append(part)
            columns |= stored[part]
    return kept


#
# Devuelve las palabras de la mención de un valor sin la puntuación que las rodea.
#
def _value_words(text) -> list[str]:
    return [word for word in (item.strip(SURROUNDING_PUNCTUATION) for item in str(text).split()) if word]


#
# Devuelve dónde empieza y termina una frase entre las palabras de un texto, o None.
#
def _word_span(words, phrase) -> tuple[int, int] | None:
    size = len(phrase)

    return next(
        ((start, start + size) for start in range(len(words) - size + 1) if words[start : start + size] == phrase),
        None,
    )


#
# Devuelve el número con que empieza un texto («20.000», «25 °C», «24,5»), leído como en una comparación, o
# None si no empieza con un número.
#
def parsed_number(text: str) -> int | float | None:
    found = _LEADING_NUMBER.fullmatch(str(text).strip())

    return _comparison_number(found.group(1)) if found else None


#
# Convierte el número de una comparación, distinguiendo miles de decimales.
#
def _comparison_number(token: str) -> int | float | None:
    unsigned = token.lstrip("-")
    sign = -1 if token.startswith("-") else 1

    if INTEGER.fullmatch(unsigned) and not unsigned.startswith("0"):
        return sign * int(_THOUSANDS_SEPARATORS.sub("", unsigned))

    try:
        return sign * float(unsigned.replace(",", "."))
    except ValueError:
        return None


#
# Devuelve el inicio de la estadística comparada que precede a una comparación.
#
def _compared_statistic(text: str, end: int) -> int | None:
    skipped = 0

    for word in reversed(list(_WORD.finditer(text[:end]))):
        normalized = normalize_text(word.group())
        if normalized in COMPARED_STATISTICS:
            return word.start()
        if normalized in COMPARISON_BOUNDARIES:
            return None
        if normalized in COMPARISON_FILLERS:
            continue

        skipped += 1
        if skipped > 1:
            return None

    return None


#
# Indica si la frase aparece en el texto como secuencia completa de palabras.
#
def contains_phrase(text: str, phrase: str, *, plural_insensitive: bool = False) -> bool:
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


#
# Indica si el texto empieza con alguna de las frases dadas.
#
def starts_with_phrase(text: str, phrases) -> bool:
    words = normalize_text(text).split()

    return any(words[: len(phrase.split())] == phrase.split() for phrase in phrases if phrase)


#
# Indica si el texto contiene alguna de las frases dadas.
#
def contains_any_phrase(text: str, phrases) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)


#
# Indica si el texto escribe una frase de varias palabras como una sola («microcystinLR»), como queda cuando la
# extracción de un PDF pierde el guion de un corte de línea.
#
def contains_joined_phrase(text: str, phrase: str) -> bool:
    target = normalize_text(phrase).split()

    return len(target) > 1 and "".join(target) in normalize_text(text).split()


#
# Devuelve el tramo del texto que escribe la mención aunque tenga una errata o esté abreviado («conductivdad»,
# «temp», «pta»), alineando cada palabra con una palabra consecutiva; None si no se alinea.
#
def aligned_span(mention: str, text: str) -> str | None:
    target = normalize_text(mention).split()
    words = normalize_text(text).split()
    for start in range(len(words) - len(target) + 1 if target else 0):
        window = words[start : start + len(target)]
        if all(_same_word(written, said) for written, said in zip(target, window)):
            return " ".join(window)

    return None


#
# Indica si un valor guardado escribe una mención de varias palabras con abreviaturas («Pta Lara», «Punta Lara») o
# con iniciales, conectores o símbolos entre sus palabras («AVENIDA ALTE G BROWN N° 419» escribe «almirante brown
# 419»): cada palabra de la mención se alinea, en orden, con una del valor y entre ellas sólo puede haber palabras
# de una o dos letras; un número nunca se saltea.
#
def abbreviates(value: str, mention: str) -> bool:
    words = normalize_text(value).split(), normalize_text(mention).split()
    if len(words[0]) == len(words[1]) > 1 and aligned_span(mention, value) is not None:
        return True

    return len(words[1]) > 1 and any(
        _aligned_from(words[0], start, words[1]) for start in range(len(words[0]))
    )


#
# Indica si la mención se alinea con el valor desde una posición, salteando sólo palabras de relleno.
#
def _aligned_from(words, start, target) -> bool:
    position = start
    for index, said in enumerate(target):
        while index and position < len(words) and _filler(words[position]) and not _same_word(words[position], said):
            position += 1
        if position >= len(words) or not _same_word(words[position], said):
            return False
        position += 1

    return True


#
# Indica si una palabra de un valor guardado es relleno entre las de un nombre: una inicial, un conector corto o un
# símbolo (como «n» de «N°»), nunca un número.
#
def _filler(word) -> bool:
    return len(word) <= _MAX_FILLER_LETTERS and word.isalpha()


#
# Indica si dos palabras son la misma: iguales, abreviada una de la otra o con una errata. Dos términos
# conocidos sólo son la misma palabra si son sinónimos: nitrato y nitrito nunca se confunden.
#
def _same_word(first: str, second: str) -> bool:
    if first == second:
        return True
    if any(char.isdigit() for char in first + second):
        return False

    groups = _KNOWN_WORDS.get(first), _KNOWN_WORDS.get(second)
    if groups[0] and groups[1]:
        return bool(groups[0] & groups[1])

    short, long = sorted((first, second), key=len)
    if len(short) >= _MIN_PREFIX and long.startswith(short):
        return True
    if _contracted(short, long):
        return True

    return short[0] == long[0] and _edit_distance(first, second) <= _allowed_edits(len(short))


#
# Indica si una palabra corta es la contracción de otra: misma inicial y final, letras en orden («pta», «punta»).
#
def _contracted(short: str, long: str) -> bool:
    if not _MIN_CONTRACTION <= len(short) <= _MAX_CONTRACTION or len(long) < len(short) + 2:
        return False
    if short[0] != long[0] or short[-1] != long[-1]:
        return False

    letters = iter(long)
    return all(char in letters for char in short)


#
# Devuelve cuántas ediciones admite una palabra según su largo: ninguna si es corta.
#
def _allowed_edits(length: int) -> int:
    if length >= _LONG_WORD:
        return 2

    return 1 if length >= _MIN_TYPO_WORD else 0


#
# Devuelve la distancia de edición entre dos palabras, contando la transposición de letras vecinas.
#
def _edit_distance(first: str, second: str) -> int:
    previous, current = None, list(range(len(second) + 1))
    for row, char in enumerate(first, start=1):
        before, previous, current = previous, current, [row] + [0] * len(second)
        for column, other in enumerate(second, start=1):
            current[column] = min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (char != other),
            )
            swapped = row > 1 and column > 1 and char == second[column - 2] and first[row - 2] == other
            if swapped:
                current[column] = min(current[column], before[column - 2] + 1)

    return current[-1]


#
# Devuelve los grupos de sinónimos de variables a los que pertenece cada palabra conocida. Una palabra de un
# sinónimo compuesto es conocida pero sólo equivale a sí misma: «fósforo» no es «fosfato».
#
def _known_words() -> dict[str, set]:
    known = {}
    groups = [*VARIABLE_SYNONYMS, *((name,) for name in CHEMICAL_FORMULAS)]
    for index, group in enumerate(groups):
        for phrase in group:
            words = normalize_text(phrase).split()
            if len(words) == 1:
                known.setdefault(words[0], set()).add(index)
    for group in groups:
        for word in (word for phrase in group for word in normalize_text(phrase).split()):
            known.setdefault(word, {word})

    return known


#
# Devuelve los documentos que la pregunta nombra por autor, sigla, año o título completo.
#
def named_documents(question: str, documents) -> list[dict]:
    available = [item for item in documents if item.get("document_id") and item.get("is_active", True)]

    return [item for item in available if _names_document(question, item)]


#
# Indica si la pregunta nombra un documento por su título completo o por su autor o sigla, aun con una errata
# en un apellido largo («Giannuzzi» por «Gianuzzi»). Un año escrito junto al autor («Sathicq et al. (2014)»)
# elige entre sus documentos; los demás años de la pregunta son del período consultado, no de la cita.
#
def _names_document(question: str, document) -> bool:
    title = document.get("title") or ""
    if len(title.split()) >= 4 and contains_phrase(question, title):
        return True

    names = [name for pattern in DOCUMENT_AUTHOR_NAMES for name in pattern.findall(document["name"])]
    ends = [end for name in names if len(name) >= 3 for end in _author_mentions(question, name)]
    if not ends:
        return False

    cited = {year for end in ends for year in DOCUMENT_YEAR.findall(question[end : end + _CITED_YEAR_WINDOW])}
    own = _document_years(document)

    return not cited or not own or bool(own & cited)


#
# Devuelve dónde termina cada mención de un autor en la pregunta: escrito igual o, si el apellido es largo, con
# una sola errata («Giannuzzi» por «Gianuzzi»). No se admiten abreviaturas ni contracciones: «agua» no es
# «Aguilera».
#
def _author_mentions(question: str, name: str) -> list[int]:
    target = normalize_text(name)
    words = [(normalize_text(match.group()), match.end()) for match in _WORD.finditer(question)]
    exact = [end for word, end in words if word == target]
    if exact or len(target) < _MIN_FUZZY_AUTHOR:
        return exact

    return [end for word, end in words if _misspells(word, target)]


#
# Indica si una palabra es un apellido escrito con una sola errata: misma inicial y una edición de distancia.
#
def _misspells(word: str, name: str) -> bool:
    close = abs(len(word) - len(name)) <= 1 and word[:1] == name[:1]

    return close and _edit_distance(word, name) <= 1


#
# Devuelve los años que figuran en el nombre de un documento.
#
def _document_years(document) -> set[str]:
    return set(DOCUMENT_YEAR.findall(document["name"]))


#
# Quita el pedido de una línea de referencia: es una presentación documental, no una operación.
#
def without_reference_request(question: str) -> str:
    return REFERENCE_REQUEST_CLAUSE.sub("", question).strip() or question


#
# Indica si un texto nombra alguno de los términos de un tema: cada palabra del término figura en él, las
# siglas enteras y el resto por su raíz («cyanobacterial» nombra «cyanobacteria»).
#
def names_subject(text: str, subject) -> bool:
    words = set(normalize_text(text).split())

    return any(_covers_term(term, words) for term in subject)


#
# Indica si todas las palabras de un término aparecen entre las palabras de un texto.
#
def _covers_term(term, words) -> bool:
    terms = [word for word in normalize_text(term).split() if word not in SEARCH_STOPWORDS]

    return bool(terms) and all(_term_present(word, words) for word in terms)


#
# Indica si una palabra figura en un texto: las siglas enteras, el resto por su raíz.
#
def _term_present(term, words) -> bool:
    if len(term) <= _SHORT_TERM:
        return term in words

    root = term[: max(_MIN_TERM_ROOT, len(term) - _TERM_FLEXION_CHARS)]

    return any(word.startswith(root) for word in words)


#
# Devuelve la referencia documental con que la pregunta pide comparar los datos sin dar su número
# («superan el nivel guía de la OMS»), o None si no la hay o si la comparación ya trae un número.
#
def documented_reference(question: str) -> str | None:
    if reference_operator(question) is None or numeric_comparisons(question):
        return None

    return asked_reference(question)


#
# Devuelve la referencia documental que la pregunta nombra («los valores guía de la OMS para aguas
# recreativas»), desde su primera mención hasta el final, haya o no un sentido de comparación.
#
def asked_reference(question: str) -> str | None:
    cue = REFERENCE_LINE_CUES.search(question)

    return (question[cue.start() :].strip(" ¿?¡!.") or None) if cue else None


#
# Devuelve el sentido de la comparación pedida con una referencia: por encima (gt) o por debajo (lt), o incluido el
# límite («alcanzaron o superaron»: ge; «como máximo»: le).
#
def reference_operator(question: str) -> str | None:
    above = COMPARISON_DIRECTIONS["above"].search(question) is not None
    below = COMPARISON_DIRECTIONS["below"].search(question) is not None
    if above == below:
        return None

    inclusive = INCLUSIVE_BOUND.search(question) is not None
    if above:
        return "ge" if inclusive else "gt"
    return "le" if inclusive else "lt"


_KNOWN_WORDS = _known_words()
