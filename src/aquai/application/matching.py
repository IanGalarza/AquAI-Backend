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
    COMPARED_STATISTICS,
    COMPARISON_ARTICLES,
    COMPARISON_BOUNDARIES,
    COMPARISON_CUES,
    COMPARISON_FILLERS,
    FOLLOWING_STATISTIC_FILLERS,
    INTEGER,
    MENTION_CONNECTORS,
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
from aquai.application.Rules.evidence import (
    COMPARISON_DIRECTIONS,
    DOCUMENT_AUTHOR_NAMES,
    DOCUMENT_YEAR,
    REFERENCE_LINE_CUES,
    REFERENCE_REQUEST_CLAUSE,
    SEARCH_STOPWORDS,
)

_SHORT_TERM = 4
_MIN_TERM_ROOT = 4
_TERM_FLEXION_CHARS = 2
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
_COMPARISON = re.compile(
    r"\b(?:"
    + "|".join(f"(?P<{operator}>{'|'.join(cues)})" for operator, cues in COMPARISON_CUES.items())
    + rf")\s+(?:{COMPARISON_ARTICLES}\s+)?(?P<number>{NUMBER.pattern})",
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
    words = normalize_text(phrase).split()

    while words and words[0] in STATISTIC_WORDS:
        words.pop(0)
    while words and words[-1] in STATISTIC_WORDS:
        words.pop()

    return " ".join(words)


#
# Devuelve la variable como la escribió el usuario, sin su estadística.
#
def variable_name(phrase: str) -> str:
    words = phrase.split()

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
    anchors = [] if contains_any_phrase(text, SEASON_WORDS) else _period_anchors(text)
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
    days = sorted({day for match in _EXPLICIT_DAY.finditer(text) if (day := _explicit_day(match))})
    if days:
        return [(day, day) for day in days]

    years = sorted({int(year) for year in REQUESTED_YEAR.findall(text)})
    explicit = sorted(
        {
            (int(match.group("year") or match.group("y")), month)
            for match in _EXPLICIT_MONTH.finditer(text)
            if (month := _explicit_month(match))
        }
    )
    named = sorted({MONTH_NAMES[word] for word in normalize_text(text).split() if word in MONTH_NAMES})
    if named and len(years) == 1:
        explicit = sorted({*explicit, *((years[0], month) for month in named)})
    elif named:
        return []

    if explicit:
        return [
            (date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
            for year, month in explicit
        ]
    return [(date(year, 1, 1), date(year, 12, 31)) for year in years]


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
# Devuelve los documentos que la pregunta nombra por autor, sigla, año o título completo.
#
def named_documents(question: str, documents) -> list[dict]:
    available = [item for item in documents if item.get("document_id") and item.get("is_active", True)]
    named = [item for item in available if _names_document(question, item)]
    years = set(DOCUMENT_YEAR.findall(question))
    if not years:
        return named

    return [item for item in named if not _document_years(item) or _document_years(item) & years]


#
# Indica si la pregunta nombra un documento por su autor, su sigla o su título.
#
def _names_document(question: str, document) -> bool:
    title = document.get("title") or ""
    if len(title.split()) >= 4 and contains_phrase(question, title):
        return True

    names = [name for pattern in DOCUMENT_AUTHOR_NAMES for name in pattern.findall(document["name"])]

    return any(contains_phrase(question, name) for name in names if len(name) >= 3)


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
    cue = REFERENCE_LINE_CUES.search(question)
    compared = reference_operator(question) is not None
    if cue is None or not compared or numeric_comparisons(question):
        return None

    return question[cue.start() :].strip(" ¿?¡!.") or None


#
# Devuelve el sentido de la comparación pedida con una referencia: por encima (gt) o por debajo (lt).
#
def reference_operator(question: str) -> str | None:
    above = COMPARISON_DIRECTIONS["above"].search(question) is not None
    below = COMPARISON_DIRECTIONS["below"].search(question) is not None
    if above == below:
        return None

    return "gt" if above else "lt"
