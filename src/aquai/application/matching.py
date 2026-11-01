import calendar
import re
import unicodedata
from datetime import date, timedelta
from typing import NamedTuple

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
from aquai.application.units import unit_key, unitless_name

_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_SEPARATORS = re.compile(r"[\W_]+")
_THOUSANDS_SEPARATORS = re.compile(r"[.,]")
_WORD = re.compile(r"\w+")
# Palabras tras el número en las que se busca la estadística comparada: «500 de conductividad en promedio».
_MAX_FOLLOWING_WORDS = 5
_EXPLICIT_DAY = re.compile(
    r"\b(?:(?P<iso>\d{4}-\d{1,2}-\d{1,2})|(?P<dmy>\d{1,2}/\d{1,2}/\d{4}))\b"
)
# Un mes con su año en cifras: «2021-03», «03/2021».
_EXPLICIT_MONTH = re.compile(r"\b(?:(?P<year>\d{4})-(?P<month>\d{1,2})|(?P<m>\d{1,2})/(?P<y>\d{4}))\b")
# Fórmula química escrita en una pregunta: «NO3», «no3».
_FORMULA_NAMES = {
    formula.casefold(): name for name, formulas in CHEMICAL_FORMULAS.items() for formula in formulas
}
_FORMULA = re.compile(
    rf"(?<![^\W\d_])(?:{'|'.join(map(re.escape, _FORMULA_NAMES))})(?!\d)", re.IGNORECASE
)
# Dos fechas unidas como rango: «2019-2021», «01/03/2021 al 15/04/2021», «marzo a junio».
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


def singular_words(text: str) -> tuple[str, ...]:
    """Palabras normalizadas en singular: «Sitios» y «sitio» nombran lo mismo."""
    return tuple(_singular(word) for word in normalize_text(text).split())


def _synonym_groups() -> list[list[tuple[str, ...]]]:
    # Los nombres más largos se buscan primero: «conductividad eléctrica» antes que «conductividad».
    return [
        sorted((singular_words(term) for term in group), key=len, reverse=True)
        for group in VARIABLE_SYNONYMS
    ]


def _word_set(texts) -> set[str]:
    return {word for text in texts for word in singular_words(text)}


_SYNONYM_GROUPS = _synonym_groups()
_CONNECTORS = _word_set(MENTION_CONNECTORS)
# Palabras entre un extremo y su fecha: «desde el año 2020», «antes de marzo».
_PERIOD_FILLERS = _CONNECTORS | {"ano", "year"}
_WATER = _word_set(WATER_WORDS)
_OTHER_MEDIUM = _word_set(OTHER_MEDIUM_WORDS)
_SUM = _word_set(SUM_WORDS)
_SECONDARY = _word_set(SECONDARY_QUANTITY_WORDS)


def _content(words) -> tuple[str, ...]:
    """Palabras que precisan una variable: sin conectores ni el agua, el medio por defecto."""
    return tuple(word for word in words if word not in _CONNECTORS and word not in _WATER)


def _variable_terms() -> dict[str, set[tuple[str, ...]]]:
    terms = {name: {_content(singular_words(name))} for name in UNIT_CONFIRMED_SYMBOLS}
    for group in VARIABLE_SYNONYMS:
        terms.setdefault(group[0], set()).update(_content(singular_words(term)) for term in group)

    return terms


_VARIABLE_TERMS = _variable_terms()


def _formula_names(phrase: str) -> str:
    """«NO3 promedio» → «nitrato promedio»."""
    return _FORMULA.sub(lambda match: _FORMULA_NAMES[match.group().casefold()], phrase)


def variable_variants(phrase: str) -> list[str]:
    """La variable y sus nombres equivalentes: «turbidez promedio» → «turbiedad promedio», «turb promedio», …; una
    fórmula química también por su nombre: «NO3» → «nitrato», «nitrate».

    Sólo reemplaza un nombre registrado como equivalente; no traduce ni compara raíces."""
    variants = [phrase, *_synonym_variants(phrase), *_synonym_variants(_formula_names(phrase))]

    return list(dict.fromkeys(variants))


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


def _exact_variable(phrase: str) -> str | None:
    """Variable registrada que la frase nombra sin otra precisión que el agua: «la temperatura del agua» y «NO3»
    nombran temperatura y nitrato; «temperatura del aire» o «temperatura promedio», ninguna."""
    words = _content(singular_words(_formula_names(phrase)))

    return next((name for name, terms in _VARIABLE_TERMS.items() if words in terms), None)


def formula_named(name: str, phrase: str) -> bool:
    """El encabezado nombra con su fórmula el nutriente pedido: «nitratos» o «NO3» → «N-NO3 [mg N L-1]»; nunca a
    «Muestra Nº 3», porque la fórmula se busca tal cual en el nombre original, ni a un cociente («DIN/PO4»)."""
    formulas = CHEMICAL_FORMULAS.get(_exact_variable(phrase), ())

    return any(
        re.search(rf"(?<![A-Za-z\d/:]){re.escape(formula)}(?![a-z\d]|\s*[/:])", name)
        for formula in formulas
    )


def symbol_named(name: str, unit: str | None, phrase: str) -> bool:
    """Un símbolo ambiguo nombra la variable pedida si su unidad documentada lo confirma: «temperatura» → «T (°C)»;
    un «T» sin unidad o en otra unidad, no."""
    variable = _exact_variable(phrase)
    if variable not in UNIT_CONFIRMED_SYMBOLS or not unit:
        return False

    symbols, units = UNIT_CONFIRMED_SYMBOLS[variable]
    return normalize_text(unitless_name(name)) in symbols and unit_key(unit) in units


def without_medium(phrase: str) -> str:
    """La variable sin el medio que la precisa, para buscar sus columnas: «temperatura del aire» → «temperatura».
    Qué medio mide cada columna lo decide other_quantity."""
    words = normalize_text(phrase).split()
    kept = [word for word in words if _singular(word) not in _OTHER_MEDIUM]
    while kept and kept[-1] in _CONNECTORS:
        kept.pop()

    return " ".join(kept) if kept and len(kept) < len(words) else phrase


def qualified_variable(mention: str, question: str) -> str:
    """La variable con el calificativo que la sigue en la pregunta y la vuelve otra magnitud (otro medio, una
    proporción, otra unidad de recuento), que el modelo a veces omite: «¿Y la temperatura del aire?» con la mención
    «temperatura» → «temperatura del aire»."""
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


def qualifies_variable(value: str, variable: str) -> bool:
    """El valor es sólo el calificativo que la variable ya incluye («aire» en «temperatura del aire»): no es un valor
    registrado por el que filtrar."""
    words = set(singular_words(value))
    return bool(words) and words <= (_OTHER_MEDIUM | _SECONDARY) and words <= set(singular_words(variable))


def other_quantity(name: str, phrase: str) -> str | None:
    """El encabezado nombra la variable pero mide otra cosa: otro medio que el pedido («meteorológicos |
    Temperatura» si no se pide el aire, o el agua si se lo pide: «medium») o una suma de magnitudes que no se pide
    («Clorofila "a" + Feofitina "a"», «sum»)."""
    words, asked = set(singular_words(name)), set(singular_words(phrase))

    if bool(words & _OTHER_MEDIUM) != bool(asked & _OTHER_MEDIUM):
        return "medium"
    if DERIVED_SUM.search(name) and not (DERIVED_SUM.search(phrase) or asked & _SUM):
        return "sum"
    return None


def secondary_quantity(name: str, unit: str | None, phrase: str) -> bool:
    """Variante de la variable que no es su valor absoluto y la frase no pide: una proporción o saturación
    («% sat de oxígeno», «Abundancia relativa») o un recuento de colonias o individuos en vez de células."""
    described = f"{name} {unit or ''}"
    marked = PERCENT_SIGN in described or bool(set(singular_words(described)) & _SECONDARY)
    asked = PERCENT_SIGN in phrase or bool(set(singular_words(phrase)) & _SECONDARY)

    return marked and not asked


def variable_core(phrase: str) -> str:
    """Variable sin la estadística que la acompaña: «pH promedio» → «ph», «valor máximo de turbidez» → «turbidez»."""
    words = normalize_text(phrase).split()

    while words and words[0] in STATISTIC_WORDS:
        words.pop(0)
    while words and words[-1] in STATISTIC_WORDS:
        words.pop()

    return " ".join(words)


def variable_name(phrase: str) -> str:
    """La variable como la escribió el usuario, sin su estadística: «pH promedio» → «pH»."""
    words = phrase.split()

    while words and normalize_text(words[0]) in STATISTIC_WORDS:
        words.pop(0)
    while words and normalize_text(words[-1]) in STATISTIC_WORDS:
        words.pop()

    return " ".join(words) or phrase


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


def numeric_comparisons(text: str) -> list[Comparison]:
    """Umbrales literales: «una conductividad promedio mayor que 500» → «promedio mayor que 500» (gt 500).

    El texto empieza en la estadística comparada cuando la precede; un número sin comparación no es un umbral."""
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


def _following_statistic(text: str, start: int) -> int | None:
    """Fin de una estadística que sigue al número, salteando a lo sumo dos palabras (la variable o su unidad):
    «superan los 500 de conductividad en promedio», «mayor que 8 en promedio»."""
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


def period_range(text: str) -> tuple[date | None, date | None] | None:
    """El único intervalo que nombra una fecha pedida (ver period_ranges); None si nombra varios o ninguno."""
    ranges = period_ranges(text)
    return ranges[0] if len(ranges) == 1 else None


def period_ranges(text: str) -> list[tuple[date | None, date | None]]:
    """Intervalos que nombra una fecha pedida: «2021» → todo 2021; «entre 2019 y 2021» o «2019-2021» → de 2019 a
    2021; «marzo de 2022» o «2022-03» → ese mes; «15/03/2021» → ese día; «en 2019 y en 2021» → cada año por separado;
    «marzo y junio de 2022» → cada mes. Un extremo abre el intervalo: «desde 2020» → (2020-01-01, None), «antes de
    2021» → (None, 2020-12-31).

    Sin una fecha legible, con una estación (sus meses cambian con el hemisferio) o una combinación que no se lee sin
    ambigüedad, ninguno: no hay un intervalo que exigir."""
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


def _bounded(anchor, sides) -> list[tuple[date | None, date | None]]:
    start, end = anchor
    lower = start if "from" in sides else end + _DAY if "after" in sides else None
    upper = end if "until" in sides else start - _DAY if "before" in sides else None
    if lower is None and upper is None:
        return [anchor] if not sides else []
    if lower is not None and upper is not None:
        return []  # «desde 2020 hasta 2020» con una sola fecha no se interpreta.

    return [(lower, upper)]


def _period_anchors(text) -> list[tuple[date, date]]:
    """Fechas que nombra el texto, en orden: días explícitos; si no, meses (con su año); si no, años."""
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
        return []  # «marzo de 2019 y abril de 2021»: los meses no se atribuyen a cada año sin ambigüedad.

    if explicit:
        return [
            (date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))
            for year, month in explicit
        ]
    return [(date(year, 1, 1), date(year, 12, 31)) for year in years]


def period_text(start: date, end: date) -> str:
    """Nombre de un período cerrado: 2019, marzo de 2022 o 15/03/2021."""
    if (start.month, start.day, end.month, end.day) == (1, 1, 12, 31) and start.year == end.year:
        return str(start.year)
    if start.day == 1 and end == date(start.year, start.month, calendar.monthrange(start.year, start.month)[1]):
        return f"{_SPANISH_MONTHS[start.month - 1]} de {start.year}"
    if start == end:
        return start.strftime("%d/%m/%Y")
    return f"{start.strftime('%d/%m/%Y')} al {end.strftime('%d/%m/%Y')}"


_SPANISH_MONTHS = list(MONTH_NAMES)[:12]


def qualified_period(mention: str, question: str) -> str:
    """La fecha con el extremo o el rango que la precede en la pregunta y el modelo omitió: con la mención «2020»,
    «¿El pH desde el año 2020?» → «desde el ano 2020»; sin él, «2020» se exigiría como el año entero."""
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


def _original_words(question: str, mention: str, count: int) -> list[str]:
    """Las últimas palabras antes de la mención, como las escribió el usuario («después de», «año»)."""
    position = question.casefold().find(mention.casefold())
    words = [word.strip(SURROUNDING_PUNCTUATION) for word in question[:position].split()]
    words = [word for word in words if word]
    return words[-count:] if position >= 0 and len(words) >= count else []


def _explicit_month(match) -> int | None:
    """AAAA-MM o MM/AAAA; un mes inexistente («2020-21») no cuenta."""
    month = int(match.group("month") or match.group("m"))
    return month if 1 <= month <= 12 else None


def _explicit_day(match) -> date | None:
    """AAAA-MM-DD o, como se escribe en castellano, DD/MM/AAAA."""
    try:
        if match.group("iso"):
            return date(*(int(part) for part in match.group("iso").split("-")))
        day, month, year = (int(part) for part in match.group("dmy").split("/"))
        return date(year, month, day)
    except ValueError:
        return None


def _comparison_number(token: str) -> int | float | None:
    """«1.000» y «1,000» son miles; «7,5» y «7.5», decimales."""
    unsigned = token.lstrip("-")
    sign = -1 if token.startswith("-") else 1

    if INTEGER.fullmatch(unsigned) and not unsigned.startswith("0"):
        return sign * int(_THOUSANDS_SEPARATORS.sub("", unsigned))

    try:
        return sign * float(unsigned.replace(",", "."))
    except ValueError:
        return None


def _compared_statistic(text: str, end: int) -> int | None:
    """Inicio de la estadística comparada, salteando a lo sumo una palabra (la variable) y artículos."""
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


def starts_with_phrase(text: str, phrases) -> bool:
    """El texto empieza con alguna de las frases normalizadas: «¿Y la turbiedad?» empieza con «y»."""
    words = normalize_text(text).split()

    return any(words[: len(phrase.split())] == phrase.split() for phrase in phrases if phrase)


def contains_any_phrase(text: str, phrases) -> bool:
    return any(contains_phrase(text, phrase) for phrase in phrases)
