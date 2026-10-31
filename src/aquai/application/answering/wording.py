import re
from datetime import date, datetime
from decimal import Decimal

from aquai.application.answering.presentation import format_number, format_unit
from aquai.application.planning.temporalText import period_bounds
from aquai.application.Rules.vocabulary import (
    NUMBER_WORDS,
    PLACE_NAME_LOWERCASE_WORDS,
    PLACE_WORDS,
    SPREADSHEET_EXTENSION,
)
from aquai.application.units import is_ph_name, presentation_unit_status

NUMERIC_TYPES = {"NUMERIC", "BIGINT"}
SPREADSHEET_TYPES = {"xlsx", "xlsm", "xls"}
# Un registro con más variables se lee mejor en una tabla que en una oración.
INLINE_FIELDS = 4

CELL_RANGE = re.compile(r"^\$?([A-Z]{1,3})\$?(\d+):\$?([A-Z]{1,3})\$?(\d+)$")
_MONTHS = {
    "es": (
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ),
    "en": (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ),
}
_GRAINS = {
    "es": {"year": "año", "month": "mes", "day": "día"},
    "en": {"year": "year", "month": "month", "day": "day"},
}
# Sustantivo y género gramatical de cada cálculo; {0} y {1} son las variables originales.
_CALCULATIONS = {
    "es": {
        "count": ("cantidad de valores de {0}", True),
        "count_distinct": ("cantidad de valores distintos de {0}", True),
        "sum": ("suma de {0}", True),
        "mean": ("promedio de {0}", False),
        "min": ("valor mínimo de {0}", False),
        "max": ("valor máximo de {0}", False),
        "median": ("mediana de {0}", True),
        "stddev": ("desvío estándar {sample} de {0}", False),
        "quantile": ("percentil {percentile} de {0}", False),
        "difference": ("diferencia entre {0} y {1}", True),
        "ratio": ("cociente entre {0} y {1}", False),
        "correlation": ("correlación de Pearson entre {0} y {1}", True),
    },
    "en": {
        "count": ("number of {0} values", False),
        "count_distinct": ("number of distinct {0} values", False),
        "sum": ("sum of {0}", False),
        "mean": ("mean {0}", False),
        "min": ("minimum {0}", False),
        "max": ("maximum {0}", False),
        "median": ("median {0}", False),
        "stddev": ("{sample} standard deviation of {0}", False),
        "quantile": ("{percentile} percentile of {0}", False),
        "difference": ("difference between {0} and {1}", False),
        "ratio": ("ratio between {0} and {1}", False),
        "correlation": ("Pearson correlation between {0} and {1}", False),
    },
}


# Definiciones generales de cada estadística: explican el cálculo, nunca interpretan los datos.
_MEANINGS = {
    "es": {
        "mean": "El promedio es la suma de los valores dividida por su cantidad, por lo que unos pocos valores muy "
        "altos o muy bajos pueden moverlo bastante.",
        "median": "La mediana es el valor central: la mitad de los valores queda por debajo y la otra mitad por encima.",
        "stddev": "El desvío estándar indica cuánto se alejan, en general, los valores de su promedio: cuanto mayor es, "
        "más dispersos están los datos.",
        "quantile": "El percentil {percentile} es el valor por debajo del cual queda el {share} % de los valores.",
        "correlation": "La correlación de Pearson va de -1 a 1: cerca de 1, las dos variables tienden a aumentar juntas; "
        "cerca de -1, una aumenta cuando la otra disminuye; cerca de 0, no hay una relación lineal clara. "
        "No indica que una variable cause la otra.",
        "count_distinct": "Cada valor distinto se cuenta una sola vez, aunque aparezca en varios registros.",
    },
    "en": {
        "mean": "The mean is the sum of the values divided by how many there are, so a few very high or very low "
        "values can move it considerably.",
        "median": "The median is the middle value: half of the values are below it and half above.",
        "stddev": "The standard deviation shows how far values typically are from their mean: the larger it is, "
        "the more spread out the data.",
        "quantile": "The {percentile} percentile is the value below which {share}% of the values fall.",
        "correlation": "The Pearson correlation ranges from -1 to 1: near 1, both variables tend to increase together; "
        "near -1, one increases as the other decreases; near 0, there is no clear linear relationship. "
        "It does not mean that one variable causes the other.",
        "count_distinct": "Each distinct value is counted once, even if it appears in several records.",
    },
}
_ORDINAL_SUFFIXES = {"1": "st", "2": "nd", "3": "rd"}


def sentence(text: str) -> str:
    """Mayúscula inicial, salvo símbolos con mayúsculas internas como pH o mL."""
    first = text.split(" ", 1)[0]
    if any(character.isupper() for character in first[1:]):
        return text

    return text[:1].upper() + text[1:]


def _euphonic(word: str, following: str) -> str:
    """«y» pasa a «e» ante el sonido i (pH e índice) y «o» a «u» ante el sonido o."""
    start = following.lower()
    starts_with_i = start.startswith(("i", "í", "hi", "hí"))
    # «hie» y «hia» suenan como ye y ya: agua y hielo.
    sounds_like_y = start.startswith(("hie", "hia"))
    if word == "y" and starts_with_i and not sounds_like_y:
        return "e"

    sounds_like_o = start.startswith(("o", "ó", "ho", "hó"))
    if word == "o" and sounds_like_o:
        return "u"

    return word


def join_words(items, language="es", conjunction=None) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)

    word = conjunction or ("y" if language == "es" else "and")
    if language == "es":
        word = _euphonic(word, items[-1])

    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def number_words(count: int, language="es") -> str:
    words = NUMBER_WORDS[language]
    return words[count] if 0 <= count < len(words) else format_number(count, language)


def count_text(count: int, singular: str, plural: str, language="es") -> str:
    return f"{format_number(count, language)} {singular if count == 1 else plural}"


def humanize(name: str) -> str:
    return sentence(" ".join(name.replace("_", " ").split()))


def file_stem(name: str) -> str:
    return SPREADSHEET_EXTENSION.sub("", name.strip())


def dataset_label(name: str) -> str:
    parts = [part.strip() for part in name.split(" / ")]
    if len(parts) >= 2 and CELL_RANGE.match(parts[-1]):
        parts = parts[:-1]

    return " / ".join(parts[1:] if len(parts) >= 2 else parts)


def column_label(original_name: str, unit: str | None = None) -> str:
    label = " ".join(str(original_name).split())
    if is_ph_name(label):
        return "pH"
    if not unit:
        return label

    for suffix in (f" [{unit}]", f" ({unit})"):
        if label.endswith(suffix):
            return label[: -len(suffix)]

    return label


def display_unit(field, language="es") -> tuple[str | None, bool]:
    """Nunca se completa una unidad faltante."""
    status = presentation_unit_status(field)
    if status == "documented" and field.unit:
        return format_unit(field.unit), True
    if status == "unknown" and field.sql_type in NUMERIC_TYPES:
        return ("unidad no documentada" if language == "es" else "unit not documented"), False

    return None, False


def unit_suffix(field, language="es") -> str:
    unit, documented = display_unit(field, language)
    if unit is None:
        return ""

    return f" {unit}" if documented else f" ({unit})"


def _share(quantile, language):
    """Probabilidad exacta del plan en porcentaje (0,12345 → 12,345), sin redondeo de presentación."""
    value = format(Decimal(str(quantile)) * 100, "f")
    if "." in value:
        value = value.rstrip("0").rstrip(".")

    return value.replace(".", ",") if language == "es" else value


def _percentile(quantile, language):
    value = _share(quantile, language)
    if language == "es":
        return value

    return value + _ordinal_suffix(value)


def _ordinal_suffix(number):
    if number.endswith(("11", "12", "13")) or "." in number:
        return "th"

    return _ORDINAL_SUFFIXES.get(number[-1], "th")


def calculation_noun(
    function, variables, language="es", *, quantile=None, ddof=1
) -> tuple[str, bool]:
    if function == "count" and not variables:
        return ("cantidad de registros", True) if language == "es" else ("number of records", False)
    if function == "mean" and list(variables) == ["pH"]:
        return ("pH promedio" if language == "es" else "mean pH"), False

    template, feminine = _CALCULATIONS[language].get(function, ("{0}", False))
    # Los nombres de columnas se citan tal cual: pueden ser especies (Microcystis aeruginosa) o siglas.
    operands = _operands(variables, language)
    sample = _sample_label(ddof, language)
    percentile = _percentile(quantile, language) if quantile is not None else ""

    return template.format(*operands, sample=sample, percentile=percentile), feminine


def _operands(variables, language):
    """Las plantillas de dos variables repiten la única que haya."""
    operands = list(variables) or ["valores" if language == "es" else "values"]

    return operands if len(operands) > 1 else operands * 2


def _sample_label(ddof, language):
    if language == "es":
        return "muestral" if ddof == 1 else "poblacional"

    return "sample" if ddof == 1 else "population"


def calculation_meaning(function, language="es", *, quantile=None) -> str | None:
    template = _MEANINGS[language].get(function)
    if template is None:
        return None
    if quantile is None:
        return template.format(percentile="", share="")

    return template.format(
        percentile=_percentile(quantile, language), share=_share(quantile, language)
    )


def with_article(noun: str, feminine: bool, language="es") -> str:
    if language != "es":
        return "the " + noun

    return ("la " if feminine else "el ") + noun


def grain_label(time_grain: str, language="es") -> str:
    return _GRAINS[language].get(time_grain, time_grain)


def is_place(label: str) -> bool:
    return label.casefold() in PLACE_WORDS


def place_text(value) -> str:
    """Nombre de un lugar tal como está registrado; sólo las mayúsculas sostenidas se suavizan (PUNTA LARA → Punta Lara)."""
    words = str(value).split()
    shouted = bool(words) and all(word.isalpha() for word in words) and " ".join(words).isupper()
    if shouted:
        words = [_place_word(word, index) for index, word in enumerate(words)]

    return " ".join(words)


def _place_word(word, index):
    """Los conectores internos (de, la, del) van en minúscula; el resto, capitalizado."""
    if index and word.lower() in PLACE_NAME_LOWERCASE_WORDS:
        return word.lower()

    return word.capitalize()


def date_text(value: date, language="es") -> str:
    if language != "es":
        return value.isoformat()

    text = value.strftime("%d/%m/%Y")
    has_time = isinstance(value, datetime) and any(
        (value.hour, value.minute, value.second, value.microsecond)
    )
    if not has_time:
        return text

    with_seconds = value.second or value.microsecond

    return text + " " + value.strftime("%H:%M:%S" if with_seconds else "%H:%M")


def _month(day: date, language: str, *, year=True) -> str:
    name = _MONTHS[language][day.month - 1]
    if not year:
        return name

    return f"{name} de {day.year}" if language == "es" else f"{name} {day.year}"


def period(value, language="es", day_first=None) -> tuple[str, str]:
    """Fecha tipada o período de texto verificado, con su precisión; no completa días ni años."""
    if isinstance(value, date):
        return date_text(value, language), "day"

    bounds = period_bounds(value, day_first)
    if bounds is None:
        return str(value), "period"

    start, end, precision = bounds
    if precision == "day":
        return date_text(start, language), "day"
    if precision == "year":
        return str(start.year), "year"
    if precision == "month":
        return _month(start, language), "month"

    return _month_range(start, end, language), "months"


def _month_range(start, end, language):
    """De enero a marzo de 2008; el año inicial sólo se escribe si difiere del final."""
    joiner = " a " if language == "es" else " to "
    first = _month(start, language, year=start.year != end.year)

    return first + joiner + _month(end, language)


def grain(value, time_grain: str, language="es") -> tuple[str, str]:
    """Clave de una agregación temporal: el año 2008, marzo de 2008 o un día."""
    if not isinstance(value, date):
        return str(value), "year" if time_grain == "year" else "period"

    if time_grain == "year":
        return str(value.year), "year"
    if time_grain == "month":
        return _month(value, language), "month"

    return date_text(value, language), "day"


def at(text: str, precision: str, language="es") -> str:
    """Complemento temporal: el 12/03/2008, en marzo de 2008, en el período de enero a marzo de 2008."""
    if language == "es":
        return {"day": f"el {text}", "months": f"en el período de {text}"}.get(
            precision, f"en {text}"
        )

    return {"day": f"on {text}", "months": f"in the period {text}"}.get(precision, f"in {text}")


def dated(text: str, precision: str, language="es") -> str:
    """Origen temporal: del 12/03/2008, de marzo de 2008."""
    if language == "es":
        return f"del {text}" if precision == "day" else f"de {text}"

    return f"from {text}"
