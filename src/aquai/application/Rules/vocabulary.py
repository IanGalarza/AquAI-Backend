import re

STATISTIC_WORDS = {
    "promedio",
    "promedios",
    "medio",
    "media",
    "medias",
    "mediana",
    "maximo",
    "maximos",
    "maxima",
    "maximas",
    "minimo",
    "minimos",
    "minima",
    "minimas",
    "suma",
    "cantidad",
    "conteo",
    "desvio",
    "estandar",
    "valor",
    "valores",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "mean",
    "average",
    "median",
    "maximum",
    "minimum",
    "max",
    "min",
    "sum",
    "count",
    "value",
    "values",
    "of",
    "the",
}

NUMBER_WORDS = {
    "es": (
        "cero",
        "uno",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
        "diez",
    ),
    "en": ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
}

INTEGER = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+")
SURROUNDING_PUNCTUATION = "¿?¡!.,;:()[]«»\"'"

SPREADSHEET_EXTENSION = re.compile(r"\.(?:csv|xlsx|xlsm|xls)$", re.IGNORECASE)
UPLOADED_FILE_EXTENSION = re.compile(r"\.(?:csv|xlsx)\b", re.IGNORECASE)

SOURCE_CUE_WORDS = (
    "archivo",
    "fuente",
    "dataset",
    "conjunto de datos",
    "hoja",
    "tabla",
    "planilla",
    "file",
    "source",
    "workbook",
    "sheet",
    "table",
)
SOURCE_NAME_CONNECTORS = (
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "llamad[oa]",
    "denominad[oa]",
    "named",
    "called",
)
SOURCE_CUE_WINDOW = 100
SOURCE_ALIAS_CUES = ("datos", "base", "archivo", "planilla", "fuente", "data", "file")

OBSERVATION_NOUNS_ES = (
    r"(?:datos?|medici[oó]n|mediciones|registros?|observaci[oó]n|observaciones|valor(?:es)?|"
    r"lecturas?|muestras?|fechas?)"
)
OBSERVATION_NOUNS_EN = (
    r"(?:measurements?|observations?|records?|readings?|values?|data(?:\s+points?)?)"
)
RECENT_WORDS = r"(?:[uú]ltim[oa]s?|latest|last|current)"
RECENT_INTERVAL_NOUNS = (
    r"(?:a[nñ]os?|mes(?:es)?|semanas?|d[ií]as?|per[ií]odos?|d[eé]cadas?|trimestres?|semestres?|"
    r"horas?|campa[nñ]as?|years?|months?|weeks?|days?|periods?|decades?|hours?)"
)

PLACE_WORDS = {
    "localidad",
    "localidades",
    "ciudad",
    "ciudades",
    "municipio",
    "partido",
    "provincia",
    "sitio",
    "sitios",
    "estación",
    "estacion",
    "estaciones",
    "lugar",
    "lugares",
    "punto",
    "puntos",
    "zona",
    "zonas",
    "site",
    "sites",
    "station",
    "stations",
    "locality",
    "city",
    "location",
    "place",
}
PLACE_NAME_LOWERCASE_WORDS = {"de", "del", "la", "las", "los", "y", "el"}

COMPARISON_CUES = {
    "ge": (
        r"mayor(?:es)? o igual(?:es)? (?:que|a)",
        r"al menos",
        r"como m[ií]nimo",
        r"no menos de",
        r"at least",
        r"greater than or equal to",
    ),
    "le": (
        r"menor(?:es)? o igual(?:es)? (?:que|a)",
        r"como m[aá]ximo",
        r"no m[aá]s de",
        r"at most",
        r"less than or equal to",
    ),
    "gt": (
        r"mayor(?:es)? (?:que|a)",
        r"m[aá]s (?:de|que)",
        r"superior(?:es)? a",
        r"super[aeoáó]\w*",
        r"sobrepas\w*",
        r"exced\w*",
        r"por encima del?",
        r"greater than",
        r"more than",
        r"higher than",
        r"above",
        r"over",
        r"exceed\w*",
    ),
    "lt": (
        r"menor(?:es)? (?:que|a)",
        r"menos (?:de|que)",
        r"inferior(?:es)? a",
        r"por debajo del?",
        r"less than",
        r"lower than",
        r"fewer than",
        r"below",
        r"under",
    ),
}
COMPARISON_ARTICLES = r"(?:los|las|el|la|un|una|the)"
COMPARISON_REFERENCE = (
    r"(?:(?:el|la|los|las|the|a)\s+)?(?:(?:guideline|reference|alert)\s+)?"
    r"(?:nivel(?:es)?|valor(?:es)?|umbral(?:es)?|l[ií]mites?|levels?|values?|"
    r"thresholds?|limits?)(?:\s+(?:gu[ií]a|de\s+referencia|de\s+alerta|guideline|guidance|reference))?\s+(?:de|of)"
)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
COMPARED_STATISTICS = {
    "promedio",
    "promedios",
    "media",
    "medias",
    "medio",
    "medios",
    "mediana",
    "medianas",
    "maximo",
    "maximos",
    "maxima",
    "maximas",
    "minimo",
    "minimos",
    "minima",
    "minimas",
    "suma",
    "average",
    "mean",
    "median",
    "maximum",
    "minimum",
    "sum",
}
COMPARISON_FILLERS = {
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "es",
    "fue",
    "sea",
    "of",
    "the",
    "is",
    "was",
}
FOLLOWING_STATISTIC_FILLERS = {"en", "como", "por", "on", "as"}
COMPARISON_BOUNDARIES = {
    "con",
    "cuyo",
    "cuya",
    "cuyos",
    "cuyas",
    "donde",
    "y",
    "o",
    "with",
    "where",
    "whose",
    "and",
    "or",
}

MENTION_CONNECTORS = {
    "en",
    "de",
    "del",
    "para",
    "con",
    "por",
    "cada",
    "que",
    "cual",
    "cuales",
    "la",
    "el",
    "los",
    "las",
    "in",
    "at",
    "of",
    "for",
    "with",
    "by",
    "per",
    "each",
    "which",
}

CATEGORY_CUES = {"cada", "que", "cual", "cuales", "por", "each", "which", "per"}

GROUPING_CUES = ("por", "per", "by")

NON_GROUPING_WORDS = {
    "el",
    "la",
    "los",
    "las",
    "lo",
    "un",
    "una",
    "the",
    "a",
    "an",
    "debajo",
    "encima",
    "ciento",
    "cent",
    "favor",
    "ejemplo",
    "que",
}

REQUESTED_STATISTICS = {
    "mean": ("promedio", "promedios", "prom", "media", "medias", "average", "mean"),
    "median": ("mediana", "median"),
    "sum": ("suma", "sum"),
    "stddev": ("desvio estandar", "desviacion estandar", "standard deviation"),
    "count": ("cuantos", "cuantas", "how many"),
}

CHART_WORDS = (
    r"(?:gr[aá]fic\w*|plot\w*|chart\w*|diagrama\w*|histograma\w*|visualiz\w*|dibuj\w*|"
    r"barras|l[ií]neas|dispersi[oó]n|distribuci[oó]n|scatter)"
)

CONTINUATION_CUES = (
    "y",
    "e",
    "ahora",
    "tambien",
    "lo mismo",
    "la misma",
    "and",
    "now",
    "also",
    "the same",
)

ADDITIVE_CUES = (
    "tambien",
    "ademas",
    "compara",
    "comparalo",
    "comparar",
    "comparacion",
    "junto",
    "versus",
    "vs",
    "also",
    "compare",
)

TIME_GRAIN_CUES = {
    "month": (
        "por mes",
        "por meses",
        "cada mes",
        "mensual",
        "mensuales",
        "mes a mes",
        "per month",
        "monthly",
        "by month",
    ),
    "year": (
        "por ano",
        "por anos",
        "cada ano",
        "anual",
        "anuales",
        "per year",
        "yearly",
        "annual",
        "by year",
    ),
    "day": ("por dia", "por dias", "cada dia", "diario", "diaria", "per day", "daily", "by day"),
}

MONTH_NAMES = {
    **{
        name: index
        for index, name in enumerate(
            (
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
            1,
        )
    },
    "setiembre": 9,
    **{
        name: index
        for index, name in enumerate(
            (
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ),
            1,
        )
    },
}
REQUESTED_YEAR = re.compile(r"\b(1[89]\d{2}|2\d{3})\b")
PERIOD_BOUND_CUES = {
    "from": ("desde", "a partir", "since", "from"),
    "after": ("despues", "posterior", "posteriores", "after"),
    "until": ("hasta", "until", "through"),
    "before": ("antes", "anterior", "anteriores", "before", "prior to"),
}
RANGE_CUES = ("entre", "between")
OPEN_END_WORDS = ("hoy", "ahora", "la actualidad", "la fecha", "today", "now", "the present")
RANGE_CONNECTORS = r"(?:-|–|\ba\b|\bal\b|\bhasta\b|\bto\b|\buntil\b|\bthrough\b)"
SEASON_WORDS = (
    "verano",
    "otono",
    "invierno",
    "primavera",
    "summer",
    "autumn",
    "fall",
    "winter",
    "spring",
)
YEAR_COLUMN_NAMES = {("ano",), ("anio",), ("year",)}

PERIOD_QUESTION_CUES = {
    "year": ("que ano", "que anos", "cual ano", "what year", "which year"),
    "month": ("que mes", "que meses", "cual mes", "what month", "which month"),
    "day": (
        "que fecha",
        "que fechas",
        "que dia",
        "cual fecha",
        "cuando",
        "what date",
        "which date",
        "when",
    ),
}

CHART_TYPE_WORDS = {
    "bar": ("barra", "barras", "bar", "bars"),
    "line": ("linea", "lineas", "line", "lines", "evolucion"),
    "scatter": ("dispersion", "scatter"),
    "histogram": ("histograma", "histogram"),
}

HIGHEST_CUES = (
    "max",
    "maximo",
    "maxima",
    "mas alto",
    "mas alta",
    "mas elevado",
    "mas elevada",
    "mayor valor",
    "highest",
    "largest",
    "greatest",
    "biggest",
    "maximum",
)
LOWEST_CUES = (
    "min",
    "minimo",
    "minima",
    "mas bajo",
    "mas baja",
    "menor valor",
    "lowest",
    "smallest",
    "minimum",
)
SUPERLATIVE_WORDS = r"\b(mayor|menor)(?:es)?\b(?!\s+(?:a|al|que|o|parte|de\s+\d+)\b)"
COUNT_BOUNDS = r"\b(?:un|como|a)\s+(?:maximo|minimo|maximum|minimum)\s+(?:de\s+|of\s+)?\d+"
FULL_ORDER_CUES = (
    "de mayor a menor",
    "de menor a mayor",
    "ordenados",
    "ordenadas",
    "ordena",
    "ordenalos",
    "from highest to lowest",
    "from lowest to highest",
    "sorted",
)
DESCENDING_ORDER_CUES = (
    "de mayor a menor",
    "del mayor al menor",
    "de mas alto a mas bajo",
    "del mas alto al mas bajo",
    "de la mas alta a la mas baja",
    "de los mas altos a los mas bajos",
    "de las mas altas a las mas bajas",
    "descendente",
    "decreciente",
    "from highest to lowest",
    "highest to lowest",
    "descending",
)
ASCENDING_ORDER_CUES = (
    "de menor a mayor",
    "del menor al mayor",
    "de mas bajo a mas alto",
    "del mas bajo al mas alto",
    "de la mas baja a la mas alta",
    "de los mas bajos a los mas altos",
    "de las mas bajas a las mas altas",
    "ascendente",
    "creciente",
    "from lowest to highest",
    "lowest to highest",
    "ascending",
)

MEASURED_STATISTICS = (
    "promedio",
    "prom",
    "max",
    "min",
    "media",
    "mediana",
    "maximo",
    "maxima",
    "minimo",
    "minima",
    "desvio estandar",
    "average",
    "mean",
    "median",
    "maximum",
    "minimum",
)

FULL_ORDER_CUES = (*FULL_ORDER_CUES, *DESCENDING_ORDER_CUES, *ASCENDING_ORDER_CUES)
ORDER_KEY = r"\b(?:orden\w*|sort\w*)\b[^.?!]*?\b(?:por|by)\s+(?:la\s+|el\s+|the\s+)?(?P<key>[^\W\d_]+)"
INNER_STATISTIC_WORDS = frozenset(
    {"max", "min", "maximo", "maxima", "minimo", "minima", "promedio", "prom", "media", "mediana"}
)
