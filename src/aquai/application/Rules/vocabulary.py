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
        r"rebas\w*",
        r"pas(?:a|an|ó|aron|e|en|ara|aran) (?:de|los|las|el|la)",
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
        r"no (?:lleg|alcanz)\w* (?:a|los|las)",
        r"less than",
        r"lower than",
        r"fewer than",
        r"below",
        r"under",
    ),
}
COMPARISON_ARTICLES = r"(?:los|las|el|la|un|una|the)"
# Comparaciones que se escriben después del número: «20.000 cél/mL o más», «25 grados o menos», «100 como mínimo».
POSTFIX_COMPARISON_CUES = {
    "ge": (
        r"o m[aá]s",
        r"o mayor(?:es)?",
        r"o superior(?:es)?",
        r"como m[ií]nimo",
        r"or more",
        r"or (?:higher|greater|above)",
    ),
    "le": (
        r"o menos",
        r"o menor(?:es)?",
        r"o inferior(?:es)?",
        r"como m[aá]ximo",
        r"or less",
        r"or (?:lower|fewer|below)",
    ),
}
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
# Un «cuántos» seguido de «por» y una unidad pide una concentración medida, no un conteo de registros
# («¿cuántas cianobacterias por mililitro?», «how many cells per mL»); «cuántas mediciones por sitio» sí cuenta.
COUNT_CUE = re.compile(r"\b(?:cu[aá]nt[oa]s|how\s+many)\b", re.IGNORECASE)
RATE_COUNT = re.compile(
    r"\b(?:cu[aá]nt[oa]s|how\s+many)\b(?:\s+[^\s,.;:?¿!]+){0,6}?\s+(?:por|per)\s+(?:cada\s+)?(?:mililitros?|"
    r"microlitros?|litros?|ml|l|m3|metros?\s+c[uú]bicos?|cm2|cm3|gramos?|g|kg|unidad(?:es)?\s+de\s+volumen)\b",
    re.IGNORECASE,
)

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
# Abreviaturas de meses; sólo cuentan escritas entre un día y un año («22 sep 2021»), porque «mar» o «may» son
# también palabras comunes.
MONTH_ABBREVIATIONS = {
    "ene": 1, "jan": 1, "feb": 2, "mar": 3, "abr": 4, "apr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "aug": 8,
    "sep": 9, "sept": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12, "dec": 12,
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
# Pedidos de las fechas que cubren los datos, por familias de expresiones: el período («indicá promedios y
# períodos»), qué fechas abarcan, entre o desde qué fechas, primera y última fecha, cuándo empiezan o terminan, el
# rango o el lapso. No cuentan un período dado como filtro («en el período 2020-2021») ni una agrupación («por
# período»); una mención de fecha sin fecha («qué fechas abarcan») también lo pide (planning.context).
PERIOD_SPAN_CUES = re.compile(
    r"(?<!por )\b(?:per[ií]odos?(?!\s+(?:de\s+muestreo\s+)?(?:(?:de|del|entre)\s+)?(?:\d|enero|febrero|marzo|"
    r"abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre))|"
    r"rangos?\s+(?:de\s+fechas|temporal(?:es)?)|lapsos?(?:\s+de\s+tiempo)?|intervalos?\s+de\s+fechas|"
    r"cobertura\s+temporal|alcance\s+temporal|"
    r"(?:entre|de|desde|hasta)\s+qu[eé]\s+fechas?|"
    r"qu[eé]\s+fechas?\s+(?:abarca\w*|cubre\w*|comprende\w*|incluye\w*|tiene\w*|hay|corresponde\w*|son|es)|"
    r"(?:primeras?|[uú]ltimas?|iniciales?|finales?)\s+(?:y\s+(?:la\s+)?(?:primeras?|[uú]ltimas?|finales?)\s+)?fechas?|"
    r"fechas?\s+(?:de\s+(?:inicio|comienzo|fin|cierre)|iniciales?|finales?|extremas?)|"
    r"desde\s+cu[aá]ndo|hasta\s+cu[aá]ndo|"
    r"cu[aá]ndo\s+(?:\w+\s+){0,3}?(?:empie\w*|empez\w*|comien\w*|comenz\w*|termin\w*|finaliz\w*|arranc\w*)|"
    r"periods?\s+covered|date\s+ranges?|time\s+spans?|(?:what|which)\s+dates|"
    r"(?:first|earliest)\s+and\s+(?:last|latest)\s+dates?)\b",
    re.IGNORECASE,
)
# Preguntas por un cambio a lo largo del tiempo, como composición de un verbo de cambio y una referencia
# temporal en cualquier orden («fue cambiando con los años», «aumentó con el tiempo», «over the years… increased»),
# además de los nombres del cambio («tendencia», «evolución temporal», «variación interanual»). Los adjetivos
# («la turbidez más baja en los años 2012») no son cambios.
_CHANGE_WORDS = (
    r"(?:cambi(?:a|an|ó|aron|ando|ado|ar|os)|vari(?:a|an|ó|aron|ando|ado|ar|aci[oó]n|aciones)|evolucion\w*|"
    r"aument(?:a|an|ó|aron|ando|ado|ar|o|os)|disminu(?:ye|yen|yó|yeron|yendo|ido|ir|ci[oó]n)|descend\w*|"
    r"baj(?:ó|aron|ando|ado)|sub(?:ió|ieron|iendo|ido)|crec(?:e|en|ió|ieron|iendo|ido)|decrec\w*|empeor\w*|"
    r"mejor(?:ó|aron|ando|ado)|increas\w*|decreas\w*|chang\w*|ris(?:e|es|ing|en)|rose|declin\w*|grow\w*|grew)"
)
_TIME_SPAN_WORDS = (
    r"(?:(?:con|a\s+lo\s+largo\s+de|en|durante|con\s+el\s+paso\s+de|through|over|across)\s+"
    r"(?:(?:los|las|el|la|del|the)\s+)?(?:(?:[uú]ltim[oa]s|primer[oa]s|recent|past|last)\s+)?"
    r"(?:a[nñ]os|tiempo|d[eé]cadas?|time|years|decades?|campa[nñ]as))"
)
TREND_CUES = re.compile(
    rf"\b(?:tendencias?|trends?|evoluci[oó]n\s+(?:temporal|en\s+el\s+tiempo)|variaci[oó]n\s+(?:temporal|interanual)|"
    rf"{_CHANGE_WORDS}(?:\s+\S+){{0,5}}?\s+{_TIME_SPAN_WORDS}|{_TIME_SPAN_WORDS}(?:\s+\S+){{0,4}}?\s+{_CHANGE_WORDS})\b",
    re.IGNORECASE,
)
