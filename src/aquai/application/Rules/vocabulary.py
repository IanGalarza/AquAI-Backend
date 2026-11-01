"""Vocabulario con el que se leen preguntas, nombres de fuentes y columnas."""

import re

# Estadística y conectores que acompañan a una variable: «pH promedio», «valor máximo de pH».
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

# Entero con separadores de miles opcionales: 5, 1000, 1.000, 1,000.
INTEGER = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+")
SURROUNDING_PUNCTUATION = "¿?¡!.,;:()[]«»\"'"

SPREADSHEET_EXTENSION = re.compile(r"\.(?:csv|xlsx|xlsm|xls)$", re.IGNORECASE)
UPLOADED_FILE_EXTENSION = re.compile(r"\.(?:csv|xlsx)\b", re.IGNORECASE)

# Una mención es un nombre de fuente si la precede alguna de estas palabras: «el archivo llamado …».
# SOURCE_CUE_WORDS y SOURCE_NAME_CONNECTORS son fragmentos de regex unidos sin escapar («llamad[oa]»).
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
# Palabras tras las que un nombre corto puede abreviar una fuente («los datos de ABSA», «la base FQ»); sólo
# cuentan si esa palabra está en el nombre de una única fuente y no es un valor de los registros.
SOURCE_ALIAS_CUES = ("datos", "base", "archivo", "planilla", "fuente", "data", "file")

# Observaciones puntuales: «la última medición» pide un dato, «el último año» pide un intervalo.
# OBSERVATION_NOUNS_ES/EN son fragmentos de regex que se insertan sin escapar («medici[oó]n»).
OBSERVATION_NOUNS_ES = (
    r"(?:datos?|medici[oó]n|mediciones|registros?|observaci[oó]n|observaciones|valor(?:es)?|"
    r"lecturas?|muestras?|fechas?)"
)
OBSERVATION_NOUNS_EN = (
    r"(?:measurements?|observations?|records?|readings?|values?|data(?:\s+points?)?)"
)
# «Último» seguido de una variable («el último pH») también pide la medición más reciente; seguido de un
# período o de una cantidad («el último año», «los últimos 5») pide un intervalo o varias filas.
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
# Palabras que quedan en minúscula al suavizar un nombre en mayúsculas: PUNTA DE LARA → Punta de Lara.
PLACE_NAME_LOWERCASE_WORDS = {"de", "del", "la", "las", "los", "y", "el"}

# Comparaciones numéricas: «mayor que 500» acota con gt. Fragmentos de regex unidos sin escapar;
# dentro de la alternancia, «mayor o igual que» (ge) se prueba antes que «mayor que» (gt).
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
        r"por encima de",
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
        r"por debajo de",
        r"less than",
        r"lower than",
        r"fewer than",
        r"below",
        r"under",
    ),
}
# Artículos entre la comparación y el número: «superó los 100».
COMPARISON_ARTICLES = r"(?:los|las|el|la|un|una|the)"
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
# Una estadística hasta dos palabras antes de la comparación es lo comparado: «un promedio de pH superior a 8».
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
# Conectores entre el número y una estadística que lo sigue: «mayor que 8 en promedio», «500 como media».
FOLLOWING_STATISTIC_FILLERS = {"en", "como", "por", "on", "as"}
# Lo anterior a estas palabras ya no es lo comparado: «el pH promedio de las muestras con turbiedad mayor que 100».
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

# Conectores que quedan en el borde de una mención: «pH promedio en», «por sitio», «cada estación».
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

# Una mención que empieza así pregunta por una categoría («cada punto de muestreo»), no nombra un valor.
CATEGORY_CUES = {"cada", "que", "cual", "cuales", "por", "each", "which", "per"}

# Estadísticas que una pregunta pide por su nombre: «pH promedio» exige calcular mean. Frases normalizadas.
REQUESTED_STATISTICS = {
    "mean": ("promedio", "promedios", "media", "medias", "average", "mean"),
    "median": ("mediana", "median"),
    "sum": ("suma", "sum"),
    "stddev": ("desvio estandar", "desviacion estandar", "standard deviation"),
    "count": ("cuantos", "cuantas", "how many"),
}

# Pedidos de un gráfico; fragmento de regex que se inserta sin escapar.
CHART_WORDS = (
    r"(?:gr[aá]fic\w*|plot\w*|chart\w*|diagrama\w*|histograma\w*|visualiz\w*|dibuj\w*|"
    r"barras|l[ií]neas|dispersi[oó]n|distribuci[oó]n|scatter)"
)

# Inicio de una pregunta que continúa el análisis anterior: «¿Y la turbiedad?», «Ahora lo mismo para el pH».
# Frases normalizadas (sin acentos).
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

# Conectores que suman condiciones en lugar de reemplazarlas: «compará con Punta Piedras».
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

# Agrupaciones temporales pedidas por su nombre: «promedio por mes» agrupa la fecha con time_grain=month.
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

# Meses por su nombre, para leer períodos pedidos («marzo de 2022»); palabras normalizadas.
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
# Años de una fecha pedida: «durante 2021», «entre 2019 y 2021».
REQUESTED_YEAR = re.compile(r"\b(1[89]\d{2}|2\d{3})\b")
# Una fecha con un solo extremo no es un intervalo cerrado: «desde 2020» empieza el 1/1/2020 y no termina, «después
# de 2019» empieza en 2020, «hasta 2022» termina el 31/12/2022 y «antes de 2021», el 31/12/2020. Frases normalizadas.
PERIOD_BOUND_CUES = {
    "from": ("desde", "a partir", "since", "from"),
    "after": ("despues", "posterior", "posteriores", "after"),
    "until": ("hasta", "until", "through"),
    "before": ("antes", "anterior", "anteriores", "before", "prior to"),
}
# Varias fechas forman un intervalo sólo si se piden como rango: «entre 2019 y 2021», «de marzo a junio de 2022»,
# «2019-2021»; «en 2019 y en 2021» son dos años sueltos. RANGE_CUES son frases normalizadas; RANGE_CONNECTORS, un
# fragmento de regex entre dos fechas que se inserta sin escapar.
RANGE_CUES = ("entre", "between")
# «Desde 2020 hasta hoy» sólo tiene un extremo: el otro es el presente. Frases normalizadas.
OPEN_END_WORDS = ("hoy", "ahora", "la actualidad", "la fecha", "today", "now", "the present")
RANGE_CONNECTORS = r"(?:-|–|\ba\b|\bal\b|\bhasta\b|\bto\b|\buntil\b|\bthrough\b)"
# Las estaciones del año cambian de meses con el hemisferio: no fijan un intervalo. Palabras normalizadas.
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
# Columnas numéricas que guardan el año de cada registro; nombres normalizados en singular.
YEAR_COLUMN_NAMES = {("ano",), ("anio",), ("year",)}

# Preguntas por el período de un resultado («¿Y de qué año es?»), con la precisión que piden; frases normalizadas.
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

# Tipo de gráfico que nombra una pregunta; frases normalizadas.
CHART_TYPE_WORDS = {
    "bar": ("barra", "barras", "bar", "bars"),
    "line": ("linea", "lineas", "line", "lines", "evolucion"),
    "scatter": ("dispersion", "scatter"),
    "histogram": ("histograma", "histogram"),
}

# Pedidos del extremo de una variable («el valor máximo de Cond», «la turbiedad más alta»); frases normalizadas.
EXTREME_CUES = (
    "maximo",
    "maxima",
    "minimo",
    "minima",
    "mas alto",
    "mas alta",
    "mas bajo",
    "mas baja",
    "mas elevado",
    "mas elevada",
    "mayor valor",
    "menor valor",
    "highest",
    "lowest",
    "maximum",
    "minimum",
)
# Pedidos de un orden completo, no de un extremo: «ordená los registros de mayor a menor».
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

# Estadísticas de una variable medida: sin la variable no hay qué calcular («¿Cuál es el promedio en Punta Lara?»).
MEASURED_STATISTICS = (
    "promedio",
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
