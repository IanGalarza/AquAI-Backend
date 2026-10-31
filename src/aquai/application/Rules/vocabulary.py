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

# Observaciones puntuales: «la última medición» pide un dato, «el último año» pide un intervalo.
# OBSERVATION_NOUNS_ES/EN son fragmentos de regex que se insertan sin escapar («medici[oó]n»).
OBSERVATION_NOUNS_ES = (
    r"(?:datos?|medici[oó]n|mediciones|registros?|observaci[oó]n|observaciones|valores?|"
    r"lecturas?|muestras?|fechas?)"
)
OBSERVATION_NOUNS_EN = (
    r"(?:measurements?|observations?|records?|readings?|values?|data(?:\s+points?)?)"
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
