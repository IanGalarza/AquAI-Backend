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
