import re

WORD = re.compile(r"[a-záéíóúñ]+")
ENGLISH_CUES = {
    "what",
    "which",
    "show",
    "average",
    "mean",
    "please",
    "plot",
    "the",
    "rainfall",
    "how",
}
#
# Palabras interrogativas, normalizadas sin tildes: una consulta que empieza con una de ellas plantea su propia
# pregunta; una afirmación sin ellas («entonces la temperatura es un factor») se apoya en lo ya conversado.
#
INTERROGATIVE_WORDS = {
    "que",
    "cual",
    "cuales",
    "como",
    "cuando",
    "donde",
    "adonde",
    "quien",
    "quienes",
    "cuanto",
    "cuanta",
    "cuantos",
    "cuantas",
    "porque",
    "what",
    "which",
    "how",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
}
SPANISH_CUES = {
    "qué",
    "cuál",
    "mostrá",
    "promedio",
    "media",
    "gráfico",
    "graficá",
    "generá",
    "por",
    "los",
    "las",
    "del",
}
