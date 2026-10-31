"""Palabras que delatan el idioma de una pregunta cuando no hay una preferencia explícita."""

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
