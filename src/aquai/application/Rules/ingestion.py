"""Cómo se interpretan columnas y filas de una planilla al prepararla."""

import re

# Las columnas identificadoras se conservan como texto aunque sus valores parezcan números.
IDENTIFIER_COLUMN = re.compile(
    r"(?:^|[\s_-])(?:id|code|codigo|código|identificador)(?:$|[\s_-])", re.IGNORECASE
)

# Proyección numérica de una columna de texto: sólo en columnas de medición, nunca en estas.
NON_MEASUREMENT_COLUMN = re.compile(
    r"(?:id|code|c[oó]digo|identificador|fecha|date|sitio|site|muestra)", re.IGNORECASE
)
MEASUREMENT_COLUMN = re.compile(
    r"\b(?:abundancia|cantidad|recuento|valor|medida|concentraci[oó]n|"
    r"value|amount|count|measurement)\b",
    re.IGNORECASE,
)
MISSING_VALUE_MARKER = re.compile(r"[a-zA-Z .<>?-]{1,16}")

TOTAL_ROW_LABELS = {"total", "totales", "subtotal", "total general"}
NOTE_ROW = re.compile(r"^(?:nota|notas|fuente|leyenda|note|notes|source):", re.IGNORECASE)
