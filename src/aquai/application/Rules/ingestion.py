import re

IDENTIFIER_COLUMN = re.compile(
    r"(?:^|[\s_-])(?:id|code|codigo|código|identificador)(?:$|[\s_-])", re.IGNORECASE
)

NON_MEASUREMENT_COLUMN = re.compile(
    r"(?<![^\W\d_])(?:id|code|c[oó]digo|identificador|fecha|date|sitio|site|muestra)(?:e?s)?"
    r"(?![^\W\d_])",
    re.IGNORECASE,
)
MEASUREMENT_COLUMN = re.compile(
    r"\b(?:abundancia|cantidad|recuento|valor|medida|concentraci[oó]n|"
    r"value|amount|count|measurement)\b",
    re.IGNORECASE,
)
MISSING_VALUE_MARKER = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ .<>?/()-]{1,24}")
CENSORED_VALUE = re.compile(r"(?:<=|>=|<|>|≤|≥)\s*\d+(?:[.,]\d+)?")
DECIMAL_COMMA = re.compile(r"[+-]?\d+,\d+")

TOTAL_ROW_LABELS = {"total", "totales", "subtotal", "total general"}
NOTE_ROW = re.compile(r"^(?:nota|notas|fuente|leyenda|note|notes|source):", re.IGNORECASE)
