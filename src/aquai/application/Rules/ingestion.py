"""Cómo se interpretan columnas y filas de una planilla al prepararla."""

import re

# Las columnas identificadoras se conservan como texto aunque sus valores parezcan números.
IDENTIFIER_COLUMN = re.compile(
    r"(?:^|[\s_-])(?:id|code|codigo|código|identificador)(?:$|[\s_-])", re.IGNORECASE
)

# Proyección numérica de una columna de texto: sólo en columnas de medición, nunca en estas.
# Palabras completas (en singular o plural): «id» no debe excluir Turbidez ni Conductividad.
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
# Marcadores de dato faltante o no detectado («ND», «s/d», «n/d (no detectable)», «<LD», «falta»).
MISSING_VALUE_MARKER = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ .<>?/()-]{1,24}")
# Valores censurados: el límite de detección o de cuantificación no es el valor medido («<0,001», «> 10000»). Se
# conservan en el texto y quedan sin dato en la representación numérica; nunca valen 0 ni el límite.
CENSORED_VALUE = re.compile(r"(?:<=|>=|<|>|≤|≥)\s*\d+(?:[.,]\d+)?")
# Coma decimal en un texto: «12,5». Una columna la usa si ningún valor tiene punto decimal y alguno no puede ser un
# separador de miles («0,5», «12,25»; «1,234» sí podría serlo).
DECIMAL_COMMA = re.compile(r"[+-]?\d+,\d+")

TOTAL_ROW_LABELS = {"total", "totales", "subtotal", "total general"}
NOTE_ROW = re.compile(r"^(?:nota|notas|fuente|leyenda|note|notes|source):", re.IGNORECASE)
