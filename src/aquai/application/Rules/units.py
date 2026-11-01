"""Unidades que se reconocen al final de un encabezado: «Turbiedad (NTU)», «Cianobacterias [cel mL-1]»."""

import re

HEADER_UNIT = re.compile(r"(?:\[([^\[\]]+)\]|\(([^()]+)\))\s*$")
# Turbiedad: NTU/UNT/UTN (nefelométricas, en inglés y en castellano), FNU y FTU. Grados con «°» o con «º».
SIMPLE_UNIT = re.compile(
    r"(?:%|[munpkMGTµμ]?g|[munpkµμ]?m|[munpµμ]?[Ll]|[°º][CF]|[munpµμ]?mol|s|min|h|ppm|ppb|"
    r"NTU|UNT|UTN|FNU|FTU)"
)
# Potencias negativas de una unidad con prefijo: «Cel mL-1», «mg L-1», «mS cm-1».
INVERSE_UNIT = re.compile(r"[\wµμ°%^. ·-]+\s+[cmunpµμ]?[Llms](?:-\d+|\^-\d+)")
RATIO_UNIT = re.compile(r"[\wµμ°%^. ·-]+/[\wµμ°%^. ·-]+")
MAX_UNIT_LENGTH = 40
# Separador de un encabezado apilado cuya fila inferior no se reconoció como unidad: «Cond | mS cm-1».
STACKED_HEADER_SEPARATOR = " | "
# Símbolos UTF-8 leídos como Latin-1 en el archivo de origen: «ÂµS/cm» es µS/cm y «Â°C», °C.
MISDECODED_SYMBOLS = {"Âµ": "µ", "Â°": "°"}
# El indicador ordinal «º» escrito en lugar del signo de grado: «ºC» es °C.
SYMBOL_VARIANTS = {"º": "°"}
# El prefijo micro escrito con la «u» del teclado: «ug/L», «uS/cm». Nunca la «u» de UFC, UNT o «U de pH».
ASCII_MICRO_PREFIX = re.compile(r"(?<![^\W\d_])u(?=(?:g|s|m|l|mol)(?![^\W\d_]))")
# Palabras de una misma unidad, ya normalizadas: células y cells cuentan lo mismo; litro es L.
UNIT_WORDS = {
    "celulas": "cel",
    "celula": "cel",
    "cells": "cel",
    "cell": "cel",
    "cfu": "ufc",
    "mpn": "nmp",
    "litros": "l",
    "litro": "l",
    "liters": "l",
    "liter": "l",
    "litres": "l",
    "litre": "l",
}
# Especie química en que se expresa una concentración de nutriente. Encontrar la columna de un mismo analito
# (nitrato, N-NO3) no la vuelve comparable: 1 mg NO3/L son 0,226 mg N/L. La base se lee de la unidad («mg N L-1»,
# «mg N/litro», «µg P/L», «mg NO3/L», «mg/L N», «mg/L como N»), de un prefijo o sufijo del nombre («N-NO3», «NO3-N»,
# «P-PO4», «PO4-P», «N-NH3») o de un nombre que es el propio elemento («Nitrógeno amoniacal»). «Nitratos (mg/l)» no
# declara su base y nunca se convierte.
UNIT_BASIS = re.compile(
    r"(?<=g)\s*(?P<lead>N|P|NO3|NO2|NH4|PO4)(?=\s*(?:/|[Ll]\b|[Ll]-|[Ll]itros?\b))|"
    r"(?P<unit>/\s*(?:[Ll]itros?|[Ll]))\s+(?:(?:como|as|expresado como)\s+)?(?P<tail>N|P|NO3|NO2|NH4|PO4)\b"
)
NAME_BASIS = re.compile(
    r"(?<![A-Za-z])(?P<lead>N|P)\s*-\s*(?:NO3|NO2|NH4|NH3|PO4)\b|"
    r"\b(?:NO3|NO2|NH4|NH3|PO4)\s*-\s*(?P<tail>N|P)\b|"
    r"\b(?:como|as|expresado como)\s+(?P<word>N|P)\b"
)
ELEMENT_NAMES = {"nitrogeno": "N", "nitrogen": "N", "fosforo": "P", "phosphorus": "P"}
# Unidades idénticas con otra grafía, ya normalizadas: UNT y UTN son la NTU en castellano; 1 mg/m3 = 1 µg/L; la DBO,
# la DQO y el oxígeno disuelto en mg/L ya se expresan como O2. Una base distinta (mg N/L frente a mg/L) o una unidad
# de recuento distinta (células, colonias, individuos) no es la misma unidad.
UNIT_EQUIVALENTS = {"unt": "ntu", "utn": "ntu", "mg/m3": "μg/l", "mgo2/l": "mg/l"}
