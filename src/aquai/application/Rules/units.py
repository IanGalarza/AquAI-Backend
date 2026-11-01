import re

HEADER_UNIT = re.compile(r"(?:\[([^\[\]]+)\]|\(([^()]+)\))\s*$")
SIMPLE_UNIT = re.compile(
    r"(?:%|[munpkMGTµμ]?g|[munpkµμ]?m|[munpµμ]?[Ll]|[°º][CF]|[munpµμ]?mol|s|min|h|ppm|ppb|"
    r"NTU|UNT|UTN|FNU|FTU)"
)
INVERSE_UNIT = re.compile(r"[\wµμ°%^. ·-]+\s+[cmunpµμ]?[Llms](?:-\d+|\^-\d+)")
RATIO_UNIT = re.compile(r"[\wµμ°%^. ·-]+/[\wµμ°%^. ·-]+")
MAX_UNIT_LENGTH = 40
STACKED_HEADER_SEPARATOR = " | "
MISDECODED_SYMBOLS = {"Âµ": "µ", "Â°": "°"}
SYMBOL_VARIANTS = {"º": "°"}
ASCII_MICRO_PREFIX = re.compile(r"(?<![^\W\d_])u(?=(?:g|s|m|l|mol)(?![^\W\d_]))")
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
UNIT_EQUIVALENTS = {"unt": "ntu", "utn": "ntu", "mg/m3": "μg/l", "mgo2/l": "mg/l"}
