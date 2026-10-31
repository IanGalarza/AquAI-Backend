"""Unidades que se reconocen al final de un encabezado: «Turbiedad (NTU)», «Cianobacterias [cel mL-1]»."""

import re

HEADER_UNIT = re.compile(r"(?:\[([^\[\]]+)\]|\(([^()]+)\))\s*$")
SIMPLE_UNIT = re.compile(
    r"(?:%|[munpkMGTµμ]?g|[munpkµμ]?m|[munpµμ]?[Ll]|°[CF]|[munpµμ]?mol|s|min|h|ppm|ppb)"
)
INVERSE_UNIT = re.compile(r"[\wµμ°%^. ·-]+\s+[munpµμ]?[Llms](?:-\d+|\^-\d+)")
RATIO_UNIT = re.compile(r"[\wµμ°%^. ·-]+/[\wµμ°%^. ·-]+")
MAX_UNIT_LENGTH = 40
