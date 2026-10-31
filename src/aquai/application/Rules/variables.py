"""Variables con un tratamiento propio (nombre canónico, unidad implícita)."""

import re

PH_ALIASES = {"ph", "phuph", "phudeph", "phunidadesdeph"}
NAME_SEPARATORS = re.compile(r"[\s_\[\]().-]+")
