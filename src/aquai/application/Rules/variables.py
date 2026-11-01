"""Variables con un tratamiento propio (nombre canónico, unidad implícita, nombres equivalentes)."""

import re

PH_ALIASES = {"ph", "phuph", "phudeph", "phunidadesdeph"}
NAME_SEPARATORS = re.compile(r"[\s_\[\]().-]+")

# Nombres equivalentes de un mismo parámetro de calidad de agua: variantes regionales, inglés y abreviaturas de
# encabezado («Turb», «Cond»). El primero es el nombre canónico. Frases normalizadas; un plural se escribe como en
# los encabezados («coliformes fecales»). Nunca reúne parámetros distintos (nitrato y nitrito, fosfato y fósforo
# total, oxígeno disuelto y su saturación, microcistina y Microcystis) ni abreviaturas que también son palabras
# comunes (do, cod, bod).
VARIABLE_SYNONYMS = (
    ("turbiedad", "turbidez", "turbidity", "turb"),
    ("conductividad", "conductividad electrica", "conductivity", "cond"),
    ("temperatura", "temperature", "temp"),
    ("oxigeno disuelto", "dissolved oxygen", "od"),
    ("saturacion de oxigeno", "sat de oxigeno", "oxygen saturation"),
    ("clorofila", "chlorophyll", "chl"),
    ("feofitina", "pheophytin", "phaeophytin"),
    ("cianobacteria", "cyanobacteria"),
    ("fitoplancton", "phytoplankton"),
    ("microcistina", "microcystina", "microcystin"),
    ("nitrato", "nitrate"),
    ("nitrito", "nitrite"),
    # El nitrógeno amoniacal se mide como amonio; el amoníaco (NH3 no ionizado) es otra fracción.
    ("amonio", "nitrogeno amoniacal", "ammonium", "ammoniacal nitrogen"),
    ("fosfato", "ortofosfato", "fosforo reactivo soluble", "phosphate", "orthophosphate"),
    ("alcalinidad", "alkalinity"),
    ("salinidad", "salinity"),
    ("dureza", "hardness"),
    ("coliforme", "coliform"),
    (
        "coliformes fecales",
        "coliformes termotolerantes",
        "fecal coliforms",
        "faecal coliforms",
        "thermotolerant coliforms",
    ),
    ("escherichia coli", "e coli"),
    ("enterococo", "enterococci", "enterococcus"),
    ("solidos disueltos totales", "total dissolved solids", "sdt", "tds"),
    ("solidos suspendidos totales", "total suspended solids", "sst", "tss"),
    ("demanda quimica de oxigeno", "chemical oxygen demand", "dqo"),
    ("demanda biologica de oxigeno", "biochemical oxygen demand", "dbo"),
)

# Fórmulas con las que un encabezado nombra un nutriente («N-NO3 [mg N L-1]», «NO3-N», «PO4»). Se buscan tal cual en
# el nombre original: normalizada, «NO3» se confunde con «Nº 3». Sirven para encontrar la columna; si sus valores son
# comparables lo decide la base química (units.comparable_unit). «N-NH3» es el nitrógeno amoniacal total; «NH3» suelto,
# el amoníaco no ionizado, que es otra fracción y no se reúne con el amonio.
CHEMICAL_FORMULAS = {
    "nitrato": ("NO3",),
    "nitrito": ("NO2",),
    "amonio": ("NH4", "N-NH3", "NH3-N"),
    "fosfato": ("PO4",),
}

# Símbolos que nombran una variable sólo si la unidad del encabezado lo confirma: «T (°C)» es una temperatura y
# «CE (µS/cm)», una conductividad; sin unidad, «T», «Q» o «EC» (también E. coli) son ambiguos. Unidades como las
# deja unit_key.
UNIT_CONFIRMED_SYMBOLS = {
    "temperatura": (("t",), ("°c", "°f")),
    "conductividad": (("ce", "ec"), ("μs/cm", "ms/cm", "s/m", "ms/m", "ds/m")),
    "caudal": (("q",), ("m3/s", "l/s", "m3/h")),
}

# Un encabezado que nombra otro medio mide otra cosa: la temperatura del aire o «meteorológicos | Temperatura» no
# es la del agua. Sólo responde si la pregunta nombra ese medio.
OTHER_MEDIUM_WORDS = (
    "aire",
    "ambiente",
    "atmosferico",
    "atmosferica",
    "meteorologico",
    "meteorologica",
    "air",
    "ambient",
    "atmospheric",
    "meteorological",
)
# El agua es el medio por defecto: «temperatura del agua» no precisa otra variable que «temperatura».
WATER_WORDS = ("agua", "water")

# Un encabezado que suma magnitudes («Clorofila "a" + Feofitina "a"», «NO3+NO2») es otra variable, salvo que se
# pida la suma; «NH4+» es la carga del ion, no una suma.
DERIVED_SUM = re.compile(r"\+\s*[^\W\d_]")
SUM_WORDS = ("suma", "sum")

# Variantes de una variable que no son su valor absoluto: una proporción o saturación («% sat de oxígeno»,
# «Abundancia relativa») o un recuento de colonias, individuos o filamentos en vez de células, que no se convierten
# entre sí. Se usan si la pregunta las pide o si el dataset no registra otra columna de esa variable.
SECONDARY_QUANTITY_WORDS = (
    "relativa",
    "relativo",
    "relative",
    "porcentaje",
    "percentage",
    "percent",
    "saturacion",
    "saturation",
    "sat",
    "colonias",
    "colonies",
    "individuos",
    "individuals",
    "ind",
    "filamentos",
    "filaments",
    "fil",
    "tricomas",
    "trichomes",
)
PERCENT_SIGN = "%"
