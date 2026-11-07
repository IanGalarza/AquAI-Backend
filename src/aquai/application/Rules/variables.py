import re

PH_ALIASES = {"ph", "phuph", "phudeph", "phunidadesdeph"}
NAME_SEPARATORS = re.compile(r"[\s_\[\]().-]+")

VARIABLE_SYNONYMS = (
    ("turbiedad", "turbidez", "turbidity", "turb"),
    ("conductividad", "conductividad electrica", "conductivity", "cond"),
    ("temperatura", "temperature", "temp"),
    ("oxigeno disuelto", "dissolved oxygen", "od"),
    ("saturacion de oxigeno", "sat de oxigeno", "oxygen saturation"),
    ("clorofila", "chlorophyll", "chl"),
    ("feofitina", "pheophytin", "phaeophytin"),
    ("cianobacteria", "cyanobacteria", "cyanobacterial"),
    ("fitoplancton", "phytoplankton"),
    ("microcistina", "microcystina", "microcystin"),
    ("nitrato", "nitrate"),
    ("nitrito", "nitrite"),
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

CHEMICAL_FORMULAS = {
    "nitrato": ("NO3",),
    "nitrito": ("NO2",),
    "amonio": ("NH4", "N-NH3", "NH3-N"),
    "fosfato": ("PO4",),
}

UNIT_CONFIRMED_SYMBOLS = {
    "temperatura": (("t",), ("°c", "°f")),
    "conductividad": (("ce", "ec"), ("μs/cm", "ms/cm", "s/m", "ms/m", "ds/m")),
    "caudal": (("q",), ("m3/s", "l/s", "m3/h")),
}

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
WATER_WORDS = ("agua", "water")

DERIVED_SUM = re.compile(r"\+\s*[^\W\d_]")
SUM_WORDS = ("suma", "sum")

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
