import re

# Codificaciones publicadas de las fuentes matemáticas de TeX (OMS, OML, OMX y OT1). Muchos PDF
# no declaran qué dibujan estas fuentes y sólo nombran cada glifo por su código: /C14 es el código
# 14. Las tablas se limitan a lo que aparece en texto científico.
TEX_SYMBOL_CODES = {
    0x00: "−",
    0x01: "·",
    0x02: "×",
    0x03: "∗",
    0x04: "÷",
    0x06: "±",
    0x07: "∓",
    0x0E: "°",
    0x0F: "•",
    0x11: "≡",
    0x12: "⊆",
    0x13: "⊇",
    0x14: "≤",
    0x15: "≥",
    0x18: "∼",
    0x19: "≈",
    0x1A: "⊂",
    0x1B: "⊃",
    0x1C: "≪",
    0x1D: "≫",
    0x8A: "]",
}
TEX_ITALIC_CODES = {
    0x0B: "α",
    0x0C: "β",
    0x0D: "γ",
    0x0E: "δ",
    0x0F: "ε",
    0x10: "ζ",
    0x11: "η",
    0x12: "θ",
    0x13: "ι",
    0x14: "κ",
    0x15: "λ",
    0x16: "µ",
    0x17: "ν",
    0x18: "ξ",
    0x19: "π",
    0x1A: "ρ",
    0x1B: "σ",
    0x1C: "τ",
    0x1D: "υ",
    0x1E: "φ",
    0x1F: "χ",
    0x20: "ψ",
    0x21: "ω",
}
TEX_EXTENSION_CODES = {
    0x00: "(",
    0x01: ")",
    0x02: "[",
    0x03: "]",
    0x08: "{",
    0x09: "}",
    0x0C: "|",
    0x10: "(",
    0x11: ")",
    0x12: "(",
    0x13: ")",
    0x14: "[",
    0x15: "]",
    0x1A: "{",
    0x1B: "}",
    0x20: "(",
    0x21: ")",
}
TEX_ACCENT_CODES = {
    0x12: "̀",
    0x13: "́",
    0x14: "̌",
    0x15: "̆",
    0x16: "̄",
    0x17: "̊",
    0x18: "̧",
}

# Estas fuentes no tienen letras ni cifras: lo que el PDF declara como tales son sus símbolos.
TEX_SYMBOL_LETTERS = {
    "0": "′",
    "j": "|",
    "f": "{",
    "g": "}",
    "¼": "=",
    "½": "[",
    "þ": "+",
    "ð": "(",
    "Þ": ")",
}
TEX_ITALIC_LETTERS = {":": ".", ";": ",", "=": "/"}
TEX_EXTENSION_LETTERS = {"P": "∑", "X": "∑", "ð": "∫"}

TEX_SYMBOL_FONT = re.compile(
    r"(?:^|\+)(?:CMSY\d|MTSY|AdvMT_SY|AdvP4C4E74|LucidaNewMath-Symbol"
    r"|TeX_CM(?:_Bold)?_Maths_Symbols)",
    re.IGNORECASE,
)
TEX_ITALIC_FONT = re.compile(
    r"(?:^|\+)(?:CMMI\d|MTMI|AdvMT_MI|AdvP4C4E51|TeX_CM_Maths_Italic)", re.IGNORECASE
)
TEX_EXTENSION_FONT = re.compile(
    r"(?:^|\+)(?:CMEX\d|MTEX|BLEX|AdvP4C4E46|TeX_CM_Maths_Extension)", re.IGNORECASE
)
TEX_ROMAN_FONT = re.compile(r"(?:^|\+)AdvP4C4E59", re.IGNORECASE)

_NO_CODE = "�"
_CODED_FONTS = (
    (TEX_SYMBOL_FONT, TEX_SYMBOL_CODES),
    (TEX_ITALIC_FONT, TEX_ITALIC_CODES),
    (TEX_EXTENSION_FONT, TEX_EXTENSION_CODES),
    (TEX_ROMAN_FONT, TEX_ACCENT_CODES),
)


#
# Indica si una fuente sigue una de las codificaciones de TeX.
#
def is_tex_font(font: str) -> bool:
    return any(pattern.search(font) for pattern, _ in _CODED_FONTS)


#
# Devuelve lo que dibuja un código de una fuente de TeX. El código 0 no llega como carácter: el
# texto extraído lo deja como un reemplazo.
#
def tex_symbol(font: str, char: str) -> str | None:
    code = 0 if char == _NO_CODE else ord(char)
    for pattern, encoding in _CODED_FONTS:
        if pattern.search(font):
            return encoding.get(code)

    return None
