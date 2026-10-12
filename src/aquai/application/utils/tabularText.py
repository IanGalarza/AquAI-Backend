"""Normalización compartida para comparar nombres e instrucciones del catálogo."""

import unicodedata


def normalized_name(value: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).strip()
