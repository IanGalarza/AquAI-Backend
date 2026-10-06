"""Tipos de valores JSON compartidos por los modelos de dominio."""

from typing import TypeAlias


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

# Los documentos JSON obligatorios pueden contener valores nulos, pero su raíz no puede ser nula.
JsonDocument: TypeAlias = (
    str | int | float | bool | list[JsonValue] | dict[str, JsonValue]
)
