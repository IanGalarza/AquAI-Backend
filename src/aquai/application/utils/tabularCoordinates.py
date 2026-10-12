"""Coordenadas Excel sin dependencias de lectura de archivos."""

import re


def column_label(column: int) -> str:
    label = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        label = chr(65 + remainder) + label
    return label


def cell_coordinate(row: int, column: int) -> str:
    return f"{column_label(column)}{row}"


def cell_position(cell: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell)
    if match is None:
        return None
    column = 0
    for character in match[1]:
        column = column * 26 + ord(character) - ord("A") + 1
    return int(match[2]), column
