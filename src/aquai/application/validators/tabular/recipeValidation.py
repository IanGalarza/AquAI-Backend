from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.application.contracts.tabular.recipe import PreparationRecipe, SheetRecipe


def validate_header_rows(block: SheetRecipe) -> None:
    if block.header_rows != sorted(set(block.header_rows)):
        raise ValueError("header_rows debe contener filas únicas en orden ascendente.")
    if block.header_rows and block.header_rows[-1] >= block.data_start_row:
        raise ValueError("Los encabezados deben estar antes de data_start_row.")


def validate_data_ranges(block: SheetRecipe) -> None:
    if block.data_end_row < block.data_start_row:
        raise ValueError("data_end_row debe ser mayor o igual a data_start_row.")
    if block.data_end_column < block.data_start_column:
        raise ValueError("data_end_column debe ser mayor o igual a data_start_column.")


def validate_block_role(block: SheetRecipe) -> None:
    match block.role:
        case "unknown" if block.transformations:
            raise ValueError("Un bloque unknown debe conservarse sin transformaciones.")
        case "header" if block.header_rows:
            raise ValueError("Un bloque de rol header conserva los encabezados en su rango y debe tener header_rows vacío.")
        case "observations" | "total" | "legend" | "derived" | "summary" | "header" | "unknown":
            return
        case _:
            raise ValueError("El rol del bloque no está permitido.")


def validate_sheet_recipe(block: SheetRecipe) -> None:
    validate_header_rows(block)
    validate_data_ranges(block)
    validate_block_role(block)


def validate_block_identifiers(recipe: PreparationRecipe) -> None:
    block_ids = [block.block_id for block in recipe.sheets]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("Los block_id deben ser únicos en toda la receta.")


def _validate_header_reference(referencing: SheetRecipe, content: SheetRecipe) -> None:
    if content.role == "header":
        return
    conflicting_rows = [
        row for row in referencing.header_rows
        if content.data_start_row <= row <= content.data_end_row
    ]
    if conflicting_rows:
        raise ValueError(
            f"Hoja '{referencing.sheet_name}': El bloque '{referencing.block_id}' "
            f"usa como encabezados las filas {conflicting_rows} del rango de "
            f"datos no header del bloque '{content.block_id}'."
        )


def _validate_block_pair(block: SheetRecipe, other: SheetRecipe) -> None:
    columns_overlap = max(block.data_start_column, other.data_start_column) <= min(
        block.data_end_column, other.data_end_column,
    )
    if not columns_overlap:
        return
    rows_overlap = max(block.data_start_row, other.data_start_row) <= min(
        block.data_end_row, other.data_end_row,
    )
    if rows_overlap:
        raise ValueError(
            f"Hoja '{block.sheet_name}': Los rangos de datos de los bloques "
            f"'{block.block_id}' y '{other.block_id}' no pueden solaparse."
        )
    _validate_header_reference(block, other)
    _validate_header_reference(other, block)


def validate_block_overlaps(recipe: PreparationRecipe) -> None:
    by_sheet: dict[str, list[SheetRecipe]] = {}
    for block in recipe.sheets:
        by_sheet.setdefault(block.sheet_name, []).append(block)
    for blocks in by_sheet.values():
        for block, other in combinations(blocks, 2):
            _validate_block_pair(block, other)


def validate_recipe_status(recipe: PreparationRecipe) -> None:
    if recipe.status == "ready":
        if not recipe.sheets:
            raise ValueError("Una receta ready debe incluir al menos una hoja.")
        if recipe.requires_human_review or recipe.warnings:
            raise ValueError("Una receta ready debe indicar requires_human_review=false y warnings vacío.")
    if any(block.role == "unknown" for block in recipe.sheets) and recipe.status != "review_required":
        raise ValueError("Los bloques unknown requieren status=review_required.")
    match recipe.status:
        case "review_required":
            if not recipe.requires_human_review:
                raise ValueError("Una receta review_required debe indicar requires_human_review=true.")
            if not recipe.warnings:
                raise ValueError("Una receta review_required debe explicar su incertidumbre en warnings.")
        case "failed":
            if not recipe.warnings:
                raise ValueError("Una receta failed debe explicar el impedimento identificado en warnings.")
        case "ready":
            return
        case _:
            raise ValueError("El estado de la receta no está permitido.")


def validate_preparation_recipe(recipe: PreparationRecipe) -> None:
    validate_block_identifiers(recipe)
    validate_block_overlaps(recipe)
    validate_recipe_status(recipe)
