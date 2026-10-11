"""Entrada del catálogo del resultado actual de validación de una receta."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.domain.models.source_recipe import SourceRecipe


@dataclass(kw_only=True, eq=False)
class ValidationStatus:
    id: int
    name: str

    source_recipes: list[SourceRecipe] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
