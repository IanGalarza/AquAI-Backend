"""Entrada del catálogo que describe el estado de procesamiento de una fuente."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.domain.models.source import Source


@dataclass(kw_only=True, eq=False)
class ProcessingStatus:
    id: int
    name: str

    sources: list[Source] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
