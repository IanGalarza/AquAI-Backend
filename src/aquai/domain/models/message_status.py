"""Entrada del catálogo que describe el estado de generación de un mensaje."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.domain.models.message import Message


@dataclass(kw_only=True, eq=False)
class MessageStatus:
    id: int
    name: str

    messages: list[Message] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
