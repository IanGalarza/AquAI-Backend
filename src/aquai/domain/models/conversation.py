"""Conversación compartida dentro del historial del chat."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from aquai.domain.models.message import Message


@dataclass(kw_only=True, eq=False)
class Conversation:
    id: UUID = field(default_factory=uuid4)
    title: str
    last_message_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    messages: list[Message] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
