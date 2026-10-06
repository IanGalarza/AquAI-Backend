"""Mensaje del chat, contenido estructurado y estado de generación."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.json_types import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.conversation import Conversation
    from aquai.domain.models.message_source import MessageSource
    from aquai.domain.models.message_status import MessageStatus


@dataclass(kw_only=True, eq=False)
class Message:
    id: UUID = field(default_factory=uuid4)
    conversation_id: UUID
    sequence_number: int
    role: str
    content: str
    content_data: JsonDocument = field(default_factory=dict)
    status_id: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    conversation: Conversation | None = field(
        default=None, init=False, repr=False, compare=False
    )
    status: MessageStatus | None = field(
        default=None, init=False, repr=False, compare=False
    )
    message_sources: list[MessageSource] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
