from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.dataset import Dataset
    from aquai.domain.models.message import Message
    from aquai.domain.models.source import Source


@dataclass(kw_only=True, eq=False)
class MessageSource:
    id: UUID = field(default_factory=uuid4)
    message_id: UUID
    source_id: UUID
    dataset_id: UUID | None = None
    reference_order: int
    details: JsonDocument
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    message: Message | None = field(default=None, init=False, repr=False, compare=False)
    source: Source | None = field(default=None, init=False, repr=False, compare=False)
    dataset: Dataset | None = field(default=None, init=False, repr=False, compare=False)
