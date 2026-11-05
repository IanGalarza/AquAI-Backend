from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from aquai.domain.models.document import Document


@dataclass(kw_only=True, eq=False)
class DocumentSection:
    id: UUID = field(default_factory=uuid4)
    document_id: UUID
    position: int
    level: int
    title: str
    path: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    document: Document | None = field(default=None, init=False, repr=False, compare=False)
