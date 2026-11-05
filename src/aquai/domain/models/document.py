from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.documentChunk import DocumentChunk
    from aquai.domain.models.documentElement import DocumentElement
    from aquai.domain.models.documentSection import DocumentSection
    from aquai.domain.models.source import Source


@dataclass(kw_only=True, eq=False)
class Document:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID
    title: str | None = None
    language: str | None = None
    page_count: int | None = None
    text_hash: str
    duplicate_of_source_id: UUID | None = None
    details: JsonDocument = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    source: Source | None = field(default=None, init=False, repr=False, compare=False)
    sections: list[DocumentSection] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    elements: list[DocumentElement] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    chunks: list[DocumentChunk] = field(default_factory=list, init=False, repr=False, compare=False)
