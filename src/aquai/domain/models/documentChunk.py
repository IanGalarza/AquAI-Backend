from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.enums.documentContent import DocumentChunkKind
from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.document import Document


@dataclass(kw_only=True, eq=False)
class DocumentChunk:
    id: UUID = field(default_factory=uuid4)
    document_id: UUID
    section_id: UUID | None = None
    element_id: UUID | None = None
    position: int
    kind: DocumentChunkKind
    text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    locations: list[JsonDocument] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    document: Document | None = field(default=None, init=False, repr=False, compare=False)
