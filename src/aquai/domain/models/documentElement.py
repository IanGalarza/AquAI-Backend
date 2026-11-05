from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.enums.documentContent import DocumentElementKind
from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.document import Document


@dataclass(kw_only=True, eq=False)
class DocumentElement:
    id: UUID = field(default_factory=uuid4)
    document_id: UUID
    section_id: UUID | None = None
    position: int
    kind: DocumentElementKind
    label: str | None = None
    caption: str | None = None
    content: str | None = None
    cells: list[list[str]] | None = None
    page_start: int | None = None
    page_end: int | None = None
    locations: list[JsonDocument] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    document: Document | None = field(default=None, init=False, repr=False, compare=False)
