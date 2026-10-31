from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.messageSource import MessageSource
    from aquai.domain.models.source import Source


@dataclass(kw_only=True, eq=False)
class Dataset:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID
    name: str
    description: str | None = None
    source_locator: JsonDocument
    physical_table_name: str
    version: int = 1
    is_active: bool = True
    row_count: int = 0
    columns_metadata: list[JsonDocument] = field(default_factory=list)
    queryable: bool = False
    quality_warnings: list[str] = field(default_factory=list)
    semantic_description: str | None = None
    semantic_point_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    source: Source | None = field(default=None, init=False, repr=False, compare=False)
    message_sources: list[MessageSource] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
