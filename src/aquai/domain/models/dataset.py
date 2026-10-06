"""Conjunto de datos lógico preparado a partir de una fuente original."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.json_types import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.message_source import MessageSource
    from aquai.domain.models.source import Source


@dataclass(kw_only=True, eq=False)
class Dataset:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID
    name: str
    description: str | None = None
    source_locator: JsonDocument
    physical_table_name: str | None = None
    version: int
    is_active: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    source: Source | None = field(
        default=None, init=False, repr=False, compare=False
    )
    message_sources: list[MessageSource] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
