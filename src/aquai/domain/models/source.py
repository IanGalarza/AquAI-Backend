"""Archivo original cargado y sus metadatos de procesamiento."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from aquai.domain.models.dataset import Dataset
    from aquai.domain.models.message_source import MessageSource
    from aquai.domain.models.processing_status import ProcessingStatus


@dataclass(kw_only=True, eq=False)
class Source:
    id: UUID = field(default_factory=uuid4)
    name: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    is_active: bool
    description: str | None = None
    processing_status_id: int
    file_content: bytes = field(repr=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    processing_status: ProcessingStatus | None = field(
        default=None, init=False, repr=False, compare=False
    )
    datasets: list[Dataset] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    message_sources: list[MessageSource] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
