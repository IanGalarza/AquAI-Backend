"""Archivo original cargado y sus metadatos de procesamiento."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus
from aquai.domain.models.jsonTypes import JsonDocument

if TYPE_CHECKING:
    from aquai.domain.models.dataset import Dataset
    from aquai.domain.models.messageSource import MessageSource


@dataclass(kw_only=True, eq=False)
class Source:
    id: UUID = field(default_factory=uuid4)
    name: str
    original_filename: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    is_active: bool = True
    description: str | None = None
    file_content: bytes = field(repr=False)
    content_hash: str = ""
    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    processing_report: JsonDocument = field(default_factory=dict)
    semantic_index_status: SemanticIndexStatus = SemanticIndexStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    datasets: list[Dataset] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    message_sources: list[MessageSource] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
