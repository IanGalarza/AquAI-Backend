"""Receta actual de preparación y validación del archivo completo de una fuente."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aquai.domain.models.json_types import JsonValue

if TYPE_CHECKING:
    from aquai.domain.models.source import Source
    from aquai.domain.models.validation_status import ValidationStatus


@dataclass(kw_only=True, eq=False)
class SourceRecipe:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID
    recipe_data: dict[str, JsonValue]
    rules_snapshot: dict[str, JsonValue]
    validation_status_id: int = 1
    validation_report: dict[str, JsonValue] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    source: Source | None = field(
        default=None, init=False, repr=False, compare=False
    )
    validation_status: ValidationStatus | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("recipe_data", "rules_snapshot", "validation_report"):
            document = getattr(self, name)
            if name == "validation_report" and document is None:
                continue
            if not isinstance(document, dict):
                raise ValueError(f"{name} debe ser un objeto JSON.")
