"""Puertos de persistencia: cada escritura termina su propia transacción."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol
from uuid import UUID

from aquai.domain.models.source import Source


class SourceQueries(Protocol):
    def list(self, *, page: int, page_size: int, search: str | None = None,
             processing_status: str | None = None,
             validation_status: str | None = None) -> dict: ...

    def get(self, source_id: UUID, *, include_content: bool = False) -> dict: ...


class SourceCommands(Protocol):
    def create_upload(self, source: Source, recipe_data: dict) -> tuple[UUID, bool]: ...

    def worker_lock(self) -> AbstractContextManager[bool]: ...

    def claim_next(self) -> UUID | None: ...

    def recover_interrupted(self) -> int: ...

    def save(self, source_id: UUID, *, recipe_data: dict | None = None,
             rules_snapshot: dict | None = None, validation_report: dict | None = None,
             processing_status: str | None = None, validation_status: str | None = None,
             expected_processing_status: str | None = None) -> bool: ...

    def mutate(self, source_id: UUID,
               callback: Callable[[dict], tuple[dict, dict]]) -> dict: ...
