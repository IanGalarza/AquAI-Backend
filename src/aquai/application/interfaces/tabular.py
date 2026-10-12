"""Lectura determinística y descripción semántica sin planes ejecutables."""
from typing import Protocol
from aquai.application.contracts.ingestion import PreparationResult, PreparedTable, SemanticDescription


class TabularProcessor(Protocol):
    def validate_upload(self, filename: str | None, content: bytes) -> str: ...
    def prepare(self, filename: str, content: bytes) -> PreparationResult: ...


class SemanticProvider(Protocol):
    async def describe(self, table: PreparedTable) -> SemanticDescription: ...


class SemanticIndex(Protocol):
    async def index_source(self, source: dict) -> None: ...
    async def set_source_active(self, source_id: str, active: bool) -> None: ...
