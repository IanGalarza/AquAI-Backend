from typing import Protocol

from aquai.application.contracts.ingestion import (
    PreparationResult,
    PreparedTable,
    SemanticDescription,
    StructureSuggestion,
)


class TabularProcessor(Protocol):
    #
    # Valida el archivo cargado y devuelve su tipo.
    #
    def validate_upload(self, filename: str | None, content: bytes) -> str: ...

    #
    # Prepara las tablas del archivo, aplicando las sugerencias de estructura si existen.
    #
    def prepare(
        self, filename: str, content: bytes, suggestions: list[StructureSuggestion] | None = None
    ) -> PreparationResult: ...


class SemanticProvider(Protocol):
    #
    # Sugiere la estructura de las tablas candidatas detectadas en un archivo.
    #
    async def suggest_structure(self, candidates: list[dict]) -> list[StructureSuggestion]: ...

    #
    # Describe semánticamente una tabla preparada y sus columnas.
    #
    async def describe(self, table: PreparedTable) -> SemanticDescription: ...


class SemanticIndex(Protocol):
    #
    # Indexa semánticamente los datasets de una fuente.
    #
    async def index_source(self, source: dict) -> None: ...

    #
    # Marca en el índice semántico si una fuente está activa.
    #
    async def set_source_active(self, source_id: str, active: bool) -> None: ...
