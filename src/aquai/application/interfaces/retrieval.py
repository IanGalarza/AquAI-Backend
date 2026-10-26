"""Puerto de lectura del índice existente; no depende de Qdrant ni de Ollama."""

from typing import Any, Protocol


class DatasetSemanticSearch(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...
