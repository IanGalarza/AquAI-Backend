from typing import Any, Protocol


class DatasetSemanticSearch(Protocol):
    #
    # Busca semánticamente los datasets más cercanos a una consulta, hasta un límite.
    #
    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...
