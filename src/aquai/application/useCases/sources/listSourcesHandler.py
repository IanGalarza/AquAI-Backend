"""Listado paginado y filtrado de fuentes."""

import asyncio

from aquai.application.errors import TabularInputError
from aquai.application.interfaces.sources import SourceQueries


PROCESSING_STATUSES = {
    "uploaded", "processing", "ready", "ready_with_warnings", "review_required", "failed",
}
VALIDATION_STATUSES = {"pending", "ready", "ready_with_warnings", "review_required", "failed"}


class ListSourcesHandler:
    def __init__(self, queries: SourceQueries) -> None:
        self._queries = queries

    async def handle(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        name: str | None = None,
        processing_status: str | None = None,
        validation_status: str | None = None,
    ) -> dict:
        if not 1 <= page or not 1 <= page_size <= 100:
            raise TabularInputError("La página debe ser positiva y el tamaño estar entre 1 y 100.")
        for value, allowed in (
            (processing_status, PROCESSING_STATUSES),
            (validation_status, VALIDATION_STATUSES),
        ):
            if value is not None and value not in allowed:
                raise TabularInputError("El estado solicitado no pertenece al catálogo.")
        return await asyncio.to_thread(
            self._queries.list,
            page=page,
            page_size=page_size,
            search=name,
            processing_status=processing_status,
            validation_status=validation_status,
        )
