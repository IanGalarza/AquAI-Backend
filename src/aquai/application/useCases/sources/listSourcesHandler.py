"""Listado paginado y filtrado de fuentes."""
import asyncio
from aquai.application.dtos.requests.listSources import ListSourcesRequest
from aquai.application.dtos.responses.listSources import SourceListResponse
from aquai.application.interfaces.sources import SourceQueries


class ListSourcesHandler:
    def __init__(self, queries: SourceQueries) -> None:
        self._queries = queries

    async def handle(self, request: ListSourcesRequest) -> SourceListResponse:
        result = await asyncio.to_thread(
            self._queries.list, page=request.page, page_size=request.page_size,
            search=request.name, processing_status=request.processing_status, sort=request.sort,
        )
        return SourceListResponse.model_validate(result)
