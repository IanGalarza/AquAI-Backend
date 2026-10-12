"""Proyecta el catálogo SQL y su disponibilidad efectiva."""
import asyncio
from uuid import UUID
from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.interfaces.sources import SourceQueries


class GetSourceHandler:
    def __init__(self, queries: SourceQueries) -> None:
        self._queries = queries

    async def handle(self, source_id: UUID) -> SourceDetailResponse:
        source = await asyncio.to_thread(self._queries.get, source_id)
        ready = source["processing_status"] in {"ready", "ready_with_warnings"}
        for dataset in source["datasets"]:
            dataset["queryable"] = bool(
                ready and source["is_active"] and dataset["is_active"] and dataset["queryable"]
            )
        source["queryable"] = any(item["queryable"] for item in source["datasets"])
        return SourceDetailResponse.model_validate(source)
