import asyncio
from uuid import UUID

from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.interfaces.sources import SourceQueries

READY_STATUSES = {"ready", "ready_with_warnings"}


class SourceDetailService:
    #
    # Guarda las consultas de fuentes que usa el servicio.
    #
    def __init__(self, queries: SourceQueries) -> None:
        self._queries = queries

    #
    # Devuelve el detalle de una fuente indicando qué datasets son consultables.
    #
    async def get(self, source_id: UUID) -> SourceDetailResponse:
        source = await asyncio.to_thread(self._queries.get, source_id)
        source_ready = source["processing_status"] in READY_STATUSES

        for dataset in source["datasets"]:
            dataset["queryable"] = bool(
                source_ready
                and source["is_active"]
                and dataset["is_active"]
                and dataset["queryable"]
            )

        source["queryable"] = any(dataset["queryable"] for dataset in source["datasets"])

        return SourceDetailResponse.model_validate(source)
