import asyncio
from uuid import UUID

from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.interfaces.documents import DocumentQueries
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.Rules.documents import DOCUMENT_FILE_TYPES

READY_STATUSES = {"ready", "ready_with_warnings"}


class SourceDetailService:
    #
    # Guarda las consultas de fuentes y de documentos que usa el servicio.
    #
    def __init__(self, queries: SourceQueries, documents: DocumentQueries | None = None) -> None:
        self._queries = queries
        self._documents = documents

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
        if self._documents and source["file_type"] in DOCUMENT_FILE_TYPES:
            source["document"] = await self._document_summary(source)
            source["queryable"] = self._searchable_document(source, source_ready)

        return SourceDetailResponse.model_validate(source)

    #
    # Indica si el chat puede citar el documento: activo, preparado, indexado y sin repetir otra fuente.
    #
    @staticmethod
    def _searchable_document(source: dict, ready: bool) -> bool:
        document = source.get("document") or {}

        return bool(
            ready
            and source["is_active"]
            and str(source["semantic_index_status"]) == "ready"
            and not document.get("duplicate_of_source_id")
        )

    #
    # Devuelve el resumen del documento con el motivo del último fallo de indexación.
    #
    async def _document_summary(self, source: dict) -> dict | None:
        summary = await asyncio.to_thread(self._documents.summary, source["id"])
        if summary is None:
            return None

        return {**summary, "index_error": source["processing_report"].get("semantic_index_error")}
