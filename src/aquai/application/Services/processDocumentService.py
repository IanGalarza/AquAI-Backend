import asyncio
from collections import Counter
from uuid import UUID

from aquai.application.contracts.documents import PreparedDocument
from aquai.application.interfaces.documents import DocumentCommands, DocumentProcessor
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.Services.indexSourceService import IndexSourceService
from aquai.application.utils.background import complete_write


class ProcessDocumentService:
    #
    # Guarda los puertos de consultas, comandos, procesamiento e indexación que usa el servicio.
    #
    def __init__(
        self,
        queries: SourceQueries,
        documents: DocumentCommands,
        processor: DocumentProcessor,
        indexing: IndexSourceService,
    ) -> None:
        self._queries = queries
        self._documents = documents
        self._processor = processor
        self._indexing = indexing

    #
    # Procesa una fuente documental: extrae su contenido, lo guarda y lo indexa.
    #
    async def process(self, source_id: UUID, *, reprocess: bool = False) -> None:
        source = await asyncio.to_thread(self._queries.get, source_id, include_content=True)
        prepared = await self._processor.prepare(source["original_filename"], source["file_content"])

        saved = await complete_write(
            self._documents.save_prepared,
            source_id,
            prepared,
            _preparation_report(prepared, reprocess),
            source["content_hash"],
        )
        if saved:
            await self.reindex(source_id)

    #
    # Reconstruye el índice de una fuente documental y registra su estado.
    #
    async def reindex(self, source_id: UUID) -> None:
        await self._indexing.reindex(source_id)


#
# Construye el informe de preparación con avisos, recuentos y la calidad de la extracción.
#
def _preparation_report(prepared: PreparedDocument, reprocess: bool) -> dict:
    kinds = Counter(chunk.kind for chunk in prepared.chunks)
    report = {
        "summary": f"{len(prepared.chunks)} fragmentos preparados.",
        "warnings": [item[:500] for item in prepared.warnings],
        "warning_count": len(prepared.warnings),
        "errors": [],
        "steps": ["original_saved", "document_converted", "chunks_prepared", "sql_persisted"],
        "page_count": prepared.page_count,
        "section_count": len(prepared.sections),
        "element_count": len(prepared.elements),
        "chunk_count": len(prepared.chunks),
        "chunk_kinds": dict(kinds),
        "extraction": prepared.details,
    }
    if reprocess:
        report["reprocessed"] = True

    return report
