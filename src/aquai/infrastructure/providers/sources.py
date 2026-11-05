from dataclasses import dataclass

from aquai.application.interfaces.documents import (
    DocumentCommands,
    DocumentIndex,
    DocumentProcessor,
    DocumentQueries,
)
from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.interfaces.tabular import (
    SemanticIndex,
    SemanticProvider,
    TabularProcessor,
)
from aquai.application.Rules.documents import DOCUMENT_FILE_TYPES
from aquai.application.Services.indexSourceService import IndexSourceService
from aquai.application.Services.processDocumentService import ProcessDocumentService
from aquai.application.Services.processSourceService import ProcessSourceService
from aquai.application.Services.sourceDetailService import SourceDetailService
from aquai.application.useCases.SourceOperation.Commands.activateSourceHandler import (
    ActivateSourceHandler,
)
from aquai.application.useCases.SourceOperation.Commands.deleteSourceHandler import (
    DeleteSourceHandler,
)
from aquai.application.useCases.SourceOperation.Commands.replaceSourceFileHandler import (
    ReplaceSourceFileHandler,
)
from aquai.application.useCases.SourceOperation.Commands.uploadSourceHandler import (
    UploadSourceHandler,
)
from aquai.application.useCases.SourceOperation.Queries.getSourceHandler import (
    GetSourceHandler,
)
from aquai.application.useCases.SourceOperation.Queries.listSourcesHandler import (
    ListSourcesHandler,
)
from aquai.bootstrap import initialize_application
from aquai.infrastructure.commands.documents import (
    DOCUMENT_SCOPE,
    SqlAlchemyDocumentCommands,
)
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.documents import get_document_settings
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.queries.documents import SqlAlchemyDocumentQueries
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from aquai.infrastructure.workers.sourceWorker import SourceWorker


@dataclass(frozen=True)
class DocumentPorts:
    source_commands: SourceCommands
    commands: DocumentCommands
    queries: DocumentQueries
    processor: DocumentProcessor
    index: DocumentIndex
    max_upload_bytes: int


@dataclass(frozen=True)
class SourceRuntime:
    upload: UploadSourceHandler
    list: ListSourcesHandler
    get: GetSourceHandler
    activate: ActivateSourceHandler
    worker: SourceWorker
    document_worker: SourceWorker | None = None
    delete: DeleteSourceHandler | None = None
    replace_file: ReplaceSourceFileHandler | None = None

    #
    # Devuelve los workers en funcionamiento: tabular y, si está configurado, documental.
    #
    @property
    def workers(self) -> tuple[SourceWorker, ...]:
        if self.document_worker is None:
            return (self.worker,)

        return (self.worker, self.document_worker)

    #
    # Devuelve el worker que atiende un tipo de archivo.
    #
    def worker_for(self, file_type: str) -> SourceWorker:
        if file_type in DOCUMENT_FILE_TYPES and self.document_worker is not None:
            return self.document_worker

        return self.worker


#
# Ensambla los manejadores de fuentes y el worker a partir de sus dependencias.
#
def build_source_runtime(
    commands: SourceCommands,
    queries: SourceQueries,
    processor: TabularProcessor,
    semantics: SemanticProvider,
    index: SemanticIndex,
    *,
    max_upload_bytes: int,
    poll_seconds: float = 1.0,
    documents: DocumentPorts | None = None,
) -> SourceRuntime:
    processing = ProcessSourceService(commands, queries, processor, semantics, index)
    details = SourceDetailService(queries, documents.queries if documents else None)
    worker = SourceWorker(commands, processing, poll_seconds=poll_seconds)
    document_worker = _document_worker(queries, documents, poll_seconds)
    wake = _wake_workers(worker, document_worker)

    return SourceRuntime(
        upload=UploadSourceHandler(
            commands,
            processor,
            max_upload_bytes=max(max_upload_bytes, documents.max_upload_bytes)
            if documents
            else max_upload_bytes,
            notify=wake,
            documents=documents.processor if documents else None,
        ),
        list=ListSourcesHandler(queries),
        get=GetSourceHandler(details),
        activate=ActivateSourceHandler(commands, details, notify=wake),
        worker=worker,
        document_worker=document_worker,
        delete=DeleteSourceHandler(
            documents.commands, documents.source_commands, documents.index, notify=wake
        )
        if documents
        else None,
        replace_file=ReplaceSourceFileHandler(
            documents.commands, documents.processor, notify=wake
        )
        if documents
        else None,
    )


#
# Crea el worker documental: su cola y su bloqueo no interfieren con los del tabular.
#
def _document_worker(
    queries: SourceQueries, documents: DocumentPorts | None, poll_seconds: float
) -> SourceWorker | None:
    if documents is None:
        return None

    indexing = IndexSourceService(documents.source_commands, queries, documents.index)
    processing = ProcessDocumentService(
        queries, documents.commands, documents.processor, indexing
    )
    return SourceWorker(documents.source_commands, processing, poll_seconds=poll_seconds)


#
# Devuelve la función que despierta a los workers disponibles.
#
def _wake_workers(worker: SourceWorker, document_worker: SourceWorker | None):
    if document_worker is None:
        return worker.wake

    def wake() -> None:
        worker.wake()
        document_worker.wake()

    return wake


#
# Crea el entorno de ejecución de fuentes con sus procesadores, Ollama, Qdrant y la base de datos.
#
def create_source_runtime() -> SourceRuntime:
    from aquai.infrastructure.processing.documents.processor import DoclingDocumentProcessor
    from aquai.infrastructure.processing.tabular.processor import SimpleTabularProcessor
    from aquai.infrastructure.providers.documentIndex import QdrantDocumentIndex
    from aquai.infrastructure.providers.llm.ollama import (
        OllamaEmbeddingProvider,
        OllamaSemanticProvider,
    )
    from aquai.infrastructure.providers.llm.settings import get_llm_settings
    from aquai.infrastructure.providers.semanticIndex import QdrantSemanticIndex

    initialize_application()
    context = get_db_context()
    settings = get_tabular_settings()
    models = get_llm_settings()
    queries = SqlAlchemySourceQueries(context)
    index = QdrantSemanticIndex(
        settings,
        embeddings=OllamaEmbeddingProvider(models.ollama, models.embedding),
        queries=queries,
    )

    document_settings = get_document_settings()
    document_queries = SqlAlchemyDocumentQueries(context)
    documents = DocumentPorts(
        source_commands=SqlAlchemySourceCommands(context, scope=DOCUMENT_SCOPE),
        commands=SqlAlchemyDocumentCommands(context),
        queries=document_queries,
        processor=DoclingDocumentProcessor(document_settings),
        index=QdrantDocumentIndex(
            document_settings,
            embeddings=OllamaEmbeddingProvider(models.ollama, models.document_embedding),
            queries=document_queries,
        ),
        max_upload_bytes=document_settings.max_upload_bytes,
    )

    return build_source_runtime(
        SqlAlchemySourceCommands(context),
        queries,
        SimpleTabularProcessor(settings),
        OllamaSemanticProvider(settings, models.ollama, models.tabular),
        index,
        max_upload_bytes=settings.max_upload_bytes,
        documents=documents,
    )
