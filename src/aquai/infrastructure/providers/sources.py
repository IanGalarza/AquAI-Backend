"""Composición de handlers, adaptadores y worker de fuentes."""
from dataclasses import dataclass

from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.interfaces.tabular import SemanticIndex, SemanticProvider, TabularProcessor
from aquai.application.useCases.sources import (
    ActivateSourceHandler, GetSourceHandler, ListSourcesHandler,
    ProcessSourceHandler, UploadSourceHandler,
)
from aquai.bootstrap import initialize_application
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from aquai.infrastructure.workers.sourceWorker import SourceWorker


@dataclass(frozen=True)
class SourceRuntime:
    upload: UploadSourceHandler
    list: ListSourcesHandler
    get: GetSourceHandler
    activate: ActivateSourceHandler
    worker: SourceWorker


def build_source_runtime(commands: SourceCommands, queries: SourceQueries,
                         processor: TabularProcessor, semantics: SemanticProvider,
                         index: SemanticIndex, *, max_upload_bytes: int,
                         poll_seconds: float = 1.0) -> SourceRuntime:
    handler = ProcessSourceHandler(commands, queries, processor, semantics, index)
    worker = SourceWorker(commands, handler, poll_seconds=poll_seconds)
    return SourceRuntime(
        upload=UploadSourceHandler(
            commands, queries, processor, max_upload_bytes=max_upload_bytes, notify=worker.wake,
        ),
        list=ListSourcesHandler(queries), get=GetSourceHandler(queries),
        activate=ActivateSourceHandler(commands, queries, notify=worker.wake),
        worker=worker,
    )


def create_source_runtime() -> SourceRuntime:
    from aquai.infrastructure.processing.tabular.processor import SimpleTabularProcessor
    from aquai.infrastructure.providers.llm.ollama import (
        OllamaEmbeddingProvider, OllamaSemanticProvider,
    )
    from aquai.infrastructure.providers.semanticIndex import QdrantSemanticIndex

    initialize_application()
    context = get_db_context()
    settings = get_tabular_settings()
    queries = SqlAlchemySourceQueries(context)
    index = QdrantSemanticIndex(settings, embeddings=OllamaEmbeddingProvider(settings), queries=queries)
    return build_source_runtime(
        SqlAlchemySourceCommands(context), queries, SimpleTabularProcessor(settings),
        OllamaSemanticProvider(settings), index, max_upload_bytes=settings.max_upload_bytes,
    )
