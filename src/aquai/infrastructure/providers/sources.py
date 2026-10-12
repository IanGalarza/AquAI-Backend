"""Composición de handlers y worker de fuentes."""

from dataclasses import dataclass

from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.services.sourceReviews import SourceReviewService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.application.useCases.sources import (
    GetSourceHandler,
    ListSourcesHandler,
    ProcessSourceHandler,
    ReviewSourceHandler,
    UploadSourceHandler,
)
from aquai.bootstrap import initialize_application
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.tabular import get_tabular_evaluator, get_tabular_settings
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries
from aquai.infrastructure.workers.sourceWorker import SourceWorker


@dataclass(frozen=True)
class SourceRuntime:
    """Dependencias compartidas; cada operación se ejecuta en su propio handler."""

    upload: UploadSourceHandler
    list: ListSourcesHandler
    get: GetSourceHandler
    review: ReviewSourceHandler
    worker: SourceWorker


def build_source_runtime(
    commands: SourceCommands,
    queries: SourceQueries,
    evaluator: TabularEvaluationService,
    *,
    max_upload_bytes: int,
    reviews: SourceReviewService | None = None,
    poll_seconds: float = 1.0,
) -> SourceRuntime:
    reviews = reviews or SourceReviewService()
    processor = ProcessSourceHandler(commands, queries, evaluator, reviews=reviews)
    worker = SourceWorker(commands, processor, poll_seconds=poll_seconds)
    return SourceRuntime(
        upload=UploadSourceHandler(
            commands,
            queries,
            evaluator,
            max_upload_bytes=max_upload_bytes,
            notify=worker.wake,
        ),
        list=ListSourcesHandler(queries),
        get=GetSourceHandler(queries, reviews=reviews),
        review=ReviewSourceHandler(commands, reviews=reviews, notify=worker.wake),
        worker=worker,
    )


def create_source_runtime() -> SourceRuntime:
    initialize_application()
    context = get_db_context()
    return build_source_runtime(
        SqlAlchemySourceCommands(context),
        SqlAlchemySourceQueries(context),
        get_tabular_evaluator(),
        max_upload_bytes=get_tabular_settings().max_upload_bytes,
    )
