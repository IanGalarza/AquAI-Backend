from aquai.application.Services.retrieveDatasetsService import RetrieveDatasetsService
from aquai.bootstrap import initialize_application
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.semanticIndex import QdrantSemanticIndex
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


def create_dataset_retriever() -> RetrieveDatasetsService:
    initialize_application()
    queries = SqlAlchemySourceQueries(get_db_context())
    index = QdrantSemanticIndex(get_tabular_settings(), queries=queries)

    return RetrieveDatasetsService(index, queries)
