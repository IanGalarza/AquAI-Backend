from aquai.application.Services.retrieveDatasetsService import RetrieveDatasetsService
from aquai.bootstrap import initialize_application
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider
from aquai.infrastructure.providers.llm.settings import get_llm_settings
from aquai.infrastructure.providers.semanticIndex import QdrantSemanticIndex
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


#
# Crea el servicio que recupera datasets mediante el índice semántico y las consultas de fuentes.
#
def create_dataset_retriever() -> RetrieveDatasetsService:
    initialize_application()
    models = get_llm_settings()
    queries = SqlAlchemySourceQueries(get_db_context())
    index = QdrantSemanticIndex(
        get_tabular_settings(),
        embeddings=OllamaEmbeddingProvider(models.ollama, models.embedding),
        queries=queries,
    )

    return RetrieveDatasetsService(index, queries)
