from functools import lru_cache

from aquai.infrastructure.config.documents import DocumentSettings


#
# Devuelve la configuración del procesamiento documental, cacheada.
#
@lru_cache
def get_document_settings() -> DocumentSettings:
    return DocumentSettings()


#
# Crea los servicios que recuperan evidencia documental y la contrastan con los resultados.
#
def create_document_services():
    from aquai.application.Services.contrastReferencesService import ContrastReferencesService
    from aquai.application.Services.retrieveDocumentsService import RetrieveDocumentsService
    from aquai.bootstrap import initialize_application
    from aquai.infrastructure.providers.database import get_db_context
    from aquai.infrastructure.providers.documentIndex import QdrantDocumentIndex
    from aquai.infrastructure.providers.documentLexicalIndex import Bm25DocumentIndex
    from aquai.infrastructure.providers.llm.documents import OllamaDocumentEvidenceProvider
    from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider
    from aquai.infrastructure.providers.llm.settings import get_llm_settings
    from aquai.infrastructure.queries.documents import SqlAlchemyDocumentQueries

    initialize_application()
    settings = get_document_settings()
    models = get_llm_settings()
    queries = SqlAlchemyDocumentQueries(get_db_context())
    evidence = OllamaDocumentEvidenceProvider(models.ollama, models.evidence)
    index = QdrantDocumentIndex(
        settings,
        embeddings=OllamaEmbeddingProvider(models.ollama, models.document_embedding),
        queries=queries,
    )
    documents = RetrieveDocumentsService(
        index,
        queries,
        evidence,
        lexical=Bm25DocumentIndex(queries),
        search_limit=settings.search_limit,
        candidates=settings.search_candidates,
        fragments=settings.search_fragments,
        per_document=settings.search_fragments_per_document,
        candidate_chars=settings.search_candidate_chars,
    )

    return documents, ContrastReferencesService(evidence)
