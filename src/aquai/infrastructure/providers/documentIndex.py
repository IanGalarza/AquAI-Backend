import asyncio
import math
from typing import Any
from uuid import UUID

from qdrant_client import QdrantClient, models

from aquai.application.errors import LLMProviderError, SemanticIndexError
from aquai.application.interfaces.documents import DocumentQueries
from aquai.domain.enums.documentContent import DocumentChunkKind
from aquai.infrastructure.config.documents import DocumentSettings
from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider
from aquai.infrastructure.providers.semanticIndex import (
    QdrantSemanticIndex,
    _field_equals,
)

_CONTENT_TYPE = "document_chunk"
_SEARCHABLE_KINDS = (
    DocumentChunkKind.TEXT.value,
    DocumentChunkKind.TABLE.value,
    DocumentChunkKind.FIGURE.value,
)
_PAYLOAD_INDEXES = {
    "source_id": models.PayloadSchemaType.KEYWORD,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
    "is_active": models.PayloadSchemaType.BOOL,
}


class QdrantDocumentIndex(QdrantSemanticIndex):
    #
    # Guarda la configuración, los embeddings y las consultas del índice documental.
    #
    def __init__(
        self,
        settings: DocumentSettings,
        *,
        embeddings: OllamaEmbeddingProvider,
        queries: DocumentQueries,
        client: QdrantClient | None = None,
    ) -> None:
        super().__init__(settings, embeddings=embeddings, client=client)
        self._documents = queries

    #
    # Indexa los fragmentos vigentes de una fuente y elimina los de versiones anteriores.
    #
    async def index_source(self, source: dict) -> None:
        source_id = source["id"]
        content = await asyncio.to_thread(self._documents.indexable, source_id)
        if content is None or not content["chunks"]:
            await self.delete_source(source_id)
            return

        if await self._is_complete(content):
            await self.set_source_active(source_id, bool(source["is_active"]))
            await self._delete_stale(source_id, content["document_id"])
            return

        chunks = content["chunks"]
        batch_size = self._settings.index_batch_size
        for offset in range(0, len(chunks), batch_size):
            await self._index_batch(source, content, chunks[offset : offset + batch_size])

        await self._delete_stale(source_id, content["document_id"])

    #
    # Elimina del índice todos los fragmentos de una fuente.
    #
    async def delete_source(self, source_id: UUID) -> None:
        if not await self._collection_exists():
            return

        await self._call(
            self._client.delete,
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_field_equals("source_id", str(source_id))])
            ),
            wait=True,
        )

    #
    # Busca los fragmentos más cercanos a una consulta y los verifica en PostgreSQL.
    #
    async def search(
        self, query: str, limit: int = 5, *, kinds: tuple[str, ...] | None = None
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or len(query) > 6000:
            raise ValueError("La consulta semántica debe tener entre 1 y 6000 caracteres.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("El límite de búsqueda debe estar entre 1 y 100.")
        if not await self._collection_exists():
            return []

        vectors = await self._embed([self._embedding_text(query.strip(), query=True)])
        await self._ensure_collection(self._validate_vectors(vectors, expected=1))

        return await self._verified_matches(vectors[0], limit, kinds or _SEARCHABLE_KINDS)

    #
    # Calcula los embeddings e informa los fallos del proveedor como fallos del índice.
    #
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._embeddings.embed(texts)
        except LLMProviderError as error:
            raise SemanticIndexError(f"Embeddings no disponibles: {error}") from error

    #
    # Indica si el índice ya tiene todos los fragmentos del documento con el modelo vigente.
    #
    async def _is_complete(self, content: dict) -> bool:
        if not await self._collection_exists():
            return False

        result = await self._call(
            self._client.count,
            collection_name=self._settings.qdrant_collection,
            count_filter=models.Filter(
                must=[
                    _field_equals("document_id", str(content["document_id"])),
                    _field_equals("embedding_model", self._embedding_model),
                ]
            ),
            exact=True,
        )
        return result.count == len(content["chunks"])

    #
    # Embebe un lote de fragmentos y lo guarda en Qdrant.
    #
    async def _index_batch(self, source: dict, content: dict, chunks: list[dict]) -> None:
        texts = [self._chunk_text(content, chunk) for chunk in chunks]
        vectors = await self._embed([self._embedding_text(text, query=False) for text in texts])
        await self._ensure_collection(self._validate_vectors(vectors, expected=len(chunks)))

        await self._call(
            self._client.upsert,
            collection_name=self._settings.qdrant_collection,
            points=[
                self._chunk_point(source, content, chunk, vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    #
    # Arma el texto a embeber con el título y la ruta de la sección que dan contexto al fragmento.
    #
    @staticmethod
    def _chunk_text(content: dict, chunk: dict) -> str:
        context = [content["title"], *chunk["section_path"], chunk["text"]]
        return "\n".join(dict.fromkeys(part for part in context if part))

    #
    # Construye el punto de un fragmento; el texto vigente siempre se lee de PostgreSQL.
    #
    def _chunk_point(
        self, source: dict, content: dict, chunk: dict, vector: list[float]
    ) -> models.PointStruct:
        return models.PointStruct(
            id=str(chunk["id"]),
            vector={self._vector_name: vector},
            payload={
                "source_id": str(source["id"]),
                "document_id": str(content["document_id"]),
                "content_type": _CONTENT_TYPE,
                "kind": str(chunk["kind"]),
                "is_active": bool(source["is_active"]),
                "embedding_model": self._embedding_model,
                "position": chunk["position"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_path": chunk["section_path"],
                "title": content["title"],
                "language": content["language"],
                "file_type": source["file_type"],
            },
        )

    #
    # Elimina los puntos de la fuente que pertenecen a un procesamiento anterior.
    #
    async def _delete_stale(self, source_id: UUID, document_id: UUID) -> None:
        await self._call(
            self._client.delete,
            collection_name=self._settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[_field_equals("source_id", str(source_id))],
                    must_not=[_field_equals("document_id", str(document_id))],
                )
            ),
            wait=True,
        )

    #
    # Pagina los resultados del índice hasta reunir los fragmentos vigentes.
    #
    async def _verified_matches(
        self, vector: list[float], limit: int, kinds: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        page_size = limit * 4
        offset = 0

        while len(matches) < limit:
            response = await self._chunk_page(vector, page_size, offset, kinds)
            points = [point for point in response.points if math.isfinite(point.score)]
            verified = await asyncio.to_thread(
                self._documents.verified_chunks, [UUID(str(point.id)) for point in points]
            )
            for point in points:
                chunk = verified.get(UUID(str(point.id)))
                if chunk is not None and len(matches) < limit:
                    matches.append({**chunk, "score": float(point.score)})

            if len(response.points) < page_size:
                break

            offset += len(response.points)

        return matches

    #
    # Consulta una página de fragmentos activos del modelo de embeddings vigente.
    #
    async def _chunk_page(
        self, vector: list[float], page_size: int, offset: int, kinds: tuple[str, ...]
    ) -> Any:
        return await self._call(
            self._client.query_points,
            collection_name=self._settings.qdrant_collection,
            query=vector,
            using=self._vector_name,
            query_filter=models.Filter(
                must=[
                    _field_equals("content_type", _CONTENT_TYPE),
                    _field_equals("embedding_model", self._embedding_model),
                    _field_equals("is_active", True),
                    models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))),
                ]
            ),
            limit=page_size,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

    #
    # Crea la colección con índices sobre los campos que filtran las búsquedas y borrados.
    #
    async def _create_collection(self, dimension: int) -> None:
        await super()._create_collection(dimension)
        for field, schema in _PAYLOAD_INDEXES.items():
            await self._call(
                self._client.create_payload_index,
                collection_name=self._settings.qdrant_collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
