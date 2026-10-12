"""Índice recuperable de descripciones; los registros permanecen en SQL."""

import asyncio
import math
from hashlib import sha256
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from weakref import WeakKeyDictionary

from qdrant_client import QdrantClient, models

from aquai.application.interfaces.sources import SourceQueries
from aquai.application.errors import SemanticIndexError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider, model_identity


_collection_locks: WeakKeyDictionary = WeakKeyDictionary()


class QdrantSemanticIndex:
    def __init__(
        self,
        settings: TabularSettings,
        *,
        client: QdrantClient | None = None,
        embeddings: OllamaEmbeddingProvider | None = None,
        queries: SourceQueries | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or QdrantClient(url=settings.qdrant_url, timeout=math.ceil(settings.index_timeout_seconds))
        self._embeddings = embeddings or OllamaEmbeddingProvider(settings)
        self._queries = queries
        self._embedding_model = model_identity(settings.embedding_model)
        self._vector_name = "embedding_" + sha256(self._embedding_model.encode()).hexdigest()[:24]

    async def index_source(self, source: dict) -> None:
        datasets = source.get("datasets", [])
        if not datasets:
            return
        if len(datasets) > self._settings.max_datasets:
            raise SemanticIndexError("La fuente excede la cantidad máxima de datasets indexables.")
        documents = [self._document(dataset) for dataset in datasets]
        vectors = await self._embeddings.embed([self._embedding_text(text, query=False) for text, _ in documents])
        dimension = self._validate_vectors(vectors, expected=len(datasets))
        await self._ensure_collection(dimension)
        points = []
        for dataset, vector, (_, truncated) in zip(datasets, vectors, documents, strict=True):
            dataset_id = UUID(str(dataset["id"]))
            version = int(dataset.get("version", 1))
            point_id = dataset.get("semantic_point_id") or uuid5(
                NAMESPACE_URL, f"aquai:dataset:{dataset_id}:version:{version}",
            )
            payload = {
                "source_id": str(source["id"]),
                "dataset_id": str(dataset_id),
                "content_type": "dataset_description",
                "version": version,
                "is_active": bool(source.get("is_active", False)) and bool(dataset.get("is_active", True)),
                "embedding_model": self._embedding_model,
                "description": dataset.get("semantic_description") or dataset.get("description", ""),
                "dataset_name": dataset.get("name", ""),
                "column_names": [column["original_name"] for column in dataset.get("columns_metadata", [])],
                "embedding_text_truncated": truncated,
            }
            points.append(models.PointStruct(id=str(UUID(str(point_id))), vector={self._vector_name: vector}, payload=payload))
        await self._call(self._client.upsert, collection_name=self._settings.qdrant_collection,
                         points=points, wait=True)

    async def set_source_active(self, source_id: UUID, active: bool) -> None:
        if not await self._call(self._client.collection_exists, self._settings.qdrant_collection):
            return
        await self._call(
            self._client.set_payload,
            collection_name=self._settings.qdrant_collection,
            payload={"is_active": active},
            points=models.Filter(must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=str(source_id)))]),
            wait=True,
        )

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._queries is None:
            raise SemanticIndexError("La búsqueda requiere verificar los datasets vigentes en PostgreSQL.")
        if not isinstance(query, str) or not query.strip() or len(query) > 6000:
            raise ValueError("La consulta semántica debe tener entre 1 y 6000 caracteres.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("El límite de búsqueda debe estar entre 1 y 100.")
        if not await self._call(self._client.collection_exists, self._settings.qdrant_collection):
            return []
        vectors = await self._embeddings.embed([self._embedding_text(query.strip(), query=True)])
        await self._ensure_collection(self._validate_vectors(vectors, expected=1))
        response = await self._call(
            self._client.query_points,
            collection_name=self._settings.qdrant_collection,
            query=vectors[0],
            using=self._vector_name,
            query_filter=models.Filter(must=[
                models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
                models.FieldCondition(key="content_type", match=models.MatchValue(value="dataset_description")),
                models.FieldCondition(key="embedding_model", match=models.MatchValue(value=self._embedding_model)),
            ]),
            limit=min(1000, limit * 4),
            with_payload=True,
            with_vectors=False,
        )
        candidates = []
        for point in response.points:
            payload = point.payload or {}
            try:
                dataset_id = UUID(str(payload["dataset_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append((dataset_id, point, payload))
        usable = await asyncio.to_thread(self._queries.usable_dataset_ids, [item[0] for item in candidates])
        usable_ids = {str(dataset_id) for dataset_id in usable}
        result: list[dict[str, Any]] = []
        seen: set[UUID] = set()
        for dataset_id, point, payload in candidates:
            if str(dataset_id) not in usable_ids or dataset_id in seen:
                continue
            result.append({**payload, "semantic_point_id": str(point.id), "score": float(point.score)})
            seen.add(dataset_id)
            if len(result) == limit:
                break
        return result

    async def _ensure_collection(self, dimension: int) -> None:
        loop = asyncio.get_running_loop()
        locks = _collection_locks.setdefault(loop, {})
        key = (self._settings.qdrant_url, self._settings.qdrant_collection)
        async with locks.setdefault(key, asyncio.Lock()):
            exists = await self._call(self._client.collection_exists, self._settings.qdrant_collection)
            if not exists:
                try:
                    await self._call(
                        self._client.create_collection,
                        collection_name=self._settings.qdrant_collection,
                        vectors_config={self._vector_name: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
                    )
                except SemanticIndexError:
                    if not await self._call(self._client.collection_exists, self._settings.qdrant_collection):
                        raise
            collection = await self._call(self._client.get_collection, self._settings.qdrant_collection)
            config = collection.config.params.vectors
            if not isinstance(config, dict) or set(config) != {self._vector_name}:
                raise SemanticIndexError("La colección Qdrant pertenece a otro modelo de embeddings; requiere un índice nuevo.")
            if config[self._vector_name].size != dimension:
                raise SemanticIndexError(
                    f"La dimensión del embedding ({dimension}) no coincide con la colección Qdrant "
                    f"({config[self._vector_name].size}); requiere un índice nuevo.",
                )
            if config[self._vector_name].distance != models.Distance.COSINE:
                raise SemanticIndexError("La colección Qdrant requiere distancia coseno.")

    async def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self._settings.index_timeout_seconds,
            )
        except asyncio.CancelledError:
            # Un thread no se cancela junto con el worker. Esperar evita que una
            # escritura pendiente sobrescriba el siguiente intento.
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise
        except TimeoutError as exc:
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise SemanticIndexError("Qdrant superó el tiempo máximo de respuesta.") from exc
        except Exception as exc:
            raise SemanticIndexError("No se pudo completar la operación de índice en Qdrant.") from exc

    def _embedding_text(self, text: str, *, query: bool) -> str:
        if self._settings.embedding_model.split(":", 1)[0] == "nomic-embed-text":
            return ("search_query: " if query else "search_document: ") + text
        return text

    @staticmethod
    def _document(dataset: dict) -> tuple[str, bool]:
        description = dataset.get("semantic_description") or dataset.get("description") or "Tabla preparada."
        text = f"{dataset.get('name', '')}\n{description}"[:4500]
        truncated = len(f"{dataset.get('name', '')}\n{description}") > len(text)
        for column in dataset.get("columns_metadata", []):
            line = "\n" + str(column["original_name"])
            if column.get("meaning"):
                line += ": " + str(column["meaning"])
            if column.get("unit"):
                line += " [" + str(column["unit"]) + "]"
            else:
                line += " [unidad no documentada]"
            remaining = 6000 - len(text)
            if len(line) > remaining:
                text += line[:remaining]
                truncated = True
                break
            text += line
        return text.strip(), truncated

    @staticmethod
    def _validate_vectors(vectors: list[list[float]], *, expected: int) -> int:
        if len(vectors) != expected or not vectors or not vectors[0]:
            raise SemanticIndexError("El proveedor no devolvió un vector por descripción.")
        dimension = len(vectors[0])
        if any(len(vector) != dimension or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                             or not math.isfinite(value) for value in vector)
               or not any(value != 0 for value in vector) for vector in vectors):
            raise SemanticIndexError("Los embeddings tienen dimensiones o componentes inválidos.")
        return dimension
