import asyncio
import math
from hashlib import sha256
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5
from weakref import WeakKeyDictionary

from qdrant_client import QdrantClient, models

from aquai.application.errors import SemanticIndexError
from aquai.application.interfaces.sources import SourceQueries
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import (
    OllamaEmbeddingProvider,
    model_identity,
)

_collection_locks: WeakKeyDictionary = WeakKeyDictionary()


def _field_equals(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_valid_vector(vector: list, dimension: int) -> bool:
    if len(vector) != dimension:
        return False
    if not all(_is_finite_number(value) for value in vector):
        return False

    return any(value != 0 for value in vector)


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
        self._client = client or QdrantClient(
            url=settings.qdrant_url, timeout=math.ceil(settings.index_timeout_seconds)
        )
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
        vectors = await self._embeddings.embed(
            [self._embedding_text(text, query=False) for text, _ in documents]
        )
        dimension = self._validate_vectors(vectors, expected=len(datasets))
        await self._ensure_collection(dimension)

        points = []
        for dataset, vector, (_, truncated) in zip(datasets, vectors, documents, strict=True):
            points.append(self._point(source, dataset, vector, truncated))

        await self._call(
            self._client.upsert,
            collection_name=self._settings.qdrant_collection,
            points=points,
            wait=True,
        )

    async def set_source_active(self, source_id: UUID, active: bool) -> None:
        if not await self._collection_exists():
            return

        await self._call(
            self._client.set_payload,
            collection_name=self._settings.qdrant_collection,
            payload={"is_active": active},
            points=models.Filter(must=[_field_equals("source_id", str(source_id))]),
            wait=True,
        )

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._queries is None:
            raise SemanticIndexError(
                "La búsqueda requiere verificar los datasets vigentes en PostgreSQL."
            )
        if not isinstance(query, str) or not query.strip() or len(query) > 6000:
            raise ValueError("La consulta semántica debe tener entre 1 y 6000 caracteres.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("El límite de búsqueda debe estar entre 1 y 100.")
        if not await self._collection_exists():
            return []

        vectors = await self._embeddings.embed([self._embedding_text(query.strip(), query=True)])
        await self._ensure_collection(self._validate_vectors(vectors, expected=1))

        return await self._usable_matches(vectors[0], limit)

    async def _usable_matches(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        seen: set[UUID] = set()
        page_size = limit * 4
        offset = 0
        while len(matches) < limit:
            response = await self._query_page(query_vector, page_size, offset)
            candidates = self._valid_candidates(response.points)
            usable_pairs = await self._usable_pairs(candidates)

            for dataset_id, point, payload in candidates:
                if (dataset_id, payload["version"]) not in usable_pairs or dataset_id in seen:
                    continue

                matches.append(
                    {**payload, "semantic_point_id": str(point.id), "score": float(point.score)}
                )
                seen.add(dataset_id)
                if len(matches) == limit:
                    break

            if len(response.points) < page_size:
                break

            offset += len(response.points)

        return matches

    async def _query_page(self, query_vector: list[float], page_size: int, offset: int) -> Any:
        return await self._call(
            self._client.query_points,
            collection_name=self._settings.qdrant_collection,
            query=query_vector,
            using=self._vector_name,
            query_filter=models.Filter(
                must=[
                    _field_equals("content_type", "dataset_description"),
                    _field_equals("embedding_model", self._embedding_model),
                ]
            ),
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

    @classmethod
    def _valid_candidates(cls, points: list) -> list[tuple[UUID, Any, dict]]:
        candidates = []
        for point in points:
            payload = point.payload or {}
            dataset_id = cls._valid_dataset_id(point, payload)
            if dataset_id is None:
                continue

            version = payload.get("version")
            if type(version) is not int or version < 1 or not math.isfinite(point.score):
                continue

            candidates.append((dataset_id, point, payload))

        return candidates

    @staticmethod
    def _valid_dataset_id(point: Any, payload: dict) -> UUID | None:
        try:
            dataset_id = UUID(str(payload["dataset_id"]))
            UUID(str(payload["source_id"]))
            UUID(str(point.id))
        except (KeyError, TypeError, ValueError):
            return None

        return dataset_id

    async def _usable_pairs(self, candidates: list) -> set[tuple[UUID, int]]:
        usable_pairs = set()
        candidate_versions = {payload["version"] for _, _, payload in candidates}
        for version in candidate_versions:
            dataset_ids = [
                dataset_id for dataset_id, _, payload in candidates if payload["version"] == version
            ]
            usable = await asyncio.to_thread(
                self._queries.usable_dataset_ids,
                dataset_ids,
                versions={dataset_id: version for dataset_id in dataset_ids},
            )
            usable_pairs.update((dataset_id, version) for dataset_id in usable)

        return usable_pairs

    def _point(
        self, source: dict, dataset: dict, vector: list[float], truncated: bool
    ) -> models.PointStruct:
        dataset_id = UUID(str(dataset["id"]))
        version = int(dataset.get("version", 1))
        point_id = dataset.get("semantic_point_id") or uuid5(
            NAMESPACE_URL,
            f"aquai:dataset:{dataset_id}:version:{version}",
        )
        payload = self._point_payload(source, dataset, dataset_id, version, truncated)

        return models.PointStruct(
            id=str(UUID(str(point_id))), vector={self._vector_name: vector}, payload=payload
        )

    def _point_payload(
        self, source: dict, dataset: dict, dataset_id: UUID, version: int, truncated: bool
    ) -> dict:
        is_active = bool(source.get("is_active", False)) and bool(dataset.get("is_active", True))
        description = dataset.get("semantic_description") or dataset.get("description", "")

        return {
            "source_id": str(source["id"]),
            "dataset_id": str(dataset_id),
            "content_type": "dataset_description",
            "version": version,
            "is_active": is_active,
            "embedding_model": self._embedding_model,
            "description": description,
            "dataset_name": dataset.get("name", ""),
            "column_names": [
                column["original_name"] for column in dataset.get("columns_metadata", [])
            ],
            "embedding_text_truncated": truncated,
        }

    async def _ensure_collection(self, dimension: int) -> None:
        async with self._collection_lock():
            if not await self._collection_exists():
                await self._create_collection(dimension)

            collection = await self._call(
                self._client.get_collection, self._settings.qdrant_collection
            )
            self._validate_collection_vectors(collection.config.params.vectors, dimension)

    def _collection_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        locks = _collection_locks.setdefault(loop, {})
        key = (self._settings.qdrant_url, self._settings.qdrant_collection)
        return locks.setdefault(key, asyncio.Lock())

    async def _collection_exists(self) -> bool:
        return await self._call(self._client.collection_exists, self._settings.qdrant_collection)

    async def _create_collection(self, dimension: int) -> None:
        try:
            await self._call(
                self._client.create_collection,
                collection_name=self._settings.qdrant_collection,
                vectors_config={
                    self._vector_name: models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    )
                },
            )
        except SemanticIndexError:
            created_concurrently = await self._collection_exists()
            if not created_concurrently:
                raise

    def _validate_collection_vectors(self, config: Any, dimension: int) -> None:
        if not isinstance(config, dict) or set(config) != {self._vector_name}:
            raise SemanticIndexError(
                "La colección Qdrant pertenece a otro modelo de embeddings; requiere un índice nuevo."
            )

        vector_params = config[self._vector_name]
        if vector_params.size != dimension:
            raise SemanticIndexError(
                f"La dimensión del embedding ({dimension}) no coincide con la colección Qdrant "
                f"({vector_params.size}); requiere un índice nuevo.",
            )
        if vector_params.distance != models.Distance.COSINE:
            raise SemanticIndexError("La colección Qdrant requiere distancia coseno.")

    async def _call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._settings.index_timeout_seconds,
            )
        except asyncio.CancelledError:
            await self._wait_for_thread(task)
            raise
        except TimeoutError as exc:
            await self._wait_for_thread(task)
            raise SemanticIndexError("Qdrant superó el tiempo máximo de respuesta.") from exc
        except Exception as exc:
            raise SemanticIndexError(
                "No se pudo completar la operación de índice en Qdrant."
            ) from exc

    @staticmethod
    async def _wait_for_thread(task: asyncio.Task) -> None:
        """Un thread no se cancela junto con el worker. Esperar evita que una
        escritura pendiente sobrescriba el siguiente intento."""
        try:
            await asyncio.shield(task)
        except Exception:
            pass

    def _embedding_text(self, text: str, *, query: bool) -> str:
        model_name = self._settings.embedding_model.split(":", 1)[0]
        if model_name != "nomic-embed-text":
            return text

        prefix = "search_query: " if query else "search_document: "
        return prefix + text

    @classmethod
    def _document(cls, dataset: dict) -> tuple[str, bool]:
        description = (
            dataset.get("semantic_description") or dataset.get("description") or "Tabla preparada."
        )
        heading = f"{dataset.get('name', '')}\n{description}"
        text = heading[:4500]
        truncated = len(heading) > len(text)

        for column in dataset.get("columns_metadata", []):
            line = cls._column_line(column)
            remaining = 6000 - len(text)
            if len(line) > remaining:
                text += line[:remaining]
                truncated = True
                break

            text += line

        return text.strip(), truncated

    @staticmethod
    def _column_line(column: dict) -> str:
        line = "\n" + str(column["original_name"])
        if column.get("meaning"):
            line += ": " + str(column["meaning"])
        if column.get("unit"):
            line += " [" + str(column["unit"]) + "]"
        else:
            line += " [unidad no documentada]"

        return line

    @staticmethod
    def _validate_vectors(vectors: list[list[float]], *, expected: int) -> int:
        if len(vectors) != expected or not vectors or not vectors[0]:
            raise SemanticIndexError("El proveedor no devolvió un vector por descripción.")

        dimension = len(vectors[0])
        if not all(_is_valid_vector(vector, dimension) for vector in vectors):
            raise SemanticIndexError("Los embeddings tienen dimensiones o componentes inválidos.")

        return dimension
