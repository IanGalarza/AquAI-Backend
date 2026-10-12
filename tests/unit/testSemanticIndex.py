import json
import unittest
from copy import deepcopy
from datetime import date
from decimal import Decimal
from hashlib import sha256
from uuid import uuid4

import httpx
from qdrant_client import QdrantClient, models

from aquai.application.contracts.ingestion import PreparedTable
from aquai.application.errors import LLMProviderError
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import OllamaEmbeddingProvider, OllamaSemanticProvider, model_identity
from aquai.infrastructure.providers.semanticIndex import QdrantSemanticIndex, SemanticIndexError


class _Queries:
    def __init__(self, usable):
        self.usable = set(usable)
        self.requested = []

    def usable_dataset_ids(self, dataset_ids):
        self.requested = dataset_ids
        return self.usable.intersection(dataset_ids)


class SemanticProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_contract_preserves_types_units_names_and_bounded_sample(self):
        settings = TabularSettings(_env_file=None, sample_rows=2)
        table = PreparedTable(
            name="Mediciones",
            source_locator={"file_type": "csv", "sheet": "Mediciones", "range": [1, 20, 1, 2]},
            columns=[
                {"original_name": "Cantidad [mg]", "sql_name": "cantidad_mg", "sql_type": "NUMERIC",
                 "unit": "mg", "numeric_eligible": True},
                {"original_name": "Fecha", "sql_name": "fecha", "sql_type": "DATE", "numeric_eligible": False},
            ],
            rows=[{"cantidad_mg": Decimal("0.0010"), "fecha": date(2026, 1, day)} for day in range(1, 20)],
            warnings=["Conservar precisión; no convertir unidades."],
        )
        before = deepcopy(table)
        response = {"description": "Cantidades por fecha.", "columns": [
            {"original_name": "Cantidad [mg]", "meaning": "Cantidad registrada."},
            {"original_name": "Fecha", "meaning": "Fecha del registro."},
        ]}
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"model": settings.default_model, "done": True,
                                           "message": {"content": json.dumps(response)}})

        provider = OllamaSemanticProvider(settings, transport=httpx.MockTransport(handle))
        result = await provider.describe(table)
        self.assertEqual(table, before)
        self.assertEqual(result.description, "Cantidades por fecha.")
        self.assertEqual(result.columns[0]["unit"], "mg")
        self.assertEqual(result.columns[0]["sql_type"], "NUMERIC")
        self.assertTrue(result.columns[0]["numeric_eligible"])
        sent = json.loads(requests[0]["messages"][1]["content"])
        self.assertEqual(sent["sample"], [["0.0010", "2026-01-01"], ["0.0010", "2026-01-19"]])
        self.assertTrue(sent["sample_is_partial"])
        self.assertEqual(sent["source_locator"], table.source_locator)
        self.assertEqual(sent["warnings"], table.warnings)
        self.assertEqual(requests[0]["options"], {"temperature": 0, "num_ctx": 32768})
        self.assertFalse(requests[0]["think"])

        oversized = deepcopy(table)
        oversized.source_locator = {"note": "x" * 6000}
        oversized.warnings = ["x" * 600] * 35
        await provider.describe(oversized)
        bounded = json.loads(requests[-1]["messages"][1]["content"])
        self.assertTrue(bounded["source_locator"]["truncated"])
        self.assertEqual(len(bounded["source_locator"]["preview"]), 4096)
        self.assertEqual(len(bounded["warnings"]), 30)
        self.assertTrue(all(len(item) <= 500 for item in bounded["warnings"]))
        self.assertTrue(bounded["warnings_are_partial"])

        for columns in (
            list(reversed(response["columns"])),
            [{"original_name": "Inventada", "meaning": ""}, response["columns"][1]],
            [{**response["columns"][0], "unit": "g"}, response["columns"][1]],
        ):
            with self.subTest(columns=columns), self.assertRaises(LLMProviderError):
                response["columns"] = columns
                await provider.describe(table)
        self.assertEqual(table, before)

    async def test_embedding_endpoint_returns_actual_vectors_and_rejects_bad_protocol(self):
        settings = TabularSettings(_env_file=None)
        requests = []
        body = {"model": "nomic-embed-text:latest", "embeddings": [[0.12, -0.3, 0.8], [0.4, 0.2, -0.1]]}

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json=body)

        provider = OllamaEmbeddingProvider(settings, transport=httpx.MockTransport(handle))
        self.assertEqual(await provider.embed(["Uno", "Dos"]), body["embeddings"])
        self.assertEqual(str(requests[0].url), "http://localhost:11434/api/embed")
        self.assertEqual(json.loads(requests[0].content), {
            "model": "nomic-embed-text", "input": ["Uno", "Dos"], "truncate": False,
        })
        self.assertEqual(await provider.embed([]), [])
        invalid = (
            {"model": settings.default_model, "embeddings": [[0.1, 0.2]]},
            {"model": "nomic-embed-text", "embeddings": []},
            {"model": "nomic-embed-text", "embeddings": [[True, 0.2]]},
            {"model": "nomic-embed-text", "embeddings": [[0, 0]]},
            {"model": "nomic-embed-text", "embeddings": [[0.1, 0.2], [0.3]]},
        )
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(LLMProviderError):
                await provider.embed(["Uno", "Dos"] if len(body["embeddings"]) == 2 else ["Uno"])


class SemanticIndexTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = TabularSettings(_env_file=None)
        self.client = QdrantClient(":memory:")
        self.requests = []

        def handle(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            return httpx.Response(200, json={"model": self.settings.embedding_model,
                                           "embeddings": [[0.25, 0.75] for _ in payload["input"]]})

        self.embeddings = OllamaEmbeddingProvider(self.settings, transport=httpx.MockTransport(handle))

    def tearDown(self):
        self.client.close()

    async def test_idempotent_dataset_points_lifecycle_and_postgres_validation(self):
        first, second, source_id = uuid4(), uuid4(), uuid4()
        source = {"id": source_id, "is_active": True, "datasets": [
            {"id": dataset_id, "semantic_point_id": dataset_id, "version": 1, "description": "Tabla de muestras.",
             "columns_metadata": [{"original_name": "Código", "sql_name": "codigo", "sql_type": "TEXT"}],
             "rows": [{"codigo": "sentinel_sql_row_value"}]}
            for dataset_id in (first, second)
        ]}
        queries = _Queries([first])
        index = QdrantSemanticIndex(self.settings, client=self.client, embeddings=self.embeddings, queries=queries)
        await index.index_source(source)
        await index.index_source(source)
        self.assertEqual(self.client.count(self.settings.qdrant_collection).count, 2)
        self.assertNotIn("sentinel_sql_row_value", json.dumps(self.requests))
        points, _ = self.client.scroll(self.settings.qdrant_collection)
        self.assertEqual({point.id for point in points}, {str(first), str(second)})
        self.assertTrue(all(point.payload["content_type"] == "dataset_description" for point in points))
        self.assertTrue(all(point.payload["embedding_model"] == "nomic-embed-text:latest" for point in points))
        config = self.client.get_collection(self.settings.qdrant_collection).config.params.vectors
        self.assertTrue(all(":" not in name for name in config))
        results = await index.search("muestras")
        self.assertEqual([item["dataset_id"] for item in results], [str(first)])
        self.assertEqual(set(queries.requested), {first, second})
        await index.set_source_active(source_id, False)
        self.assertEqual(await index.search("muestras"), [])
        await index.set_source_active(source_id, True)
        self.assertEqual(len(await index.search("muestras")), 1)
        queries.usable.clear()
        self.assertEqual(await index.search("muestras"), [])

    async def test_collection_rejects_dimension_or_model_mismatch_without_upsert(self):
        source = {"id": uuid4(), "is_active": True, "datasets": [{"id": uuid4(), "description": "Muestras."}]}
        for vector_name, dimension, message in (
            ("embedding_" + sha256(model_identity(self.settings.embedding_model).encode()).hexdigest()[:24], 3, "dimensión"),
            ("otro_modelo", 2, "otro modelo"),
        ):
            with self.subTest(vector_name=vector_name, dimension=dimension):
                client = QdrantClient(":memory:")
                try:
                    client.create_collection(self.settings.qdrant_collection, vectors_config={
                        vector_name: models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                    })
                    index = QdrantSemanticIndex(self.settings, client=client, embeddings=self.embeddings)
                    with self.assertRaisesRegex(SemanticIndexError, message):
                        await index.index_source(source)
                    self.assertEqual(client.count(self.settings.qdrant_collection).count, 0)
                    with self.assertRaisesRegex(SemanticIndexError, "PostgreSQL"):
                        await index.search("muestras")
                finally:
                    client.close()


if __name__ == "__main__":
    unittest.main()
