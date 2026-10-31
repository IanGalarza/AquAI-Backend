import asyncio
import json
import logging
import math
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from weakref import WeakKeyDictionary

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aquai.application.contracts.ingestion import (
    PreparedTable,
    SemanticDescription,
    StructureSuggestion,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.tabular import (
    TABULAR_SEMANTIC_PROMPT,
    TABULAR_STRUCTURE_PROMPT,
)
from aquai.infrastructure.config.tabular import TabularSettings

_inference_locks: WeakKeyDictionary = WeakKeyDictionary()

_DESCRIBED_COLUMN_KEYS = (
    "original_name",
    "sql_type",
    "unit",
    "header_missing",
    "derived_from",
    "representation",
    "excluded_count",
)


def _inference_lock() -> asyncio.Lock:
    """Las instancias comparten una cola por loop para no saturar Ollama."""
    loop = asyncio.get_running_loop()
    if loop not in _inference_locks:
        _inference_locks[loop] = asyncio.Lock()

    return _inference_locks[loop]


def model_identity(model: str) -> str:
    image_name = model.rsplit("/", 1)[-1]
    if ":" in image_name:
        return model

    return f"{model}:latest"


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class _ColumnMeaning(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    original_name: str
    meaning: str = Field(max_length=800)


class _Description(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    description: str = Field(min_length=1, max_length=4000)
    columns: list[_ColumnMeaning]


class _StructureHint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    region_id: str = Field(max_length=80)
    orientation: Literal["rows", "columns"]
    header_start: int = Field(ge=1)
    header_depth: int = Field(ge=1, le=4)
    data_start: int = Field(ge=2)


class _StructureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    suggestions: list[_StructureHint] = Field(max_length=3)


class _OllamaClient:
    def __init__(
        self, settings: TabularSettings, transport: httpx.AsyncBaseTransport | None
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def post(self, path: str, payload: dict, timeout: float) -> dict:
        payload = self._with_keep_alive(payload)
        response = await self._send(path, payload, timeout)
        body = self._protocol_body(response)

        if not self._answered_by_requested_model(body, payload):
            raise LLMProviderError("Ollama respondió con un modelo distinto del solicitado.")

        return body

    def _with_keep_alive(self, payload: dict) -> dict:
        if self._settings.llm_keep_alive is None:
            return payload

        return {**payload, "keep_alive": self._settings.llm_keep_alive}

    async def _send(self, path: str, payload: dict, timeout: float) -> httpx.Response:
        try:
            async with _inference_lock():
                async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                    response = await client.post(
                        f"{self._settings.ollama_base_url}{path}", json=payload
                    )
                    response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMProviderError("Ollama superó el tiempo máximo de respuesta.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama respondió con un error HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError("No se pudo completar la conexión con Ollama.") from exc

        return response

    @staticmethod
    def _protocol_body(response: httpx.Response) -> dict:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama devolvió una respuesta HTTP que no es JSON.") from exc

        if not isinstance(body, dict) or body.get("error"):
            raise LLMProviderError("Ollama devolvió una respuesta de protocolo inválida.")

        return body

    @staticmethod
    def _answered_by_requested_model(body: dict, payload: dict) -> bool:
        returned_model = body.get("model")
        if not isinstance(returned_model, str):
            return False

        return model_identity(returned_model) == model_identity(payload["model"])


class OllamaSemanticProvider:
    def __init__(
        self, settings: TabularSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._client = _OllamaClient(settings, transport)

    async def suggest_structure(self, candidates: list[dict]) -> list[StructureSuggestion]:
        if not candidates or not self._settings.structure_assistance:
            return []
        if len(candidates) > 3:
            raise LLMProviderError(
                "La asistencia estructural admite como máximo tres regiones por archivo."
            )

        region_ids = [candidate["region_id"] for candidate in candidates]
        schema = self._structure_schema(region_ids)
        content = await self._chat(TABULAR_STRUCTURE_PROMPT, schema, {"regions": candidates})
        result = self._parse_structure(content)

        returned = [hint.region_id for hint in result.suggestions]
        repeated = len(returned) != len(set(returned))
        if repeated or any(region_id not in region_ids for region_id in returned):
            raise LLMProviderError("La asistencia alteró o repitió las regiones originales.")

        return [StructureSuggestion(**hint.model_dump()) for hint in result.suggestions]

    async def describe(self, table: PreparedTable) -> SemanticDescription:
        names = [column["original_name"] for column in table.columns]
        if not names or len(names) > self._settings.max_columns:
            raise LLMProviderError("La cantidad de columnas excede el límite de descripción.")

        schema = self._description_schema(names)
        evidence = self._description_evidence(table)
        content = await self._chat(TABULAR_SEMANTIC_PROMPT, schema, evidence)
        result = self._parse_description(content)

        if [column.original_name for column in result.columns] != names:
            raise LLMProviderError("Ollama alteró, omitió o reordenó columnas de la tabla.")

        columns = deepcopy(table.columns)
        for column, semantic in zip(columns, result.columns, strict=True):
            column["meaning"] = semantic.meaning.strip()

        if not result.description.strip():
            raise LLMProviderError("Ollama devolvió una descripción vacía.")

        return SemanticDescription(description=result.description.strip(), columns=columns)

    @staticmethod
    def _structure_schema(region_ids: list) -> dict:
        schema = _StructureResponse.model_json_schema()
        schema["$defs"]["_StructureHint"]["properties"]["region_id"]["enum"] = region_ids
        return schema

    @staticmethod
    def _parse_structure(content: str) -> _StructureResponse:
        try:
            return _StructureResponse.model_validate_json(content)
        except ValidationError as exc:
            raise LLMProviderError("La asistencia estructural no cumple el contrato JSON.") from exc

    @staticmethod
    def _description_schema(names: list) -> dict:
        schema = _Description.model_json_schema()
        original_name = schema["$defs"]["_ColumnMeaning"]["properties"]["original_name"]
        original_name["enum"] = list(dict.fromkeys(names))
        return schema

    def _description_evidence(self, table: PreparedTable) -> dict:
        sample = self._sample(table)

        return {
            "name": table.name,
            "description": table.description,
            "source_locator": self._bounded_locator(table.source_locator),
            "warnings": [str(warning)[:500] for warning in table.warnings[:30]],
            "warnings_are_partial": len(table.warnings) > 30,
            "columns": [self._column_evidence(column) for column in table.columns],
            "row_count": len(table.rows),
            "sample": sample,
            "sample_is_partial": len(table.rows) > len(sample),
        }

    @staticmethod
    def _column_evidence(column: dict) -> dict:
        return {key: column[key] for key in _DESCRIBED_COLUMN_KEYS if key in column}

    @staticmethod
    def _parse_description(content: str) -> _Description:
        try:
            return _Description.model_validate_json(content)
        except ValidationError as exc:
            raise LLMProviderError(
                "Ollama devolvió una descripción que no cumple el contrato JSON."
            ) from exc

    async def _chat(self, prompt: str, schema: dict, evidence: dict) -> str:
        content = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content) > 96_000:
            raise LLMProviderError("Los metadatos exceden el límite de contexto de descripción.")

        body = await self._client.post(
            "/api/chat",
            {
                "model": self._settings.default_model,
                "stream": False,
                "think": self._settings.llm_think,
                "format": schema,
                "options": {"temperature": 0, "num_ctx": self._settings.llm_num_ctx},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ],
            },
            self._settings.llm_timeout_seconds,
        )

        message = body.get("message")
        incomplete = (
            body.get("done") is not True
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
        )
        if incomplete:
            raise LLMProviderError("Ollama devolvió una descripción incompleta.")

        return message["content"]

    @staticmethod
    def _bounded_locator(locator: dict[str, Any]) -> dict[str, Any]:
        content = json.dumps(locator, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(content) > 4096:
            return {"preview": content[:4096], "truncated": True}

        return json.loads(content)

    def _sample(self, table: PreparedTable) -> list[list[Any]]:
        rows: list[list[Any]] = []
        remaining = 16_000
        for index in self._evenly_spaced_indices(len(table.rows)):
            row = [
                self._cell_preview(table.rows[index].get(column["sql_name"]))
                for column in table.columns
            ]
            size = len(json.dumps(row, ensure_ascii=False))
            if size > remaining:
                break

            rows.append(row)
            remaining -= size

        return rows

    def _evenly_spaced_indices(self, row_count: int) -> list[int]:
        count = min(row_count, self._settings.sample_rows)
        return [round(index * (row_count - 1) / max(1, count - 1)) for index in range(count)]

    @staticmethod
    def _cell_preview(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return str(value)
        text = str(value)
        if len(text) <= 160:
            return text

        return text[:160] + " [muestra truncada]"


class OllamaEmbeddingProvider:
    def __init__(
        self, settings: TabularSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._client = _OllamaClient(settings, transport)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise LLMProviderError("Los embeddings requieren textos no vacíos.")

        body = await self._client.post(
            "/api/embed",
            {
                "model": self._settings.embedding_model,
                "input": texts,
                "truncate": False,
            },
            self._settings.embedding_timeout_seconds,
        )

        vectors = body.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise LLMProviderError("Ollama no devolvió un embedding por texto.")

        self._validate_vectors(vectors)
        return [[float(value) for value in vector] for vector in vectors]

    @staticmethod
    def _validate_vectors(vectors: list) -> None:
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise LLMProviderError("Ollama devolvió un embedding vacío o inválido.")

            all_finite = all(_is_finite_number(value) for value in vector)
            if not all_finite or not any(value != 0 for value in vector):
                raise LLMProviderError("Ollama devolvió componentes inválidos en un embedding.")

            if dimension is not None and len(vector) != dimension:
                raise LLMProviderError("Ollama devolvió embeddings con dimensiones diferentes.")

            dimension = len(vector)


async def warm_up_models(settings: TabularSettings) -> None:
    """Carga los modelos al iniciar para que la primera consulta no pague la carga."""
    client = _OllamaClient(settings, None)
    try:
        # Mismo num_ctx que las consultas: otro tamaño obligaría a Ollama a recargar el modelo.
        await client.post(
            "/api/chat",
            {
                "model": settings.default_model,
                "messages": [],
                "options": {"num_ctx": settings.llm_num_ctx},
            },
            settings.llm_timeout_seconds,
        )
        await client.post(
            "/api/embed",
            {"model": settings.embedding_model, "input": ["warm-up"]},
            settings.embedding_timeout_seconds,
        )
    except LLMProviderError as error:
        logging.getLogger(__name__).warning(
            "No se pudieron precargar los modelos de Ollama: %s", error
        )
