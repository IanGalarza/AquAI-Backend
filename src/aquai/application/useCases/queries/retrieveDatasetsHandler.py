"""Recupera evidencia para un futuro Planner sin seleccionar ni combinar datos."""

import asyncio
from datetime import datetime, timezone
from uuid import UUID

from aquai.application.contracts.retrieval import DatasetCandidate, DatasetRetrievalResult
from aquai.application.contracts.routing import DatasetBinding
from aquai.application.interfaces.retrieval import DatasetSemanticSearch
from aquai.application.interfaces.sources import SourceQueries
from aquai.domain.enums.sourceStatus import SemanticIndexStatus


class RetrieveDatasetsHandler:
    def __init__(self, index: DatasetSemanticSearch, queries: SourceQueries) -> None:
        self._index = index
        self._queries = queries

    async def handle(
        self, question: str, *, limit: int = 5,
        dataset_bindings: list[DatasetBinding] | None = None,
    ) -> DatasetRetrievalResult:
        if dataset_bindings:
            return await self._explicit(question, dataset_bindings)
        hits = await self._index.search(question, limit=limit)
        versions = {UUID(hit["dataset_id"]): hit["version"] for hit in hits}
        verified = await asyncio.to_thread(
            self._queries.verified_datasets, list(versions), versions=versions,
        ) if hits else []
        snapshots = {dataset.dataset_id: dataset for dataset in verified}
        candidates = []
        for hit in hits:
            dataset = snapshots.get(UUID(hit["dataset_id"]))
            if dataset is None or dataset.source_id != UUID(hit["source_id"]):
                continue
            if dataset.semantic_point_id != UUID(hit["semantic_point_id"]):
                continue
            warnings = list(dataset.warnings)
            if dataset.provenance.semantic_index_status != SemanticIndexStatus.READY:
                warnings.append(
                    "La fuente tiene una actualización del índice pendiente o fallida; "
                    "la versión del candidato sí fue verificada en PostgreSQL."
                )
            locator = dataset.provenance.source_locator
            if locator.get("possible_overlap") or locator.get("has_possible_overlap"):
                warnings.append(
                    "Posible solapamiento: revisar la procedencia antes de combinar observaciones."
                )
            if hit.get("embedding_text_truncated"):
                warnings.append(
                    "El texto indexado fue truncado; algunas columnas pueden no haber "
                    "participado de la búsqueda."
                )
            if any(
                column.numeric_eligible and column.unit is None
                for column in dataset.columns_metadata
            ):
                warnings.append(
                    "Hay columnas numéricas sin unidad documentada; "
                    "no inferir ni convertir unidades."
                )
            candidates.append(DatasetCandidate(
                **dataset.model_dump(exclude={"warnings"}),
                score=hit["score"], indexed_version=hit["version"],
                embedding_model=hit["embedding_model"],
                warnings=list(dict.fromkeys(warnings)),
            ))
        warnings = [
            "La similitud ordena candidatos, pero no confirma relevancia ni cobertura. "
            "El Planner debe verificar si permiten responder; "
            "puede no haber datasets relevantes.",
        ]
        if len(candidates) < len(hits):
            warnings.append(
                "Se descartaron candidatos que cambiaron o no coincidían "
                "con la identidad verificada en PostgreSQL."
            )
        if not candidates:
            warnings.append(
                "No se encontraron candidatos disponibles. El índice puede estar vacío, "
                "incompleto o desactualizado; esto no demuestra ausencia de datos en el catálogo."
            )
        return DatasetRetrievalResult(
            question=question.strip(), candidates=candidates,
            verified_at=datetime.now(timezone.utc),
            status="candidates_available" if candidates else "no_available_candidates",
            warnings=warnings,
        )

    async def _explicit(self, question, bindings):
        """Reutiliza la misma verificación física; no inventa similitudes del índice."""
        ids = list(dict.fromkeys(item.dataset_id for item in bindings))
        verified = await asyncio.to_thread(self._queries.verified_datasets, ids)
        by_id = {item.dataset_id: item for item in verified}
        warnings = [
            "Selección explícita o histórica: sólo se evaluaron los datasets indicados. "
            "La disponibilidad y la versión se verificaron nuevamente en PostgreSQL."
        ]
        candidates = []
        for binding in bindings:
            dataset = by_id.get(binding.dataset_id)
            if dataset is None or dataset.source_id != binding.source_id:
                warnings.append(
                    "No se pudo verificar toda la selección indicada; no se sustituyó "
                    "por otros datasets ni se autorizó una ejecución parcial."
                )
                candidates = []
                break
            if (
                binding.historical_version is not None
                and binding.historical_version != dataset.verified_version
            ):
                warnings.append(
                    f"El dataset {dataset.dataset_id} cambió de versión; "
                    f"se utiliza la versión vigente {dataset.verified_version}."
                )
            candidates.append(dataset)
        return DatasetRetrievalResult(
            question=question.strip(), candidates=candidates, retrieval_method="explicit",
            verified_at=datetime.now(timezone.utc),
            status="candidates_available" if candidates else "no_available_candidates",
            warnings=warnings,
        )
