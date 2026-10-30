"""Recupera evidencia para un futuro Planner sin seleccionar ni combinar datos."""

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from aquai.application.contracts.retrieval import (
    DatasetCandidate, DatasetEvidence, DatasetRetrievalResult, ValueMatch,
)
from aquai.application.contracts.routing import DatasetBinding
from aquai.application.interfaces.retrieval import DatasetSemanticSearch
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import contains_phrase
from aquai.domain.enums.sourceStatus import SemanticIndexStatus


_logger = logging.getLogger(__name__)


class RetrieveDatasetsHandler:
    def __init__(self, index: DatasetSemanticSearch, queries: SourceQueries) -> None:
        self._index = index
        self._queries = queries

    async def handle(
        self, question: str, *, limit: int = 5,
        dataset_bindings: list[DatasetBinding] | None = None, data_mentions=(), tentative_values=(),
    ) -> DatasetRetrievalResult:
        if dataset_bindings:
            result = await self._explicit(question, dataset_bindings)
            # Una fuente nombrada por el usuario se respeta; los datasets heredados del
            # historial sí se acotan por la evidencia de la nueva pregunta.
            inherited = all(item.historical_version is not None for item in dataset_bindings)
            return await self._rank(result, data_mentions, limit=None, restrict=inherited,
                                    tentative=tentative_values)
        # La similitud ordena un conjunto amplio; la evidencia de registros decide el recorte.
        hits = await self._index.search(question, limit=min(100, max(limit * 6, 30)))
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
        result = DatasetRetrievalResult(
            question=question.strip(), candidates=candidates,
            verified_at=datetime.now(timezone.utc),
            status="candidates_available" if candidates else "no_available_candidates",
            warnings=warnings,
        )
        return await self._rank(result, data_mentions, limit=limit, restrict=True, tentative=tentative_values)

    async def _rank(self, result, mentions, *, limit, restrict, tentative=()):
        """Ordena por evidencia verificable: valores almacenados, columnas y similitud."""
        values = list(dict.fromkeys(item.text for item in mentions if item.kind == "value"))
        variables = list(dict.fromkeys(item.text for item in mentions if item.kind == "variable"))
        candidates = self._prune_overlaps(result)

        def named(dataset, text):
            return [column.sql_name for column in dataset.columns_metadata
                    if contains_phrase(column.original_name, text, plural_insensitive=True)
                    or contains_phrase(column.sql_name, text, plural_insensitive=True)]

        evidence = {dataset.dataset_id: DatasetEvidence(dataset_id=dataset.dataset_id, variable_matches={
            variable: names for variable in variables if (names := named(dataset, variable))
        }) for dataset in candidates}
        with_variables = [dataset for dataset in candidates
                          if variables and len(evidence[dataset.dataset_id].variable_matches) == len(variables)]
        # Una «mención de valor» que nombra una columna de un dataset con la variable pedida
        # es la dimensión que se pregunta (sitio -> Sitios), no un valor para filtrar.
        result.dimension_mentions = [value for value in values
                                     if any(named(dataset, value) for dataset in with_variables)]
        if result.dimension_mentions:
            values = [value for value in values if value not in result.dimension_mentions]
            for dataset in candidates:
                evidence[dataset.dataset_id].dimension_matches = {
                    dimension: names for dimension in result.dimension_mentions if (names := named(dataset, dimension))}
            result.warnings.append(
                "Se interpretó como columna, no como valor de filtro: "
                + "; ".join(f"«{value}»" for value in result.dimension_mentions) + "."
            )
        tentative = [item for item in dict.fromkeys(tentative) if item not in values]
        found = {}
        if (values or tentative) and candidates:
            try:
                found = await asyncio.to_thread(self._queries.find_text_values, candidates, [*values, *tentative])
            except Exception as error:  # La recuperación semántica sigue disponible.
                _logger.warning("No se pudieron verificar valores mencionados: %s", type(error).__name__)
                result.warnings.append("No se pudo comprobar en los registros los valores mencionados.")
                values, tentative = [], []
        # Un valor tentativo sólo se convierte en filtro si aparece como palabras completas.
        result.confirmed_values = [value for value in tentative if any(
            isinstance(row["value"], str) and contains_phrase(row["value"], value)
            for rows in found.values() for row in rows.get(value, []))]
        if result.confirmed_values:
            values = [*values, *result.confirmed_values]
            result.warnings.append("Se interpretó la aclaración como valor registrado: "
                                   + "; ".join(f"«{value}»" for value in result.confirmed_values) + ".")
        for dataset in candidates:
            item = evidence[dataset.dataset_id]
            for value in values:
                columns = {}
                for row in found.get(dataset.dataset_id, {}).get(value, []):
                    if isinstance(row["value"], str) and contains_phrase(row["value"], value):
                        match = columns.setdefault(row["column"], ValueMatch(column=row["column"], values=[], rows=0))
                        match.values.append(row["value"])
                        match.rows += row["rows"]
                if columns:
                    item.value_matches[value] = list(columns.values())
        result.unmatched_values = [value for value in values
                                   if not any(value in item.value_matches for item in evidence.values())]
        if result.unmatched_values:
            result.warnings.append(
                "Ningún candidato contiene en sus registros: " + "; ".join(f"«{value}»" for value in result.unmatched_values)
                + ". No se debe filtrar por un valor inexistente ni sustituirlo por otro."
            )
        position = {dataset.dataset_id: index for index, dataset in enumerate(candidates)}

        def has_values(dataset):
            return bool(values) and len(evidence[dataset.dataset_id].value_matches) == len(values)

        def has_variables(dataset):
            return bool(variables) and len(evidence[dataset.dataset_id].variable_matches) == len(variables)

        def complete(dataset):
            # La dimensión preguntada («qué sitio») exige una columna que la identifique.
            dimensions = len(evidence[dataset.dataset_id].dimension_matches) == len(result.dimension_mentions)
            return (bool(values) or bool(variables)) and (has_values(dataset) or not values) and (
                has_variables(dataset) or not variables) and dimensions

        ranked = sorted(candidates, key=lambda item: (
            complete(item), len(evidence[item.dataset_id].value_matches) + len(evidence[item.dataset_id].variable_matches),
            -position[item.dataset_id]), reverse=True)
        if restrict:
            # Un dataset sin el sitio o la variable pedidos no puede responder por ellos;
            # si ninguno reúne todo, la variable medida define la pregunta.
            for rule, note in ((complete, "contienen en sus registros los valores y columnas mencionados"),
                               (has_variables, "tienen columnas cuyo nombre coincide con la variable pedida"),
                               (has_values, "contienen en sus registros los valores mencionados")):
                kept = [item for item in ranked if rule(item)]
                if kept:
                    if len(kept) < len(ranked):
                        result.warnings.append(
                            f"Se priorizaron {len(kept)} datasets que {note}; se descartaron {len(ranked) - len(kept)} candidatos."
                        )
                    ranked = kept
                    break
        if limit is not None:
            ranked = ranked[:limit]
        result.candidates = ranked
        result.evidence = [evidence[item.dataset_id] for item in ranked
                           if evidence[item.dataset_id].value_matches or evidence[item.dataset_id].variable_matches
                           or evidence[item.dataset_id].dimension_matches]
        result.status = "candidates_available" if ranked else "no_available_candidates"
        return result

    @staticmethod
    def _prune_overlaps(result):
        """Una hoja cuyas filas están todas en otra candidata no se cuenta dos veces."""
        names = {item.name: item for item in result.candidates}
        kept = []
        for dataset in result.candidates:
            covered = next((
                names[overlap["with"]] for overlap in dataset.provenance.source_locator.get("possible_overlap") or []
                if isinstance(overlap, dict) and overlap.get("with") in names
                and names[overlap["with"]].source_id == dataset.source_id
                and overlap.get("matching_rows") == dataset.row_count
                and names[overlap["with"]].row_count >= dataset.row_count
            ), None)
            if covered is not None and covered.dataset_id != dataset.dataset_id:
                result.warnings.append(
                    f"Se omitió «{dataset.name}»: sus {dataset.row_count} filas también están en «{covered.name}»."
                )
                continue
            kept.append(dataset)
        return kept

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
