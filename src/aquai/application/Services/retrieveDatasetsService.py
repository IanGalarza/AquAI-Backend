import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from aquai.application.contracts.retrieval import (
    DatasetCandidate,
    DatasetEvidence,
    DatasetRetrievalResult,
    ValueMatch,
)
from aquai.application.contracts.routing import DatasetBinding
from aquai.application.interfaces.retrieval import DatasetSemanticSearch
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import contains_phrase, variable_core
from aquai.domain.enums.sourceStatus import SemanticIndexStatus

_logger = logging.getLogger(__name__)


def _named_columns(dataset, text):
    return [column.sql_name for column in dataset.columns_metadata if _column_named(column, text)]


def _column_named(column, text):
    return contains_phrase(column.original_name, text, plural_insensitive=True) or contains_phrase(
        column.sql_name, text, plural_insensitive=True
    )


def _stored_as_words(found, value):
    for rows in found.values():
        for row in rows.get(value, []):
            if isinstance(row["value"], str) and contains_phrase(row["value"], value):
                return True
    return False


class _Coverage:
    def __init__(self, evidence, values, variables, dimensions):
        self._evidence = evidence
        self._values = values
        self._variables = variables
        self._dimensions = dimensions

    def has_values(self, dataset):
        matches = self._evidence[dataset.dataset_id].value_matches
        return bool(self._values) and len(matches) == len(self._values)

    def has_variables(self, dataset):
        matches = self._evidence[dataset.dataset_id].variable_matches
        return bool(self._variables) and len(matches) == len(self._variables)

    def complete(self, dataset):
        # La dimensión preguntada («qué sitio») exige una columna que la identifique.
        dimensions = len(self._evidence[dataset.dataset_id].dimension_matches) == len(
            self._dimensions
        )
        return (
            (bool(self._values) or bool(self._variables))
            and (self.has_values(dataset) or not self._values)
            and (self.has_variables(dataset) or not self._variables)
            and dimensions
        )

    def matches(self, dataset):
        item = self._evidence[dataset.dataset_id]
        return len(item.value_matches) + len(item.variable_matches)


class RetrieveDatasetsService:
    def __init__(self, index: DatasetSemanticSearch, queries: SourceQueries) -> None:
        self._index = index
        self._queries = queries

    async def retrieve(
        self,
        question: str,
        *,
        limit: int = 5,
        dataset_bindings: list[DatasetBinding] | None = None,
        data_mentions=(),
        tentative_values=(),
    ) -> DatasetRetrievalResult:
        if not dataset_bindings:
            return await self._search(question, limit, data_mentions, tentative_values)

        result = await self._explicit(question, dataset_bindings)
        # Una fuente nombrada por el usuario se respeta; los datasets heredados del
        # historial sí se acotan por la evidencia de la nueva pregunta.
        inherited = all(item.historical_version is not None for item in dataset_bindings)
        ranked = await self._rank(
            result, data_mentions, limit=None, restrict=inherited, tentative=tentative_values
        )
        if not inherited or self._answers(ranked, data_mentions):
            return ranked

        # Un seguimiento sobre otra variable o sitio («¿y la turbiedad?») no queda atado a
        # datasets que no los registran: se busca en todo el catálogo con la misma evidencia.
        # La similitud se calcula sobre lo pedido, no sobre los metadatos históricos de la pregunta.
        requested = " ".join(dict.fromkeys(item.text for item in data_mentions))
        found = await self._search(
            question, limit, data_mentions, tentative_values, query=requested
        )
        return found if self._answers(found, data_mentions) else ranked

    async def _search(self, question, limit, data_mentions, tentative_values, *, query=None):
        # La similitud ordena un conjunto amplio; la evidencia de registros decide el recorte.
        hits = await self._index.search(query or question, limit=min(100, max(limit * 6, 30)))
        snapshots = await self._snapshots(hits)
        candidates = []
        for hit in hits:
            candidate = self._candidate(hit, snapshots)
            if candidate is not None:
                candidates.append(candidate)

        result = DatasetRetrievalResult(
            question=question.strip(),
            candidates=candidates,
            verified_at=datetime.now(timezone.utc),
            status="candidates_available" if candidates else "no_available_candidates",
            warnings=self._search_warnings(hits, candidates),
        )
        return await self._rank(
            result, data_mentions, limit=limit, restrict=True, tentative=tentative_values
        )

    async def _snapshots(self, hits):
        versions = {UUID(hit["dataset_id"]): hit["version"] for hit in hits}
        if not hits:
            return {}

        verified = await asyncio.to_thread(
            self._queries.verified_datasets,
            list(versions),
            versions=versions,
        )
        return {dataset.dataset_id: dataset for dataset in verified}

    def _candidate(self, hit, snapshots):
        dataset = snapshots.get(UUID(hit["dataset_id"]))
        if dataset is None or dataset.source_id != UUID(hit["source_id"]):
            return None
        if dataset.semantic_point_id != UUID(hit["semantic_point_id"]):
            return None

        warnings = self._candidate_warnings(dataset, hit)
        return DatasetCandidate(
            **dataset.model_dump(exclude={"warnings"}),
            score=hit["score"],
            indexed_version=hit["version"],
            embedding_model=hit["embedding_model"],
            warnings=list(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _candidate_warnings(dataset, hit):
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
            column.numeric_eligible and column.unit is None for column in dataset.columns_metadata
        ):
            warnings.append(
                "Hay columnas numéricas sin unidad documentada; "
                "no inferir ni convertir unidades."
            )
        return warnings

    @staticmethod
    def _search_warnings(hits, candidates):
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
        return warnings

    async def _rank(self, result, mentions, *, limit, restrict, tentative=()):
        """Ordena por evidencia verificable: valores almacenados, columnas y similitud."""
        values = list(dict.fromkeys(item.text for item in mentions if item.kind == "value"))
        variables = list(dict.fromkeys(item.text for item in mentions if item.kind == "variable"))
        dimensions = list(dict.fromkeys(item.text for item in mentions if item.kind == "dimension"))
        candidates = self._prune_overlaps(result)
        evidence = self._variable_evidence(candidates, variables)
        values = self._split_dimensions(
            result, candidates, evidence, values, variables, dimensions
        )

        tentative = [item for item in dict.fromkeys(tentative) if item not in values]
        values, tentative, found = await self._stored_values(result, candidates, values, tentative)
        values = self._confirm_tentative(result, found, values, tentative)
        self._record_value_matches(candidates, evidence, found, values)
        self._record_unmatched(result, evidence, values)

        coverage = _Coverage(evidence, values, variables, result.dimension_mentions)
        position = {dataset.dataset_id: index for index, dataset in enumerate(candidates)}
        ranked = sorted(
            candidates,
            key=lambda item: (
                coverage.complete(item),
                coverage.matches(item),
                -position[item.dataset_id],
            ),
            reverse=True,
        )
        if restrict:
            ranked = self._restricted(result, ranked, coverage)
        if limit is not None:
            ranked = ranked[:limit]

        result.candidates = ranked
        result.evidence = [
            evidence[item.dataset_id]
            for item in ranked
            if self._has_evidence(evidence[item.dataset_id])
        ]
        result.status = "candidates_available" if ranked else "no_available_candidates"
        return result

    @staticmethod
    def _variable_evidence(candidates, variables):
        columns = {
            variable: RetrieveDatasetsService._variable_columns(candidates, variable)
            for variable in variables
        }
        evidence = {}
        for dataset in candidates:
            matches = {}
            for variable, by_dataset in columns.items():
                names = by_dataset[dataset.dataset_id]
                if names:
                    matches[variable] = names
            evidence[dataset.dataset_id] = DatasetEvidence(
                dataset_id=dataset.dataset_id, variable_matches=matches
            )
        return evidence

    @staticmethod
    def _variable_columns(candidates, variable):
        # La estadística de la mención («pH promedio») se descarta sólo si ningún candidato nombra la mención entera.
        exact = {dataset.dataset_id: _named_columns(dataset, variable) for dataset in candidates}
        if any(exact.values()):
            return exact

        core = variable_core(variable)
        return {dataset.dataset_id: _named_columns(dataset, core) for dataset in candidates}

    @staticmethod
    def _split_dimensions(result, candidates, evidence, values, variables, dimensions):
        with_variables = [
            dataset
            for dataset in candidates
            if variables and len(evidence[dataset.dataset_id].variable_matches) == len(variables)
        ]
        # Una «mención de valor» que nombra una columna de un dataset con la variable pedida
        # es la dimensión que se pregunta (sitio -> Sitios), no un valor para filtrar.
        reclassified = [
            value
            for value in values
            if any(_named_columns(dataset, value) for dataset in with_variables)
        ]
        # Una dimensión que no nombra ninguna columna («punto de muestreo») no exige nada.
        asked = [
            dimension
            for dimension in dimensions
            if any(_named_columns(dataset, dimension) for dataset in with_variables)
        ]
        result.dimension_mentions = list(dict.fromkeys([*reclassified, *asked]))
        if not result.dimension_mentions:
            return values

        values = [value for value in values if value not in reclassified]
        for dataset in candidates:
            evidence[dataset.dataset_id].dimension_matches = {
                dimension: names
                for dimension in result.dimension_mentions
                if (names := _named_columns(dataset, dimension))
            }
        if reclassified:
            result.warnings.append(
                "Se interpretó como columna, no como valor de filtro: "
                + "; ".join(f"«{value}»" for value in reclassified)
                + "."
            )
        return values

    async def _stored_values(self, result, candidates, values, tentative):
        if not ((values or tentative) and candidates):
            return values, tentative, {}

        try:
            found = await asyncio.to_thread(
                self._queries.find_text_values, candidates, [*values, *tentative]
            )
        except Exception as error:  # La recuperación semántica sigue disponible.
            _logger.warning(
                "No se pudieron verificar valores mencionados: %s", type(error).__name__
            )
            result.warnings.append("No se pudo comprobar en los registros los valores mencionados.")
            return [], [], {}
        return values, tentative, found

    @staticmethod
    def _confirm_tentative(result, found, values, tentative):
        # Un valor tentativo sólo se convierte en filtro si aparece como palabras completas.
        result.confirmed_values = [value for value in tentative if _stored_as_words(found, value)]
        if not result.confirmed_values:
            return values

        values = [*values, *result.confirmed_values]
        result.warnings.append(
            "Se interpretó la aclaración como valor registrado: "
            + "; ".join(f"«{value}»" for value in result.confirmed_values)
            + "."
        )
        return values

    @staticmethod
    def _record_value_matches(candidates, evidence, found, values):
        for dataset in candidates:
            item = evidence[dataset.dataset_id]
            for value in values:
                rows = found.get(dataset.dataset_id, {}).get(value, [])
                matches = RetrieveDatasetsService._value_matches(rows, value)
                if matches:
                    item.value_matches[value] = matches

    @staticmethod
    def _value_matches(rows, value):
        columns = {}
        for row in rows:
            if not (isinstance(row["value"], str) and contains_phrase(row["value"], value)):
                continue
            match = columns.setdefault(
                row["column"], ValueMatch(column=row["column"], values=[], rows=0)
            )
            match.values.append(row["value"])
            match.rows += row["rows"]
        return list(columns.values())

    @staticmethod
    def _record_unmatched(result, evidence, values):
        result.unmatched_values = [
            value
            for value in values
            if not any(value in item.value_matches for item in evidence.values())
        ]
        if result.unmatched_values:
            result.warnings.append(
                "Ningún candidato contiene en sus registros: "
                + "; ".join(f"«{value}»" for value in result.unmatched_values)
                + ". No se debe filtrar por un valor inexistente ni sustituirlo por otro."
            )

    @staticmethod
    def _restricted(result, ranked, coverage):
        # Un dataset sin el sitio o la variable pedidos no puede responder por ellos;
        # si ninguno reúne todo, la variable medida define la pregunta.
        for rule, note in (
            (coverage.complete, "contienen en sus registros los valores y columnas mencionados"),
            (coverage.has_variables, "tienen columnas cuyo nombre coincide con la variable pedida"),
            (coverage.has_values, "contienen en sus registros los valores mencionados"),
        ):
            kept = [item for item in ranked if rule(item)]
            if not kept:
                continue
            if len(kept) < len(ranked):
                result.warnings.append(
                    f"Se priorizaron {len(kept)} datasets que {note}; se descartaron {len(ranked) - len(kept)} candidatos."
                )
            return kept
        return ranked

    @staticmethod
    def _has_evidence(item):
        return bool(item.value_matches or item.variable_matches or item.dimension_matches)

    @staticmethod
    def _answers(result, mentions):
        variables = {item.text for item in mentions if item.kind == "variable"}
        values = {item.text for item in mentions if item.kind == "value"} - set(
            result.dimension_mentions
        )
        if not variables and not values:
            return True

        return any(
            variables <= set(item.variable_matches) and values <= set(item.value_matches)
            for item in result.evidence
        )

    @staticmethod
    def _prune_overlaps(result):
        """Una hoja cuyas filas están todas en otra candidata no se cuenta dos veces."""
        names = {item.name: item for item in result.candidates}
        kept = []
        for dataset in result.candidates:
            covered = RetrieveDatasetsService._covering_candidate(dataset, names)
            if covered is not None and covered.dataset_id != dataset.dataset_id:
                result.warnings.append(
                    f"Se omitió «{dataset.name}»: sus {dataset.row_count} filas también están en «{covered.name}»."
                )
                continue
            kept.append(dataset)
        return kept

    @staticmethod
    def _covering_candidate(dataset, names):
        for overlap in dataset.provenance.source_locator.get("possible_overlap") or []:
            if not isinstance(overlap, dict) or overlap.get("with") not in names:
                continue

            other = names[overlap["with"]]
            if (
                other.source_id == dataset.source_id
                and overlap.get("matching_rows") == dataset.row_count
                and other.row_count >= dataset.row_count
            ):
                return other
        return None

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

            version_changed = (
                binding.historical_version is not None
                and binding.historical_version != dataset.verified_version
            )
            if version_changed:
                warnings.append(
                    f"El dataset {dataset.dataset_id} cambió de versión; "
                    f"se utiliza la versión vigente {dataset.verified_version}."
                )
            candidates.append(dataset)

        return DatasetRetrievalResult(
            question=question.strip(),
            candidates=candidates,
            retrieval_method="explicit",
            verified_at=datetime.now(timezone.utc),
            status="candidates_available" if candidates else "no_available_candidates",
            warnings=warnings,
        )
