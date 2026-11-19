import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

from aquai.application.contracts.retrieval import (
    DatasetCandidate,
    DatasetEvidence,
    DatasetRetrievalResult,
    OmittedCandidate,
    ValueMatch,
)
from aquai.application.contracts.routing import DatasetBinding
from aquai.application.interfaces.retrieval import DatasetSemanticSearch
from aquai.application.interfaces.sources import SourceQueries
from aquai.application.matching import (
    abbreviates,
    aligned_span,
    contains_phrase,
    formula_named,
    normalize_text,
    other_quantity,
    secondary_quantity,
    singular_words,
    spelled_as,
    symbol_named,
    variable_core,
    variable_in_unit,
    variable_variants,
    without_medium,
)
from aquai.application.Rules.vocabulary import MENTION_CONNECTORS, PLACE_WORDS
from aquai.domain.enums.sourceStatus import SemanticIndexStatus

_logger = logging.getLogger(__name__)
_PLACE_NAMES = {singular_words(word) for word in PLACE_WORDS}
_DISCARD_REASONS = {
    "medium": "descartadas por medir otro medio que el pedido",
    "sum": "descartadas por sumar varias magnitudes",
    "secondary": "descartadas por ser una proporción o un recuento en otra unidad, con el valor absoluto en el "
    "mismo dataset",
}


#
# Devuelve las columnas del dataset cuyo nombre contiene el texto.
#
def _named_columns(dataset, text):
    return [column.sql_name for column in dataset.columns_metadata if _column_named(column, text)]


#
# Devuelve las columnas del dataset que corresponden a una variable, sin las descartadas.
#
def _variable_columns_named(dataset, text):
    return [column.sql_name for column in _variable_columns_found(dataset, text)[0]]


#
# Encuentra las columnas de una variable y las descartadas, con su motivo.
#
def _variable_columns_found(dataset, text):
    base = without_medium(text)
    variants = variable_variants(base)
    named = [
        column
        for column in dataset.columns_metadata
        if _identifies_variable(column, base, variants) or _measured_as_asked(column, text)
    ] or _abbreviated_columns(dataset, base)
    discarded = {
        column.sql_name: reason
        for column in named
        if (reason := other_quantity(column.original_name, text)) is not None
    }
    kept = [column for column in named if column.sql_name not in discarded]
    primary = [
        column
        for column in kept
        if not secondary_quantity(column.original_name, column.unit, text)
    ]
    if primary:
        discarded |= {column.sql_name: "secondary" for column in kept if column not in primary}

    return primary or kept, discarded


#
# Devuelve las columnas que nombran la variable con las palabras que la mención abrevia («cianos totales» en
# «Cianobacterias totales»): sólo si ninguna columna la nombra tal cual.
#
def _abbreviated_columns(dataset, base):
    columns = []
    for column in dataset.columns_metadata:
        spelled = spelled_as(base, column.original_name)
        if spelled != normalize_text(base) and _identifies_variable(column, spelled, variable_variants(spelled)):
            columns.append(column)
    return columns


#
# Indica si la columna corresponde a la variable por su nombre, su fórmula química o su símbolo.
#
def _identifies_variable(column, base, variants):
    return (
        any(_variable_named(column, variant) for variant in variants)
        or formula_named(column.original_name, base)
        or symbol_named(column.original_name, column.unit, base)
    )


#
# Indica si la columna es de la variable pedida y mide en la unidad que nombra la mención.
#
def _measured_as_asked(column, text):
    asked = variable_in_unit(text, column.original_name, column.unit)
    if not asked:
        return False

    base = without_medium(asked)
    return _identifies_variable(column, base, variable_variants(base))


#
# Indica cómo se reconoció una columna que no lleva el nombre pedido.
#
def _match_reason(column, text):
    base = without_medium(text)
    if _variable_named(column, base):
        return None
    if any(_variable_named(column, variant) for variant in variable_variants(base)):
        return "reconocidas por un nombre equivalente"
    if formula_named(column.original_name, base):
        return "reconocidas por su fórmula química"
    if symbol_named(column.original_name, column.unit, base):
        return "reconocidas por su símbolo y su unidad"
    return "reconocidas por la unidad pedida"


#
# Devuelve las columnas que identifican la dimensión preguntada, priorizando la de nombre exacto.
#
def _dimension_columns(dataset, text):
    named = _dimension_named(dataset, text)
    asked = _content_words(text)
    exact = [column for column in named if _content_words(column.original_name) == asked]

    return [column.sql_name for column in exact or named]


#
# Devuelve las columnas que nombran la dimensión: las que la contienen o, si ninguna, las que la escriben
# abreviada, con una errata o con la misma raíz en otro idioma («alerta», «alert_class»).
#
def _dimension_named(dataset, text):
    named = [column for column in dataset.columns_metadata if _column_named(column, text)]

    return named or [
        column
        for column in dataset.columns_metadata
        if any(aligned_span(text, name) for name in (column.original_name, column.sql_name))
    ]


#
# Devuelve las columnas de texto llamadas sólo como un lugar (sitio, localidad, estación).
#
def _place_columns(dataset):
    return {
        column.sql_name
        for column in dataset.columns_metadata
        if column.sql_type == "TEXT" and _content_words(column.original_name) in _PLACE_NAMES
    }


#
# Normaliza un texto a sus palabras de contenido en singular, sin conectores.
#
def _content_words(text):
    words = [word for word in normalize_text(text).split() if word not in MENTION_CONNECTORS]
    return singular_words(" ".join(words))


#
# Indica si el nombre original o el nombre SQL de la columna contiene el texto.
#
def _column_named(column, text):
    return contains_phrase(column.original_name, text, plural_insensitive=True) or contains_phrase(
        column.sql_name, text, plural_insensitive=True
    )


#
# Indica si la columna lleva el nombre de la variable; un conector interno de la mención no lo impide.
#
def _variable_named(column, text):
    if _column_named(column, text):
        return True
    if not _has_inner_connector(text):
        return False

    asked = " ".join(_content_words(text))
    return any(
        contains_phrase(" ".join(_content_words(name)), asked)
        for name in (column.original_name, column.sql_name)
    )


#
# Indica si hay un conector entre las palabras de un texto.
#
def _has_inner_connector(text):
    return any(word in MENTION_CONNECTORS for word in normalize_text(text).split()[1:-1])


#
# Indica si un valor guardado escribe la mención como palabras completas o con abreviaturas.
#
def _names_value(stored, value) -> bool:
    return contains_phrase(stored, value) or abbreviates(stored, value)


#
# Indica si un valor figura en los registros hallados como palabras completas.
#
def _stored_as_words(found, value):
    for rows in found.values():
        for row in rows.get(value, []):
            if isinstance(row["value"], str) and _names_value(row["value"], value):
                return True
    return False


class _Coverage:
    #
    # Guarda la evidencia y las menciones de valores, variables y dimensiones a cubrir.
    #
    def __init__(self, evidence, values, variables, dimensions):
        self._evidence = evidence
        self._values = values
        self._variables = variables
        self._dimensions = dimensions

    #
    # Indica si el dataset registra todos los valores pedidos.
    #
    def has_values(self, dataset):
        matches = self._evidence[dataset.dataset_id].value_matches
        return bool(self._values) and len(matches) == len(self._values)

    #
    # Indica si el dataset tiene columnas de todas las variables pedidas.
    #
    def has_variables(self, dataset):
        matches = self._evidence[dataset.dataset_id].variable_matches
        return bool(self._variables) and len(matches) == len(self._variables)

    #
    # Indica si el dataset tiene columnas para todas las dimensiones preguntadas.
    #
    def has_dimensions(self, dataset):
        matches = self._evidence[dataset.dataset_id].dimension_matches
        return len(matches) == len(self._dimensions)

    #
    # Indica si el dataset registra las variables y los valores pedidos, aunque falte la dimensión.
    #
    def covers_records(self, dataset):
        return (
            bool(self._values or self._variables)
            and (self.has_variables(dataset) or not self._variables)
            and (self.has_values(dataset) or not self._values)
        )

    #
    # Indica si el dataset cubre los valores, las variables y las dimensiones pedidos.
    #
    def complete(self, dataset):
        return (
            (bool(self._values) or bool(self._variables))
            and (self.has_values(dataset) or not self._values)
            and (self.has_variables(dataset) or not self._variables)
            and self.has_dimensions(dataset)
        )

    #
    # Cuenta las coincidencias de valores y variables del dataset.
    #
    def matches(self, dataset):
        item = self._evidence[dataset.dataset_id]
        return len(item.value_matches) + len(item.variable_matches)


class RetrieveDatasetsService:
    #
    # Guarda el índice semántico y las consultas de fuentes que usa el servicio.
    #
    def __init__(self, index: DatasetSemanticSearch, queries: SourceQueries) -> None:
        self._index = index
        self._queries = queries

    #
    # Recupera los datasets candidatos de una pregunta, por selección explícita o búsqueda semántica.
    #
    async def retrieve(
        self,
        question: str,
        *,
        limit: int = 5,
        dataset_bindings: list[DatasetBinding] | None = None,
        data_mentions=(),
        tentative_values=(),
        reselect=False,
    ) -> DatasetRetrievalResult:
        if not dataset_bindings:
            return await self._search(question, limit, data_mentions, tentative_values)

        result = await self._explicit(question, dataset_bindings)
        inherited = all(item.historical_version is not None for item in dataset_bindings)
        ranked = await self._rank(
            result, data_mentions, limit=None, restrict=inherited, tentative=tentative_values
        )
        if not inherited or (self._answers(ranked, data_mentions) and not reselect):
            return ranked

        requested = " ".join(dict.fromkeys(item.text for item in data_mentions))
        found = await self._search(
            question, limit, data_mentions, tentative_values, query=requested
        )
        return found if self._answers(found, data_mentions) else ranked

    #
    # Busca candidatos por similitud, los verifica en PostgreSQL y los ordena por evidencia.
    #
    async def _search(self, question, limit, data_mentions, tentative_values, *, query=None):
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

    #
    # Verifica en PostgreSQL las versiones de los datasets devueltos por la búsqueda.
    #
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

    #
    # Convierte un resultado de búsqueda en candidato si coincide con el dataset verificado.
    #
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

    #
    # Reúne los avisos de un candidato: índice, solapamiento, texto truncado y unidades.
    #
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

    #
    # Construye los avisos de la búsqueda sobre el alcance de la similitud y los descartes.
    #
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

    #
    # Ordena y recorta los candidatos por evidencia de valores, variables y dimensiones.
    #
    async def _rank(self, result, mentions, *, limit, restrict, tentative=()):
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

        result.omitted_candidates = self._omitted(candidates, ranked, coverage)
        result.warnings.extend(self._rule_notes(ranked, variables))
        result.candidates = ranked
        result.evidence = [
            evidence[item.dataset_id]
            for item in ranked
            if self._has_evidence(evidence[item.dataset_id])
        ]
        result.status = "candidates_available" if ranked else "no_available_candidates"
        return result

    #
    # Calcula, por dataset, las columnas que corresponden a cada variable pedida.
    #
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

    #
    # Devuelve las columnas de una variable en cada candidato.
    #
    @staticmethod
    def _variable_columns(candidates, variable):
        text = RetrieveDatasetsService._matched_text(candidates, variable)
        return {
            dataset.dataset_id: _variable_columns_named(dataset, text) for dataset in candidates
        }

    #
    # Elige el texto de la variable a buscar: la mención entera o su núcleo sin estadística.
    #
    @staticmethod
    def _matched_text(candidates, variable):
        if any(_variable_columns_named(dataset, variable) for dataset in candidates):
            return variable
        return variable_core(variable)

    #
    # Genera avisos de trazabilidad sobre columnas reconocidas por reglas o descartadas.
    #
    @staticmethod
    def _rule_notes(ranked, variables):
        notes = []
        for variable in variables:
            text = RetrieveDatasetsService._matched_text(ranked, variable)
            reasons = {}
            for dataset in ranked:
                kept, discarded = _variable_columns_found(dataset, text)
                names = {column.sql_name: column.original_name for column in dataset.columns_metadata}
                for column in kept:
                    if (reason := _match_reason(column, text)) is not None:
                        reasons.setdefault(reason, []).append(column.original_name)
                for column, reason in discarded.items():
                    reasons.setdefault(_DISCARD_REASONS[reason], []).append(names[column])

            described = [
                f"{reason}: " + ", ".join(f"«{name}»" for name in dict.fromkeys(names))
                for reason, names in reasons.items()
            ]
            if described:
                notes.append(f"Variable «{variable}»: " + "; ".join(described) + ".")
        return notes

    #
    # Separa las menciones que en realidad son dimensiones y registra sus columnas por dataset.
    #
    @staticmethod
    def _split_dimensions(result, candidates, evidence, values, variables, dimensions):
        with_variables = [
            dataset
            for dataset in candidates
            if variables and len(evidence[dataset.dataset_id].variable_matches) == len(variables)
        ]
        reclassified = [
            value
            for value in values
            if any(_named_columns(dataset, value) for dataset in with_variables)
        ]
        asked = [
            dimension
            for dimension in dimensions
            if any(_dimension_named(dataset, dimension) for dataset in with_variables)
        ]
        result.dimension_mentions = list(dict.fromkeys([*reclassified, *asked]))
        if not result.dimension_mentions:
            return values

        values = [value for value in values if value not in reclassified]
        for dataset in candidates:
            evidence[dataset.dataset_id].dimension_matches = {
                dimension: names
                for dimension in result.dimension_mentions
                if (names := _dimension_columns(dataset, dimension))
            }
        if reclassified:
            result.warnings.append(
                "Se interpretó como columna, no como valor de filtro: "
                + "; ".join(f"«{value}»" for value in reclassified)
                + "."
            )
        return values

    #
    # Busca en los registros de los candidatos los valores mencionados y los tentativos.
    #
    async def _stored_values(self, result, candidates, values, tentative):
        if not ((values or tentative) and candidates):
            return values, tentative, {}

        try:
            found = await asyncio.to_thread(
                self._queries.find_text_values, candidates, [*values, *tentative]
            )
        except Exception as error:
            _logger.warning(
                "No se pudieron verificar valores mencionados: %s", type(error).__name__
            )
            result.warnings.append("No se pudo comprobar en los registros los valores mencionados.")
            return [], [], {}
        return values, tentative, found

    #
    # Confirma como valores los tentativos que aparecen como palabras completas en los registros.
    #
    @staticmethod
    def _confirm_tentative(result, found, values, tentative):
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

    #
    # Registra las columnas y filas donde cada candidato contiene los valores pedidos.
    #
    @staticmethod
    def _record_value_matches(candidates, evidence, found, values):
        for dataset in candidates:
            item = evidence[dataset.dataset_id]
            places = _place_columns(dataset)
            for value in values:
                rows = found.get(dataset.dataset_id, {}).get(value, [])
                matches = RetrieveDatasetsService._value_matches(rows, value)
                canonical = [match for match in matches if match.column in places]
                matches = canonical or matches
                if matches:
                    item.value_matches[value] = matches

    #
    # Agrupa por columna las filas que contienen el valor pedido.
    #
    @staticmethod
    def _value_matches(rows, value):
        columns = {}
        for row in rows:
            if not (isinstance(row["value"], str) and _names_value(row["value"], value)):
                continue
            match = columns.setdefault(
                row["column"], ValueMatch(column=row["column"], values=[], rows=0)
            )
            match.values.append(row["value"])
            match.rows += row["rows"]
        return list(columns.values())

    #
    # Registra los valores que ningún candidato contiene y avisa de ello.
    #
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

    #
    # Restringe los candidatos a los que reúnen los valores, variables o columnas pedidos.
    #
    @staticmethod
    def _restricted(result, ranked, coverage):
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

    #
    # Lista los candidatos que cubren lo pedido pero quedaron fuera del recorte, con su motivo.
    #
    @staticmethod
    def _omitted(candidates, ranked, coverage):
        kept = {item.dataset_id for item in ranked}
        return [
            OmittedCandidate(
                dataset_id=item.dataset_id,
                name=item.name,
                source_name=item.provenance.source_name,
                reason="beyond_limit" if coverage.has_dimensions(item) else "missing_dimension",
            )
            for item in candidates
            if item.dataset_id not in kept and coverage.covers_records(item)
        ]

    #
    # Indica si un dataset tiene alguna evidencia de coincidencia.
    #
    @staticmethod
    def _has_evidence(item):
        return bool(item.value_matches or item.variable_matches or item.dimension_matches)

    #
    # Indica si algún dataset evidencia todas las variables y valores pedidos.
    #
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

    #
    # Descarta las hojas cuyas filas están completamente incluidas en otra candidata.
    #
    @staticmethod
    def _prune_overlaps(result):
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

    #
    # Devuelve la candidata que contiene todas las filas de un dataset solapado, si existe.
    #
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

    #
    # Verifica los datasets indicados por el usuario o el historial sin usar el índice.
    #
    async def _explicit(self, question, bindings):
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
