import asyncio
from typing import NamedTuple

from aquai.application.contracts.planning import (
    ColumnRef,
    FilterValueResolution,
    SelectedDataset,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import RetrievedColumn
from aquai.application.matching import abbreviates, contains_phrase

_MAX_EXPANDED_VALUES = 30
_EQUALITY_OPERATORS = {"eq", "ieq", "in"}


class _FilterTarget(NamedTuple):
    step_id: str
    ref: ColumnRef
    dataset: SelectedDataset
    column: RetrievedColumn


#
# Indica si un valor almacenado y una mención se igualan o se contienen como palabras.
#
def _overlaps(value, mention) -> bool:
    return (
        value.casefold() == mention.casefold()
        or contains_phrase(value, mention)
        or contains_phrase(mention, value)
        or abbreviates(value, mention)
    )


#
# Devuelve la mención pedida que coincide con un valor, si existe.
#
def _mention_for(value, requested):
    for item in requested:
        if _overlaps(value, item):
            return item

    return None


#
# Contrasta los filtros de texto de una decisión con los valores almacenados y los corrige.
#
async def resolve_filter_values(
    decision, selected, operations, mentions, queries, evidence=(), source_names=()
):
    resolver = _FilterValueResolver(selected, operations, mentions, queries, evidence, source_names)

    return await resolver.resolve(decision)


class _FilterValueResolver:
    #
    # Guarda los datasets, menciones y consultas necesarios para resolver filtros de texto.
    #
    def __init__(self, selected, operations, mentions, queries, evidence, source_names):
        self._requested = list(
            dict.fromkeys(item.text for item in mentions if item.kind == "value")
        )
        self._observed = {item.dataset_id: item.value_matches for item in evidence}
        self._datasets = {item.dataset_id: item for item in selected}
        self._steps = {step.id: step for step in operations}
        self._queries = queries
        self._source_names = source_names
        self._resolutions = []
        self._issues = []
        self._cache = {}

    #
    # Resuelve los filtros de texto de cada paso y devuelve la decisión, resoluciones y avisos.
    #
    async def resolve(self, decision):
        normalized = decision.model_copy(deep=True)
        removed = []

        for step in normalized.steps:
            for predicate in step.filters:
                target = self._text_target(step, predicate)
                if target is None:
                    continue

                values = await self._resolved_values(target, predicate)
                if not values:
                    removed.append((step, predicate))
                    continue

                self._apply(predicate, values, target.ref)

        for step, predicate in removed:
            step.filters = [item for item in step.filters if item is not predicate]

        return normalized, self._resolutions, self._issues

    #
    # Devuelve el destino de un filtro de igualdad sobre una columna TEXT, si corresponde.
    #
    def _text_target(self, step, predicate) -> _FilterTarget | None:
        if predicate.operator not in _EQUALITY_OPERATORS:
            return None

        refs = self._column_refs(predicate.field)
        if len(refs) != 1:
            return None

        ref = refs[0]
        dataset = self._datasets[ref.dataset_id]
        column = next(item for item in dataset.columns_metadata if item.sql_name == ref.column)
        if column.sql_type != "TEXT":
            return None

        return _FilterTarget(step.id, ref, dataset, column)

    #
    # Devuelve las columnas de origen a las que apunta la referencia de un filtro.
    #
    def _column_refs(self, field_ref) -> list:
        if field_ref.kind == "column":
            return [field_ref]

        field = next(
            item
            for item in self._steps[field_ref.step_id].output_fields
            if item.name == field_ref.field
        )

        return field.sources if field.sql_type == "TEXT" else []

    #
    # Devuelve los valores almacenados que resuelven todos los valores de un filtro.
    #
    async def _resolved_values(self, target: _FilterTarget, predicate) -> list:
        values = []

        for value in predicate.values:
            values.extend(await self._resolve_value(target, predicate.operator, value))

        return list(dict.fromkeys(values))

    #
    # Resuelve un valor de filtro con un valor almacenado y avisa si es ambiguo.
    #
    async def _resolve_value(self, target: _FilterTarget, operator: str, value) -> list:
        if not isinstance(value, str):
            return [value]

        step_id, ref, _dataset, column = target
        matches = await self._exact_matches(target, value)

        if len(matches) > 1:
            options = ", ".join(str(item) for item in matches)
            self._issues.append(
                ValidationIssue(
                    code="ambiguous_filter_value",
                    severity="clarification",
                    step_id=step_id,
                    message=f"{value} coincide con varios valores de la columna {column.original_name}: {options}. ¿Cuál querés usar?",
                )
            )
            return [value]

        if matches:
            if matches[0] != value or operator == "ieq":
                self._resolutions.append(
                    FilterValueResolution(
                        dataset_id=ref.dataset_id,
                        column=ref.column,
                        requested=value,
                        resolved=matches[0],
                    )
                )
            return [matches[0], *await self._abbreviated_siblings(target, value, matches[0])]

        return await self._resolve_by_containment(target, value)

    #
    # Devuelve las otras grafías abreviadas del mismo valor en la columna («Pta Lara» junto a «Punta Lara»),
    # que también se filtran y quedan registradas como resolución.
    #
    async def _abbreviated_siblings(self, target: _FilterTarget, value: str, exact) -> list:
        mention = _mention_for(value, self._requested) or value
        siblings = [
            item
            for item in await self._containing_values(target, mention)
            if item != exact and abbreviates(item, mention)
        ]
        self._resolutions.extend(
            FilterValueResolution(
                dataset_id=target.ref.dataset_id,
                column=target.ref.column,
                requested=value,
                resolved=item,
            )
            for item in siblings
        )
        return siblings

    #
    # Devuelve los valores almacenados de la columna que igualan al valor, usando caché.
    #
    async def _exact_matches(self, target: _FilterTarget, value: str) -> list:
        ref = target.ref
        key = (ref.dataset_id, ref.column, value)

        if key not in self._cache:
            self._cache[key] = await asyncio.to_thread(
                self._queries.matching_column_values,
                ref.dataset_id,
                target.dataset.verified_version,
                ref.column,
                value,
            )

        return self._cache[key]

    #
    # Resuelve un valor de filtro por los valores almacenados que lo contienen como palabras.
    #
    async def _resolve_by_containment(self, target: _FilterTarget, value: str) -> list:
        step_id, ref, dataset, column = target
        mention = _mention_for(value, self._requested) or value
        candidates = await self._containing_values(target, mention)

        if not candidates and any(contains_phrase(name, value) for name in self._source_names):
            self._issues.append(
                ValidationIssue(
                    code="source_name_filter_removed",
                    severity="warning",
                    step_id=step_id,
                    message=f"Se quitó el filtro {value} en la columna {column.original_name}: es el nombre de la fuente indicada.",
                )
            )
            return []

        if not candidates:
            self._issues.append(
                ValidationIssue(
                    code="filter_value_not_found",
                    severity="clarification",
                    step_id=step_id,
                    message=f"{value} no aparece en la columna {column.original_name} de {dataset.name}.",
                )
            )
            return [value]

        if len(candidates) > _MAX_EXPANDED_VALUES:
            sample = ", ".join(str(item) for item in candidates[:6])
            self._issues.append(
                ValidationIssue(
                    code="ambiguous_filter_value",
                    severity="clarification",
                    step_id=step_id,
                    message=f"{value} aparece en muchos valores de la columna {column.original_name} ({sample}, …). ¿Cuál querés usar?",
                )
            )
            return [value]

        self._resolutions.append(
            FilterValueResolution(
                dataset_id=ref.dataset_id,
                column=ref.column,
                requested=value,
                resolved=candidates,
                method="contains_words",
            )
        )

        return candidates

    #
    # Devuelve los valores almacenados que contienen la mención como palabras completas.
    #
    async def _containing_values(self, target: _FilterTarget, mention: str) -> list:
        ref = target.ref
        stored = [
            match
            for match in self._observed.get(ref.dataset_id, {}).get(mention, [])
            if match.column == ref.column
        ]

        if stored:
            candidates = [item for match in stored for item in match.values]
        else:
            candidates = await self._queried_values(target, mention)

        return [
            item
            for item in dict.fromkeys(candidates)
            if isinstance(item, str) and (contains_phrase(item, mention) or abbreviates(item, mention))
        ]

    #
    # Busca en la base los valores de texto de la columna que contienen la mención.
    #
    async def _queried_values(self, target: _FilterTarget, mention: str) -> list:
        ref = target.ref
        found = await asyncio.to_thread(
            self._queries.find_text_values,
            [target.dataset],
            [mention],
            per_column=_MAX_EXPANDED_VALUES + 1,
        )

        return [
            row["value"]
            for row in found.get(ref.dataset_id, {}).get(mention, [])
            if row["column"] == ref.column
        ]

    #
    # Reemplaza en el filtro el operador y los valores por los resueltos.
    #
    def _apply(self, predicate, values: list, ref) -> None:
        if values == list(predicate.values) and predicate.operator != "ieq":
            return

        literal = all(isinstance(item, str) for item in values)
        if (
            predicate.operator == "ieq"
            and len(values) == 1
            and not self._was_resolved(predicate, ref)
        ):
            return

        predicate.operator = "eq" if len(values) == 1 else "in"
        predicate.values = values if literal else predicate.values

    #
    # Indica si el valor del filtro ya tiene una resolución registrada para la columna.
    #
    def _was_resolved(self, predicate, ref) -> bool:
        return any(
            item.requested == predicate.values[0]
            for item in self._resolutions
            if item.column == ref.column
        )
