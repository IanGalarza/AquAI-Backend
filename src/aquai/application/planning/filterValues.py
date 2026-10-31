import asyncio
from typing import NamedTuple

from aquai.application.contracts.planning import (
    ColumnRef,
    FilterValueResolution,
    SelectedDataset,
    ValidationIssue,
)
from aquai.application.contracts.retrieval import RetrievedColumn
from aquai.application.matching import contains_phrase

_MAX_EXPANDED_VALUES = 30
_EQUALITY_OPERATORS = {"eq", "ieq", "in"}


class _FilterTarget(NamedTuple):
    step_id: str
    ref: ColumnRef
    dataset: SelectedDataset
    column: RetrievedColumn


def _overlaps(value, mention) -> bool:
    return (
        value.casefold() == mention.casefold()
        or contains_phrase(value, mention)
        or contains_phrase(mention, value)
    )


def _mention_for(value, requested):
    for item in requested:
        if _overlaps(value, item):
            return item

    return None


async def resolve_filter_values(
    decision, selected, operations, mentions, queries, evidence=(), source_names=()
):
    resolver = _FilterValueResolver(selected, operations, mentions, queries, evidence, source_names)

    return await resolver.resolve(decision)


class _FilterValueResolver:
    """Toda constante de texto se contrasta con valores almacenados, provenga o no de una
    mención: un filtro sobre un valor inexistente devolvería un resultado vacío «válido»."""

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

    def _column_refs(self, field_ref) -> list:
        if field_ref.kind == "column":
            return [field_ref]

        field = next(
            item
            for item in self._steps[field_ref.step_id].output_fields
            if item.name == field_ref.field
        )

        return field.sources if field.sql_type == "TEXT" else []

    async def _resolved_values(self, target: _FilterTarget, predicate) -> list:
        values = []

        for value in predicate.values:
            values.extend(await self._resolve_value(target, predicate.operator, value))

        return list(dict.fromkeys(values))

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
            return [matches[0]]

        return await self._resolve_by_containment(target, value)

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

    async def _resolve_by_containment(self, target: _FilterTarget, value: str) -> list:
        step_id, ref, dataset, column = target
        mention = _mention_for(value, self._requested) or value
        candidates = await self._containing_values(target, mention)

        if not candidates and any(contains_phrase(name, value) for name in self._source_names):
            # El usuario nombró así la fuente: no es un valor de registros que deba filtrarse.
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

    async def _containing_values(self, target: _FilterTarget, mention: str) -> list:
        # Sin igualdad: valores almacenados que contienen la mención como palabras completas
        # (p. ej. un código de campaña antepuesto al nombre del sitio).
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
            if isinstance(item, str) and contains_phrase(item, mention)
        ]

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

    def _apply(self, predicate, values: list, ref) -> None:
        if values == list(predicate.values) and predicate.operator != "ieq":
            return

        literal = all(isinstance(item, str) for item in values)
        if (
            predicate.operator == "ieq"
            and len(values) == 1
            and not self._was_resolved(predicate, ref)
        ):
            return  # ieq sin valor almacenado verificado se conserva tal cual.

        predicate.operator = "eq" if len(values) == 1 else "in"
        predicate.values = values if literal else predicate.values

    def _was_resolved(self, predicate, ref) -> bool:
        return any(
            item.requested == predicate.values[0]
            for item in self._resolutions
            if item.column == ref.column
        )
