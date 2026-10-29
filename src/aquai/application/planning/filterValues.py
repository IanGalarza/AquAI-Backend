"""Resuelve la grafía de filtros sólo mediante valores reales y unívocos."""

import asyncio

from aquai.application.contracts.planning import FilterValueResolution, ValidationIssue


async def resolve_filter_values(decision, selected, operations, mentions, queries):
    requested = {item.text.casefold() for item in mentions if item.kind == "value"}
    if not requested:
        return decision, [], []
    normalized = decision.model_copy(deep=True)
    datasets = {item.dataset_id: item for item in selected}
    steps = {step.id: step for step in operations}
    resolutions, issues, cache = [], [], {}
    for step in normalized.steps:
        for predicate in step.filters:
            if predicate.operator not in {"eq", "in"}:
                continue
            field_ref = predicate.field
            if field_ref.kind == "column":
                refs = [field_ref]
            else:
                field = next(item for item in steps[field_ref.step_id].output_fields if item.name == field_ref.field)
                refs = field.sources if field.sql_type == "TEXT" else []
            if len(refs) != 1:
                continue
            ref = refs[0]
            dataset = datasets[ref.dataset_id]
            column = next(item for item in dataset.columns_metadata if item.sql_name == ref.column)
            if column.sql_type != "TEXT":
                continue
            for index, value in enumerate(predicate.values):
                if not isinstance(value, str) or value.casefold() not in requested:
                    continue
                key = (ref.dataset_id, ref.column, value)
                if key not in cache:
                    cache[key] = await asyncio.to_thread(queries.matching_column_values,
                        ref.dataset_id, dataset.verified_version, ref.column, value)
                matches = cache[key]
                if len(matches) > 1:
                    issues.append(ValidationIssue(code="ambiguous_filter_value", severity="clarification", step_id=step.id,
                        message=f"«{value}» coincide con varias variantes de un valor de columna; hay que precisar cuál corresponde."))
                elif matches and matches[0] != value:
                    predicate.values[index] = matches[0]
                    resolutions.append(FilterValueResolution(dataset_id=ref.dataset_id, column=ref.column,
                        requested=value, resolved=matches[0]))
    return normalized, resolutions, issues
