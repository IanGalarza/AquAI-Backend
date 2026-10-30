"""Resuelve la grafía de filtros sólo mediante valores reales almacenados."""

import asyncio

from aquai.application.contracts.planning import FilterValueResolution, ValidationIssue
from aquai.application.matching import contains_phrase


_MAX_EXPANDED_VALUES = 30


def _mention_for(value, requested):
    return next((item for item in requested if value.casefold() == item.casefold()
                 or contains_phrase(value, item) or contains_phrase(item, value)), None)


async def resolve_filter_values(decision, selected, operations, mentions, queries, evidence=(), source_names=()):
    # Toda constante de texto se contrasta con valores almacenados, provenga o no de una
    # mención: un filtro sobre un valor inexistente devolvería un resultado vacío «válido».
    requested = list(dict.fromkeys(item.text for item in mentions if item.kind == "value"))
    observed = {item.dataset_id: item.value_matches for item in evidence}
    normalized = decision.model_copy(deep=True)
    datasets = {item.dataset_id: item for item in selected}
    steps = {step.id: step for step in operations}
    resolutions, issues, cache, removed = [], [], {}, []
    for step in normalized.steps:
        for predicate in step.filters:
            if predicate.operator not in {"eq", "ieq", "in"}:
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
            values = []
            for value in predicate.values:
                if not isinstance(value, str):
                    values.append(value)
                    continue
                mention = _mention_for(value, requested) or value
                key = (ref.dataset_id, ref.column, value)
                if key not in cache:
                    cache[key] = await asyncio.to_thread(queries.matching_column_values,
                        ref.dataset_id, dataset.verified_version, ref.column, value)
                matches = cache[key]
                if len(matches) > 1:
                    options = ", ".join(f"«{item}»" for item in matches)
                    issues.append(ValidationIssue(code="ambiguous_filter_value", severity="clarification", step_id=step.id,
                        message=f"«{value}» coincide con varios valores de «{column.original_name}»: {options}. ¿Cuál querés usar?"))
                    values.append(value)
                    continue
                if matches:
                    values.append(matches[0])
                    if matches[0] != value or predicate.operator == "ieq":
                        resolutions.append(FilterValueResolution(dataset_id=ref.dataset_id, column=ref.column,
                            requested=value, resolved=matches[0]))
                    continue
                # Sin igualdad: valores almacenados que contienen la mención como palabras completas
                # (p. ej. un código de campaña antepuesto al nombre del sitio).
                stored = [match for match in observed.get(ref.dataset_id, {}).get(mention, [])
                          if match.column == ref.column]
                if stored:
                    candidates = [item for match in stored for item in match.values]
                else:
                    found = await asyncio.to_thread(queries.find_text_values, [dataset], [mention],
                                                    per_column=_MAX_EXPANDED_VALUES + 1)
                    candidates = [row["value"] for row in found.get(ref.dataset_id, {}).get(mention, [])
                                  if row["column"] == ref.column]
                candidates = [item for item in dict.fromkeys(candidates)
                              if isinstance(item, str) and contains_phrase(item, mention)]
                if not candidates and any(contains_phrase(name, value) for name in source_names):
                    # El usuario nombró así la fuente: no es un valor de registros que deba filtrarse.
                    issues.append(ValidationIssue(code="source_name_filter_removed", severity="warning", step_id=step.id,
                        message=f"Se quitó el filtro «{value}» en «{column.original_name}»: es el nombre de la fuente indicada."))
                elif not candidates:
                    issues.append(ValidationIssue(code="filter_value_not_found", severity="clarification", step_id=step.id,
                        message=f"«{value}» no aparece en la columna «{column.original_name}» de «{dataset.name}»."))
                    values.append(value)
                elif len(candidates) > _MAX_EXPANDED_VALUES:
                    sample = ", ".join(f"«{item}»" for item in candidates[:6])
                    issues.append(ValidationIssue(code="ambiguous_filter_value", severity="clarification", step_id=step.id,
                        message=f"«{value}» aparece en muchos valores de «{column.original_name}» ({sample}, …). ¿Cuál querés usar?"))
                    values.append(value)
                else:
                    values.extend(candidates)
                    resolutions.append(FilterValueResolution(dataset_id=ref.dataset_id, column=ref.column,
                        requested=value, resolved=candidates, method="contains_words"))
            values = list(dict.fromkeys(values))
            if not values:
                removed.append((step, predicate))
                continue
            if values != list(predicate.values) or predicate.operator == "ieq":
                literal = all(isinstance(item, str) for item in values)
                if predicate.operator == "ieq" and len(values) == 1 and not any(
                        item.requested == predicate.values[0] for item in resolutions if item.column == ref.column):
                    continue  # ieq sin valor almacenado verificado se conserva tal cual.
                predicate.operator = "eq" if len(values) == 1 else "in"
                predicate.values = values if literal else predicate.values
    for step, predicate in removed:
        step.filters = [item for item in step.filters if item is not predicate]
    return normalized, resolutions, issues
