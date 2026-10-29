"""Contexto acotado y enunciados cuyos valores se toman de evidencia autorizada."""

from datetime import date, datetime
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
import re

from aquai.application.contracts.answering import ResponseContextRecord
from aquai.application.execution import decode_rows
from aquai.application.answering.presentation import format_number
from aquai.application.units import is_ph_name, presentation_unit_status


def response_language(question, preferred=None):
    if preferred is not None:
        return preferred
    words = set(re.findall(r"[a-záéíóúñ]+", question.lower()))
    english = len(words & {"what", "which", "show", "average", "mean", "please", "plot", "the", "rainfall", "how"})
    spanish = len(words & {"qué", "cuál", "mostrá", "promedio", "media", "gráfico", "graficá", "generá", "por", "los", "las", "del"})
    return "en" if english > spanish else "es"


def _value(value, language="es", *, exact=False):
    if value is None:
        return "null"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (Decimal, int, float)):
        if exact:
            return str(value).replace(".", ",") if language == "es" else str(value)
        return format_number(value, language)
    return str(value)  


def _label(field, sources):
    if len(field.sources) == 1:
        ref = field.sources[0]
        dataset = sources.get(ref.dataset_id)
        if dataset:
            column = next((item for item in dataset.columns_metadata if item.sql_name == ref.column), None)
            if column:
                label = " ".join(column.original_name.split())
                if is_ph_name(label):
                    return "pH"
                if column.unit:
                    for suffix in (f" [{column.unit}]", f" ({column.unit})"):
                        if label.endswith(suffix):
                            label = label[:-len(suffix)]
                            break
                return label if len(label) <= 200 else column.sql_name
    return field.name


def _unit(field, language):
    status = presentation_unit_status(field)
    if status == "documented" and field.unit:
        return f" {field.unit}"
    if status == "unknown" and field.sql_type in {"NUMERIC", "BIGINT"}:
        return " (unidad no documentada)" if language == "es" else " (unit not documented)"
    return ""


def _calculation_label(calculation, field, sources, language):
    variable = _label(field, sources)
    labels = {
        "es": {"count": "conteo", "count_distinct": "conteo de valores distintos", "sum": "suma", "mean": "promedio",
               "min": "mínimo", "max": "máximo", "median": "mediana", "stddev": "desvío estándar", "quantile": "cuantil",
               "difference": "diferencia", "ratio": "cociente", "correlation": "correlación de Pearson"},
        "en": {"count": "count", "count_distinct": "distinct count", "sum": "sum", "mean": "mean", "min": "minimum",
               "max": "maximum", "median": "median", "stddev": "standard deviation", "quantile": "quantile",
               "difference": "difference", "ratio": "ratio", "correlation": "Pearson correlation"},
    }
    if calculation is None:
        return variable
    if variable == "pH" and calculation.function == "mean":
        return "pH promedio" if language == "es" else "mean pH"
    if calculation.function == "count" and not calculation.inputs:
        return "cantidad de registros" if language == "es" else "number of rows"
    label = labels[language][calculation.function]
    if calculation.function in {"difference", "ratio", "correlation"} and len(field.sources) > 1:
        variables = [_label(field.model_copy(update={"sources": [ref]}), sources) for ref in field.sources]
        return label + (" entre " if language == "es" else " between ") + (" y " if language == "es" else " and ").join(variables)
    if calculation.function == "quantile":
        label += f" (q={calculation.quantile})"
    elif calculation.function == "stddev":
        label += f" (ddof={calculation.ddof})"
    return f"{label} de {variable}" if language == "es" else f"{label} of {variable}"


def _group_scope(grouping, fields, row, sources, language):
    parts = []
    places = {"localidad", "ciudad", "municipio", "partido", "provincia", "locality", "city"}
    for name, spec in grouping.items():
        label, value = _label(fields[name], sources), _value(row[name], language)
        if spec.time_grain != "raw":
            grain = {"es": {"year": "año", "month": "mes", "day": "día"},
                     "en": {"year": "year", "month": "month", "day": "day"}}[language][spec.time_grain]
            parts.append(f"en el {grain} {value}" if language == "es" else f"in {grain} {value}")
        elif label.casefold() in places and isinstance(row[name], str):
            if value.isupper():
                words = value.title().split()
                value = " ".join(word.lower() if index and word.lower() in {"de", "del", "la", "las", "los", "y"}
                                 else word for index, word in enumerate(words))
            parts.append(f"de {value}" if language == "es" else f"in {value}")
        else:
            parts.append(f"para {label} «{value}»" if language == "es" else f"for {label} '{value}'")
    return " " + ", ".join(parts) if parts else ""


def _source_intro(evidence, sources, language):
    names = list(dict.fromkeys(sources[key].provenance.source_name for key in evidence.dataset_ids))
    quoted = (" y " if language == "es" else " and ").join(f"«{name}»" for name in names)
    return (f"Según los registros de {quoted}, " if language == "es" else f"According to the records in {quoted}, ") if names else ""


def _latest_summary(rows, field, when, sources, language):
    label = _label(field, sources)
    if language == "es":
        day = when.strftime("%d/%m/%Y")
        if isinstance(when, datetime) and any((when.hour, when.minute, when.second, when.microsecond)):
            day += " " + when.time().isoformat()
    else:
        day = when.isoformat()
    counts = Counter(row[field.name] for row in rows)
    parts = [(f"{count} sin dato" if language == "es" else f"{count} without a value") if value is None else
             f"{count} {'con' if language == 'es' else 'with'} {_value(value, language)}{_unit(field, language)}"
             for value, count in counts.items()]
    values = (", ".join(parts[:-1]) + (" y " if language == "es" else " and ") + parts[-1]) if len(parts) > 1 else parts[0]
    if language == "es":
        noun = "registro" if len(rows) == 1 else "registros"
        return f"La fecha más reciente en los datos consultados es el {day}, con {len(rows)} {noun} de {label}: {values}."
    noun = "record" if len(rows) == 1 else "records"
    return f"The most recent date in the queried data is {day}, with {len(rows)} {noun} for {label}: {values}."


def build_response_context(state, budget_bytes, max_preview_rows=10, max_sentences=24):
    validation, plan = state.validation, state.planning
    language = response_language(state.question, state.response_language)
    steps = {step.id: step for step in plan.operations}
    sources = {item.dataset_id: item for item in plan.selected_datasets}
    records = {item.operation_id: item for item in state.execution.operations} if state.execution else {}
    if state.python_execution:
        records.update({item.operation_id: item for item in state.python_execution.operations})
    catalog = []
    summary_ids, summarized_in_detail = {}, set()
    body = {"question": state.question, "language": language, "status": validation.status,
        "conversation_context": [item.model_dump() for item in state.conversation_context],
        "history_is_not_evidence": True,
        "number_presentation": {"max_decimal_places": 2, "small_value_significant_digits": 2,
                                "decimal_separator": "," if language == "es" else ".",
                                "exact_values": "Unchanged in the referenced execution results."},
        "scope": "Only validated operations of the retrieved candidates; never the entire catalog.",
        "sources": [{"dataset_id": str(item.dataset_id), "source_id": str(item.source_id), "name": item.dataset_name,
                     "version": item.version} for item in validation.sources],
        "warnings": validation.warnings, "limitations": validation.limitations,
        "pending_operations": validation.pending_operations, "sentence_catalog": catalog,
        "filters": [{"step_id": step.id, "filters": [item.model_dump(mode="json") for item in step.filters]}
                    for step in plan.operations if step.filters]}

    def serialized():
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def add(evidence, direct, conversational=None, required=False):
        if len(catalog) >= max_sentences:
            return False
        item = {"sentence_id": f"s{len(catalog) + 1}", "evidence_ids": [evidence.evidence_id],
                "operation_id": evidence.operation_id,
                "direct": direct, "conversational": conversational or direct}
        catalog.append(item)
        if not required and len(serialized()) > budget_bytes:
            catalog.pop()
            return False
        return True

    # Todas las ramas autorizadas tienen una referencia aunque no quepan sus filas.
    for evidence in validation.usable_evidence:
        if evidence.kind == "visualization":
            chart = next(item for item in state.python_execution.visualizations if item.operation_id == evidence.operation_id)
            names = {"es": {"bar": "barras", "line": "líneas", "scatter": "dispersión", "histogram": "histograma"},
                     "en": {"bar": "bar", "line": "line", "scatter": "scatter", "histogram": "histogram"}}
            if language == "es":
                noun = "fila" if chart.input_rows == 1 else "filas"
                text = f"El gráfico de {names[language][chart.chart_type]} «{chart.title}» representa {chart.input_rows} {noun}."
                natural = f"Podés ver el gráfico «{chart.title}», con {chart.input_rows} {noun}."
                if chart.excluded_rows:
                    text += f" Se excluyeron {chart.excluded_rows} filas."
                    natural += f" Se excluyeron {chart.excluded_rows} filas."
            else:
                noun = "row" if chart.input_rows == 1 else "rows"
                text = f"The {names[language][chart.chart_type]} chart '{chart.title}' shows {chart.input_rows} {noun}."
                natural = f"You can view the chart '{chart.title}', based on {chart.input_rows} {noun}."
                if chart.excluded_rows:
                    text += f" {chart.excluded_rows} rows were excluded."
                    natural += f" {chart.excluded_rows} rows were excluded."
            add(evidence, text, natural, required=True)
        else:
            noun = ("fila" if evidence.rows == 1 else "filas") if language == "es" else ("row" if evidence.rows == 1 else "rows")
            text = (f"El resultado completo contiene {evidence.rows} {noun}."
                    if language == "es" else f"The complete result contains {evidence.rows} {noun}.")
            add(evidence, text, required=True)
            summary_ids[evidence.evidence_id] = catalog[-1]["sentence_id"]

    omitted = {}
    for evidence in validation.usable_evidence:
        if evidence.kind != "table":
            continue
        step = steps[evidence.operation_id]
        chain, visit = [], [step]
        while visit:
            ancestor = visit.pop()
            chain.append(ancestor)
            visit.extend(steps[key] for key in ancestor.depends_on)
        operators = {"es": {"eq": "igual a", "ieq": "igual a (sin distinguir mayúsculas)", "ne": "distinto de", "lt": "menor que", "le": "menor o igual que",
                            "gt": "mayor que", "ge": "mayor o igual que", "in": "incluido en",
                            "not_in": "fuera de", "is_null": "sin valor", "is_not_null": "con valor"},
                     "en": {"eq": "equal to", "ieq": "equal to (case insensitive)", "ne": "different from", "lt": "less than", "le": "less than or equal to",
                            "gt": "greater than", "ge": "greater than or equal to", "in": "included in",
                            "not_in": "not in", "is_null": "missing", "is_not_null": "not missing"}}
        filters = []
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                name = predicate.field.column if predicate.field.kind == "column" else predicate.field.field
                parent = ancestor if predicate.field.kind == "column" else steps[predicate.field.step_id]
                field = next((item for item in parent.output_fields if item.name == name), None)
                label = _label(field, sources) if field else name
                # Los umbrales definen el alcance exacto: no se redondean como
                # un resultado de presentación (p. ej., > 7,149 no es > 7,15).
                values = ", ".join(_value(value, language, exact=True) for value in predicate.values)
                filters.append(f"{label} {operators[language][predicate.operator]} {values}".rstrip())
        if filters:
            add(evidence, ("Se consideraron registros con " if language == "es" else "Records were selected with ") + "; ".join(filters) + ".", required=True)
        origin = step
        latest = None
        while origin.operation in {"sort", "latest"}:
            if origin.operation == "latest":
                latest = origin
            origin = steps[origin.depends_on[0]]
        calculations = {item.output_name: item for item in origin.calculations}
        grouping = {item.output_name: item for item in origin.group_by}
        record = records[evidence.operation_id]
        rows = decode_rows(record, record.columns)
        fields = {field.name: field for field in record.columns}
        represented = 0
        numeric = [field for field in record.columns if field.numeric_eligible and field.sql_type in {"NUMERIC", "BIGINT"}]
        if latest and rows and len(rows) <= max_preview_rows and len(numeric) == 1:
            when = rows[0][latest.order_by[0].field.field]
            text = _latest_summary(rows, numeric[0], when, sources, language)
            if add(evidence, text, _source_intro(evidence, sources, language) + text[:1].lower() + text[1:]):
                represented = len(rows)
        for row in ([] if represented else rows[:max_preview_rows]):
            scope = _group_scope(grouping, fields, row, sources, language)
            candidates = []
            for name, field in fields.items():
                if name in grouping:
                    continue
                label = _calculation_label(calculations.get(name), field, sources, language)
                value = _value(row[name], language)
                if row[name] is None:
                    description = (f"no hay un valor calculable de {label}{scope}" if language == "es"
                                   else f"there is no computable value for {label}{scope}")
                else:
                    if language == "es":
                        calculation = calculations.get(name)
                        feminine = calculation and (calculation.function in {"sum", "median", "difference", "correlation"}
                                                     or (calculation.function == "count" and not calculation.inputs))
                        subject = ("la " if feminine else "el ") + label if calculation else "el valor de " + label
                        description = f"{subject}{scope} fue de {value}{_unit(field, language)}"
                    else:
                        description = f"the {label}{scope} was {value}{_unit(field, language)}"
                candidates.append(description)
            # Una fila se incluye íntegra o se conserva sólo como referencia estructurada.
            if not candidates:
                candidates = [scope.strip()]
            text = "; ".join(candidates) + "."
            if add(evidence, text[:1].upper() + text[1:], _source_intro(evidence, sources, language) + text):
                represented += 1
            else:
                break
        if represented < len(rows):
            omitted[evidence.operation_id] = len(rows) - represented
        elif rows:
            summarized_in_detail.add(evidence.evidence_id)
        for field in record.columns:
            for ref in field.sources:
                dataset = sources[ref.dataset_id]
                column = next(item for item in dataset.columns_metadata if item.sql_name == ref.column)
                if column.derived_from and column.excluded_count is not None:
                    text = (f"Se utilizó una proyección numérica documentada de {column.derived_from}; el catálogo registra {column.excluded_count} exclusiones durante su preparación."
                            if language == "es" else f"A documented numeric projection of {column.derived_from} was used; the catalog records {column.excluded_count} exclusions during its preparation.")
                    if not any(item["direct"] == text and item["evidence_ids"] == [evidence.evidence_id] for item in catalog):
                        add(evidence, text)
        seen = set()
        for exclusion in record.exclusions:
            key = (str(exclusion.group), exclusion.input_rows, exclusion.valid_rows, exclusion.null_rows, exclusion.policy, exclusion.zero_divisions)
            if key in seen or not (exclusion.null_rows or exclusion.zero_divisions):
                continue
            seen.add(key)
            group = ", ".join(f"{name}={_value(value, language)}" for name, value in exclusion.group.items())
            if language == "es":
                text = f"{group + ': ' if group else ''}Se registraron {exclusion.valid_rows} entradas completas, {exclusion.null_rows} entradas nulas y {exclusion.zero_divisions} divisiones por cero, con política {exclusion.policy}."
            else:
                text = f"{group + ': ' if group else ''}{exclusion.valid_rows} complete inputs, {exclusion.null_rows} null inputs and {exclusion.zero_divisions} zero divisions were recorded, with policy {exclusion.policy}."
            add(evidence, text)
    # No repite un resumen técnico de filas cuando todos sus valores ya se explican.
    suppressed = {summary_ids[key] for key in summarized_in_detail}
    catalog[:] = [item for item in catalog if item["sentence_id"] not in suppressed]
    body["omitted_rows"] = omitted
    # El historial es contexto lingüístico opcional, nunca sustituye evidencia.
    while body["conversation_context"] and len(serialized()) > budget_bytes:
        body["conversation_context"].pop(0)
    content = serialized()
    return ResponseContextRecord(prompt_version="aquai-response-v2", evidence=body, sha256=sha256(content).hexdigest(), size_bytes=len(content),
        budget_bytes=budget_bytes, omitted_rows=omitted), language
