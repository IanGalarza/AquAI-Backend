import json
from collections import Counter
from datetime import date
from decimal import Decimal
from hashlib import sha256
from itertools import count as counter

from aquai.application.answering.naming import FieldNames
from aquai.application.answering.presentation import format_number
from aquai.application.answering.wording import (
    INLINE_FIELDS,
    NUMERIC_TYPES,
    at,
    calculation_meaning,
    column_label,
    count_text,
    date_text,
    dated,
    display_unit,
    file_stem,
    grain,
    grain_label,
    humanize,
    is_place,
    join_words,
    number_words,
    period,
    place_text,
    sentence,
    unit_suffix,
    with_article,
)
from aquai.application.contracts.answering import ResponseContextRecord
from aquai.application.execution import decode_rows, execution_records
from aquai.application.Rules.language import ENGLISH_CUES, SPANISH_CUES, WORD

_EQUALITY_OPERATORS = {"eq", "ieq", "in"}
_RANGE_OPERATORS = {"lt", "le", "gt", "ge"}
_DATE_TYPES = {"DATE", "TIMESTAMP"}
_ORDERING_OPERATIONS = {"sort", "latest"}
_MAX_LABEL_LENGTH = 200
_SPANISH_RELATIONS = {
    "ne": "distinto de ",
    "lt": "menor que ",
    "le": "menor o igual que ",
    "gt": "mayor que ",
    "ge": "mayor o igual que ",
}
_ENGLISH_RELATIONS = {
    "ne": "other than ",
    "lt": "below ",
    "le": "at most ",
    "gt": "above ",
    "ge": "at least ",
}


def response_language(question, preferred=None):
    if preferred is not None:
        return preferred

    words = set(WORD.findall(question.lower()))
    english = len(words & ENGLISH_CUES)
    spanish = len(words & SPANISH_CUES)

    return "en" if english > spanish else "es"


def build_response_context(state, budget_bytes, max_preview_rows=3, max_sentences=24):
    return _ContextBuilder(state, budget_bytes, max_preview_rows, max_sentences).build()


class _ContextBuilder:
    def __init__(self, state, budget_bytes, max_preview_rows, max_sentences):
        plan = state.planning
        self._state = state
        self._validation = state.validation
        self._budget_bytes = budget_bytes
        self._max_preview_rows = max_preview_rows
        self._max_sentences = max_sentences

        self._language = response_language(state.question, state.response_language)
        self._steps = {step.id: step for step in plan.operations}
        self._sources = {item.dataset_id: item for item in plan.selected_datasets}
        self._files = _file_count(self._validation.usable_evidence, self._sources)
        self._multi_source = self._files > 1
        temporal = {
            (item.dataset_id, item.column): item.day_first for item in plan.temporal_text_columns
        }
        self._writer = _Writer(self._sources, self._steps, self._language, temporal)
        self._records = execution_records(state)

        self._catalog = []
        self._body = self._initial_body()
        # Identificadores únicos aunque luego se retiren las referencias genéricas de tablas descritas.
        self._sentence_ids = counter(1)
        self._summaries = {}
        self._omitted = {}
        self._meanings = {}

    def build(self):
        if self._multi_source:
            self._body["introduction"] = self._writer.introduction(self._files)

        self._add_references()
        self._add_table_sentences()
        self._add_meanings()

        return self._finish()

    def _initial_body(self):
        state = self._state
        validation = self._validation
        plan = state.planning

        return {
            "question": state.question,
            "language": self._language,
            "status": validation.status,
            "conversation_context": [item.model_dump() for item in state.conversation_context],
            "history_is_not_evidence": True,
            "number_presentation": {
                "max_decimal_places": 2,
                "small_value_significant_digits": 2,
                "decimal_separator": "," if self._language == "es" else ".",
                "exact_values": "Unchanged in the referenced execution results.",
            },
            "scope": "Only validated operations of the retrieved candidates; never the entire catalog.",
            "sources": [
                {
                    "dataset_id": str(item.dataset_id),
                    "source_id": str(item.source_id),
                    "name": item.dataset_name,
                    "version": item.version,
                }
                for item in validation.sources
            ],
            "warnings": validation.warnings,
            "limitations": validation.limitations,
            "pending_operations": validation.pending_operations,
            "sentence_catalog": self._catalog,
            "filters": [
                {
                    "step_id": step.id,
                    "filters": [item.model_dump(mode="json") for item in step.filters],
                }
                for step in plan.operations
                if step.filters
            ],
        }

    def _serialized(self):
        return json.dumps(self._body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def _over_budget(self):
        return len(self._serialized()) > self._budget_bytes

    def _add(self, evidence, text, *, kind="result", attach=False, required=False, attribute=True):
        # Alcances y referencias obligatorias nunca se descartan por el tope ni por el presupuesto.
        if not required and len(self._catalog) >= self._max_sentences:
            return None

        if self._multi_source and kind == "result" and attribute:
            text = self._writer.attributed(evidence, text)

        entry = {
            "sentence_id": f"s{next(self._sentence_ids)}",
            "evidence_ids": [evidence.evidence_id],
            "operation_id": evidence.operation_id,
            "kind": kind,
            "direct": text,
            "conversational": text,
            "attachment": evidence.evidence_id if attach else None,
        }
        self._catalog.append(entry)

        if not required and self._over_budget():
            self._catalog.pop()
            return None

        return entry

    def _add_references(self):
        """Todas las ramas autorizadas tienen una referencia aunque no quepan sus enunciados."""
        for evidence in self._validation.usable_evidence:
            if evidence.kind == "visualization":
                self._add_chart(evidence)
                continue

            summary = self._writer.summary(evidence.rows)
            self._summaries[evidence.evidence_id] = self._add(
                evidence, summary, kind="summary", attach=True, required=True
            )

    def _add_chart(self, evidence):
        chart = next(
            item
            for item in self._state.python_execution.visualizations
            if item.operation_id == evidence.operation_id
        )
        lineage = _lineage(self._steps[evidence.operation_id], self._steps)
        limits = [step.limit for step in lineage if step.limit]
        limit = min(limits) if limits else None

        self._add(
            evidence,
            self._writer.chart(chart, limit),
            kind="chart",
            attach=True,
            required=True,
        )

    def _add_table_sentences(self):
        # Mismo orden que la respuesta: los resultados derivados antes que las filas leídas que los respaldan.
        ordered = sorted(
            self._validation.usable_evidence,
            key=lambda item: self._steps[item.operation_id].operation == "read",
        )
        for evidence in ordered:
            if evidence.kind == "table":
                self._add_table(evidence)

    def _add_table(self, evidence):
        step = self._steps[evidence.operation_id]
        chain = _lineage(step, self._steps)
        filtered = self._writer.result_filters(chain)
        if filtered is not None:
            self._add(evidence, filtered, kind="scope", required=True)

        origin, latest, ranking = _trace_origin(step, self._steps)
        record = self._records[evidence.operation_id]
        rows = decode_rows(record, record.columns)
        fields = {field.name: field for field in record.columns}

        text, attach = self._table_sentence(
            evidence, record, rows, fields, origin, latest, ranking, chain
        )
        entry = self._add(evidence, text, attach=attach, attribute=bool(rows))
        self._drop_summary(evidence, entry)
        if entry is None or attach:
            self._omitted[evidence.operation_id] = len(rows)

        self._collect_meanings(origin, rows, evidence)
        self._add_details(evidence, record, origin, fields)

    def _table_sentence(self, evidence, record, rows, fields, origin, latest, ranking, chain):
        writer = self._writer
        calculations = {item.output_name: item for item in origin.calculations}
        grouping = {item.output_name: item for item in origin.group_by}
        numeric = [
            field
            for field in record.columns
            if field.numeric_eligible and field.sql_type in NUMERIC_TYPES
        ]

        if not rows:
            return writer.empty(evidence), False

        if latest and len(rows) <= self._max_preview_rows and len(numeric) == 1:
            date_field = fields[latest.order_by[0].field.field]
            return writer.latest(rows, numeric[0], date_field, fields, latest, chain)

        if ranking and not calculations and not grouping:
            ranked = writer.ranking(rows, ranking, fields, self._max_preview_rows, chain)
            if ranked is not None:
                return ranked

        direction = _sort_direction(ranking, fields)

        return writer.rows(
            rows, fields, calculations, grouping, direction, self._max_preview_rows, chain
        )

    def _drop_summary(self, evidence, entry):
        summary = self._summaries.get(evidence.evidence_id)
        if entry is not None and summary in self._catalog:
            self._catalog.remove(summary)

    def _collect_meanings(self, origin, rows, evidence):
        # Una estadística sólo se define si produjo algún valor; se explica junto al primer resultado que la usa.
        for calculation in origin.calculations:
            meaning = self._writer.meaning(calculation)
            if meaning and _has_value(rows, calculation.output_name):
                self._meanings.setdefault(meaning, evidence)

    def _add_details(self, evidence, record, origin, fields):
        projections = self._writer.projections(record)
        exclusions = self._writer.exclusions(record, origin, fields)
        for detail in [*projections, *exclusions]:
            self._add(evidence, detail, kind="detail")

    def _add_meanings(self):
        # Las definiciones son contexto: entran al final y sólo si queda lugar, sin desplazar resultados ni exclusiones.
        for meaning, evidence in self._meanings.items():
            self._add(evidence, meaning, kind="meaning")

    def _finish(self):
        self._body["omitted_rows"] = self._omitted

        # El historial es contexto lingüístico opcional, nunca sustituye evidencia.
        history = self._body["conversation_context"]
        while history and self._over_budget():
            history.pop(0)

        content = self._serialized()
        record = ResponseContextRecord(
            evidence=self._body,
            sha256=sha256(content).hexdigest(),
            size_bytes=len(content),
            budget_bytes=self._budget_bytes,
            omitted_rows=self._omitted,
        )

        return record, self._language


class _Writer:
    def __init__(self, sources, steps, language, temporal):
        self._sources = sources
        self._steps = steps
        self._temporal = temporal
        self.language = language
        self.es = language == "es"
        self.names = FieldNames(steps, self.column, language)

    def label(self, field):
        source = _single_source(field)
        label = self.column(*source) if source else None

        return label or humanize(field.name)

    def column(self, dataset_id, sql_name):
        dataset = self._sources.get(dataset_id)
        column = _column_metadata(dataset, sql_name) if dataset else None
        if not column:
            return None

        label = column_label(column.original_name, column.unit)

        return label if len(label) <= _MAX_LABEL_LENGTH else column.sql_name

    def value(self, value, *, exact=False):
        if value is None:
            return "sin dato" if self.es else "no value"
        if isinstance(value, date):
            return date_text(value, self.language)
        if type(value) is bool:
            return self._yes_no(value)
        if not isinstance(value, (Decimal, int, float)):
            return str(value)
        if exact:
            # Los umbrales definen el alcance exacto: > 7,149 no es > 7,15.
            return str(value).replace(".", ",") if self.es else str(value)

        return format_number(value, self.language)

    def _yes_no(self, value):
        if self.es:
            return "sí" if value else "no"

        return "yes" if value else "no"

    def measure(self, field, value):
        shown = self.value(value)
        if value is None:
            return shown

        return shown + unit_suffix(field, self.language)

    def count(self, total, singular, plural):
        return count_text(total, singular, plural, self.language)

    def subject(self, calculation):
        return with_article(*self.names.calculation(calculation), self.language)

    def when(self, field, value, *, temporal=False):
        if isinstance(value, date):
            return date_text(value, self.language), "day"

        key = _single_source(field)
        is_period = isinstance(value, str) and (temporal or key in self._temporal)
        if not is_period:
            return None

        return period(value, self.language, self._temporal.get(key))

    def attribute(self, field, value):
        """Lugar, fecha u otra categoría que ubica un valor: en Punta Lara, el 12/03/2008."""
        label = self.label(field)

        moment = self.when(field, value)
        if moment is not None:
            return at(*moment, self.language)
        if is_place(label) and isinstance(value, str):
            return self._located(place_text(value))

        shown = self._displayed(field, value)

        return f"con {label} {shown}" if self.es else f"with {label} {shown}"

    def _located(self, places):
        return ("en " if self.es else "in ") + places

    def _displayed(self, field, value):
        return value if isinstance(value, str) else self.measure(field, value)

    def context(self, row, fields, skip):
        attributes = [
            self.attribute(field, row[name])
            for name, field in fields.items()
            if name not in skip and row[name] is not None
        ]

        return ", ".join(attributes)

    def scope(self, spec, field, value):
        """Complemento de un grupo: en Punta Lara, en 2008, con Especie Microcystis aeruginosa."""
        if value is None:
            label = self.label(field)
            return f"sin dato de {label}" if self.es else f"without a {label} value"
        if spec.time_grain != "raw":
            return at(*grain(value, spec.time_grain, self.language), self.language)

        return self.attribute(field, value)

    def group_scope(self, grouping, fields, row):
        scopes = [self.scope(spec, fields[name], row[name]) for name, spec in grouping.items()]

        return join_words(scopes, self.language)

    def pair(self, field, value):
        label = self.label(field)
        if value is None:
            return f"{label} sin dato" if self.es else f"{label} without a value"

        moment = self.when(field, value)
        if moment is not None:
            return f"{label} {moment[0]}"

        return f"{label} {self._displayed(field, value)}"

    def attributed(self, evidence, text):
        """Con varias fuentes, cada resultado nombra la suya para no atribuirle un valor a otra."""
        preposition = "En" if self.es else "In"

        return f"{preposition} {self.files(evidence)}, {_lowercase_first(text)}"

    def files(self, evidence):
        names = _unique(
            file_stem(self._sources[key].provenance.source_name) for key in evidence.dataset_ids
        )
        single = len(names) == 1
        if self.es:
            return ("el archivo " if single else "los archivos ") + join_words(names, "es")

        return ("the file " if single else "the files ") + join_words(names, "en")

    def introduction(self, files):
        if self.es:
            return (
                f"Encontré información en {number_words(files)} archivos distintos, "
                "así que te muestro el resultado de cada uno por separado."
            )

        return (
            f"I found information in {number_words(files, 'en')} different files, "
            "so I show the result of each one separately."
        )

    def meaning(self, calculation):
        return calculation_meaning(
            calculation.function, self.language, quantile=calculation.quantile
        )

    def summary(self, rows):
        if self.es:
            if rows == 1:
                return "Encontré un resultado; podés verlo en la tabla."
            return (
                f"Encontré {self.count(rows, 'resultado', 'resultados')}; "
                "podés verlos en la tabla."
            )

        if rows == 1:
            return "I found one result; you can see it in the table."
        return f"I found {self.count(rows, 'result', 'results')}; you can see them in the table."

    def empty(self, evidence):
        if self.es:
            return f"No encontré registros que cumplan esos criterios en {self.files(evidence)}."

        return f"No records match those criteria in {self.files(evidence)}."

    def chart(self, chart, limit):
        rows = chart.input_rows
        excluded = chart.excluded_rows
        text = self._chart_description(chart.chart_type, rows, excluded)

        if limit and rows >= limit:
            text += self._limit_note(limit)
        if excluded:
            text += self._excluded_note(excluded)

        return text

    def _chart_description(self, chart_type, rows, excluded):
        if self.es:
            shown = (
                "el resultado"
                if rows == 1
                else f"los {self.count(rows, 'resultado', 'resultados')}"
            )
            values = self.count(rows - excluded, "valor", "valores")
            descriptions = {
                "histogram": f"El histograma muestra cómo se distribuyen {values}: "
                "cada barra indica cuántos valores caen dentro de un intervalo.",
                "bar": f"El gráfico de barras compara {shown}: "
                "la altura de cada barra es su valor.",
                "line": f"El gráfico de líneas muestra cómo cambian {shown} "
                "a lo largo del eje horizontal.",
                "scatter": f"El gráfico de dispersión ubica {shown} como puntos, "
                "para ver si las dos variables se relacionan.",
            }
        else:
            shown = self.count(rows, "result", "results")
            values = self.count(rows - excluded, "value is", "values are")
            descriptions = {
                "histogram": f"The histogram shows how {values} distributed: "
                "each bar counts the values that fall within an interval.",
                "bar": f"The bar chart compares {shown}: the height of each bar is its value.",
                "line": f"The line chart shows how {shown} change along the horizontal axis.",
                "scatter": f"The scatter plot shows {shown} as points, "
                "to see whether the two variables are related.",
            }

        return descriptions[chart_type]

    def _limit_note(self, limit):
        if self.es:
            return (
                f" La consulta se limitó a {self.count(limit, 'resultado', 'resultados')}, "
                "así que no incluye todos los registros."
            )

        return (
            f" The query was limited to {self.count(limit, 'result', 'results')}, "
            "so it does not include every record."
        )

    def _excluded_note(self, excluded):
        single = excluded == 1
        if self.es:
            left = "Quedó" if single else "Quedaron"
            records = self.count(excluded, "registro", "registros")
            replaced = "reemplazó" if single else "reemplazaron"
            return (
                f" {left} fuera {records} sin datos válidos para graficar; "
                f"no se {replaced} por cero."
            )

        records = self.count(excluded, "record", "records")
        verb = "was" if single else "were"

        return (
            f" {records} without valid data {verb} left out; "
            "missing values were not replaced with zero."
        )

    def qualifier(self, chain, shown=frozenset()):
        """Filtros que acotan el resultado y la oración no muestra: en Punta Lara, con Turbidez mayor que 10 NTU.

        Que un registro tenga dato de la variable medida es obvio para el lector y no se repite."""
        parts = []
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                if not _goes_without_saying(predicate, shown):
                    parts.append(self._restriction(predicate, ancestor))

        parts = _unique(parts)

        return (" " + ", ".join(parts)) if parts else ""

    def _restriction(self, predicate, ancestor):
        ref = predicate.field
        label = self.names.column(ref.dataset_id, ref.column)
        if _lists_places(predicate, label):
            places = [place_text(value) for value in predicate.values]
            return self._located(join_words(places, self.language))

        field = _source_field(ancestor, ref)

        return self.condition(predicate, label, field)

    def result_filters(self, chain):
        """Un filtro sobre resultados ya calculados cambia qué filas se ven: se dice siempre."""
        conditions = []
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                if predicate.field.kind == "result":
                    conditions.append(self._result_condition(predicate))

        conditions = _unique(conditions)
        if not conditions:
            return None
        if self.es:
            return f"Sólo incluí los resultados {join_words(conditions, 'es')}."

        return f"I only included the results {join_words(conditions, 'en')}."

    def _result_condition(self, predicate):
        ref = predicate.field
        field = _named_field(self._steps[ref.step_id], ref.field)
        label = self.names.result(ref.step_id, ref.field, field)

        return self.condition(predicate, label, field)

    def condition(self, predicate, label, field):
        operator = predicate.operator
        values = self._condition_values(predicate, field)

        date_bound = self._date_bound(predicate, field)
        if date_bound is not None:
            return date_bound

        preposition = "con" if self.es else "with"
        if operator == "is_null":
            return f"sin dato de {label}" if self.es else f"without a {label} value"
        if operator == "is_not_null":
            return f"con un valor de {label}" if self.es else f"with a {label} value"
        if operator == "in":
            conjunction = "o" if self.es else "or"
            return f"{preposition} {label} {join_words(values, self.language, conjunction)}"
        if operator == "not_in":
            excluding = "excluyendo" if self.es else "excluding"
            return f"{excluding} {label} {join_words(values, self.language)}"

        relation = self._relation(predicate)

        return f"{preposition} {label} {relation}{values[0]}"

    def _condition_values(self, predicate, field):
        unit, documented = (
            display_unit(field, self.language) if field is not None else (None, False)
        )
        suffix = f" {unit}" if documented else ""

        return [
            value if isinstance(value, str) else self.value(value, exact=True) + suffix
            for value in predicate.values
        ]

    def _date_bound(self, predicate, field):
        is_date_range = (
            field is not None
            and field.sql_type in _DATE_TYPES
            and predicate.operator in _RANGE_OPERATORS
        )
        if not is_date_range:
            return None

        moment = _iso_date(predicate.values[0])
        if moment is None:
            return None

        shown = date_text(moment, self.language)
        if self.es:
            bounds = {
                "ge": f"desde el {shown}",
                "gt": f"después del {shown}",
                "le": f"hasta el {shown}",
                "lt": f"antes del {shown}",
            }
        else:
            bounds = {
                "ge": f"from {shown}",
                "gt": f"after {shown}",
                "le": f"until {shown}",
                "lt": f"before {shown}",
            }

        return bounds[predicate.operator]

    def _relation(self, predicate):
        relations = _SPANISH_RELATIONS if self.es else _ENGLISH_RELATIONS
        relation = relations.get(predicate.operator)
        if relation is not None:
            return relation
        if isinstance(predicate.values[0], str):
            return ""

        return "igual a " if self.es else "equal to "

    def latest(self, rows, field, date_field, fields, step, chain):
        """Si la fecha tiene varios registros, la tabla muestra a qué corresponde cada valor."""
        measured = self.names.result(step.id, field.name, field)
        label = measured + self.qualifier(chain, _shown(fields))
        when = rows[0][date_field.name]
        moment = self.when(date_field, when, temporal=True) or (self.value(when), "period")
        shared = self._shared_attributes(rows, fields, {field.name, date_field.name})
        where = (" " + ", ".join(shared)) if shared else ""

        if len(rows) == 1:
            return self._latest_value(rows[0][field.name], field, label, where, moment), False

        return self._latest_records(rows, field, label, where, moment), True

    def _shared_attributes(self, rows, fields, skip):
        shared = []
        for name, item in fields.items():
            if name in skip or item.sql_type != "TEXT":
                continue
            values = {row[name] for row in rows}
            if len(values) == 1 and None not in values:
                shared.append(self.attribute(item, next(iter(values))))

        return shared

    def _latest_value(self, value, field, label, where, moment):
        when = at(*moment, self.language)
        if value is None:
            if self.es:
                return (
                    f"La medición más reciente de {label}{where}, {when}, "
                    "no tiene un valor registrado."
                )
            return f"The most recent {label} record{where}, {when}, has no value."

        measured = self.measure(field, value)
        if self.es:
            return f"La medición más reciente de {label}{where} fue {measured}, {when}."

        return f"The most recent {label} measurement{where} was {measured}, {when}."

    def _latest_records(self, rows, field, label, where, moment):
        counts = Counter(row[field.name] for row in rows)
        parts = [self._tally(field, value, times) for value, times in counts.items()]
        origin = dated(*moment, self.language)
        if self.es:
            return (
                f"Los registros más recientes de {label}{where} son {origin}: "
                f"{join_words(parts, 'es')}."
            )

        return f"The most recent {label} records{where} are {origin}: {join_words(parts, 'en')}."

    def _tally(self, field, value, times):
        if value is None:
            if self.es:
                return self.count(times, "registro sin dato", "registros sin dato")
            return self.count(times, "record without a value", "records without a value")

        if self.es:
            records = self.count(times, "registro", "registros")
        else:
            records = self.count(times, "record", "records")
        preposition = "con" if self.es else "with"

        return f"{records} {preposition} {self.measure(field, value)}"

    def ranking(self, rows, step, fields, inline_rows, chain):
        order = step.order_by[0]
        name = _ref_name(order.field)
        field = fields.get(name)
        if field is None or field.sql_type not in NUMERIC_TYPES or rows[0][name] is None:
            return None

        highest = order.direction == "desc"
        top = rows[0][name]
        label = self.names.result(step.id, name, field) + self.qualifier(chain, _shown(fields))
        tied = [row for row in rows if row[name] == top]
        contexts = [self.context(row, fields, {name}) for row in tied]
        places = [place for place in contexts if place]
        text = self._extreme(field, label, top, highest, len(tied), places)

        rest = [row for row in rows if row[name] != top]
        if not rest:
            return text, False
        if len(rows) <= inline_rows:
            return text + self._following(rest, field, fields), False

        return text + self._ranked_table(len(rows), highest), True

    def _extreme(self, field, label, top, highest, ties, places):
        measured = self.measure(field, top)
        if self.es:
            text = f"El valor más {'alto' if highest else 'bajo'} de {label} fue {measured}"
            if places:
                text += f" y se registró {join_words(places, 'es')}"
            elif ties > 1:
                text += f" y aparece en {ties} registros"
            return text + "."

        text = f"The {'highest' if highest else 'lowest'} {label} value was {measured}"
        if places:
            text += f", recorded {join_words(places, 'en')}"
        elif ties > 1:
            text += f", in {ties} records"

        return text + "."

    def _following(self, rest, field, fields):
        following = [self._described(row, field, fields) for row in rest]
        if self.es:
            verb = "Le sigue" if len(rest) == 1 else "Le siguen"
            return f" {verb} {join_words(following, 'es')}."

        return f" Next: {join_words(following, 'en')}."

    def _ranked_table(self, total, highest):
        direction = self._direction_text(highest)
        if self.es:
            return (
                f" En la tabla podés ver los {self.count(total, 'registro', 'registros')} "
                f"ordenados {direction}."
            )

        return f" The table lists all {self.count(total, 'record', 'records')} {direction}."

    def _direction_text(self, descending):
        if self.es:
            return "de mayor a menor" if descending else "de menor a mayor"

        return "from highest to lowest" if descending else "from lowest to highest"

    def _described(self, row, field, fields):
        measured = self.measure(field, row[field.name])
        context = self.context(row, fields, {field.name})

        return " ".join(filter(None, [measured, context]))

    def rows(self, rows, fields, calculations, grouping, direction, inline_rows, chain):
        values = [name for name in fields if name not in grouping]
        grouped = self.qualifier(chain, _shown({name: fields[name] for name in grouping}))

        if len(rows) == 1 and len(values) <= INLINE_FIELDS:
            return self._single(rows[0], fields, values, calculations, grouping, grouped), False

        few_rows = 1 < len(rows) <= inline_rows
        numeric = [name for name in values if fields[name].sql_type in NUMERIC_TYPES]
        one_calculation = bool(grouping) and len(values) == 1 and values[0] in calculations
        one_measure = (
            not grouping and not calculations and len(numeric) == 1 and len(values) <= INLINE_FIELDS
        )
        if few_rows and one_calculation:
            name = values[0]
            text = self._grouped(rows, fields[name], calculations[name], grouping, fields, grouped)
            return text, False
        if few_rows and one_measure:
            qualifier = self.qualifier(chain, _shown(fields))
            return self._listed(rows, fields[numeric[0]], fields, qualifier), False

        text = self._lead(rows, fields, values, calculations, grouping, direction, grouped)

        return text, True

    def _single(self, row, fields, values, calculations, grouping, qualifier):
        scope = " ".join(filter(None, [qualifier.strip(), self.group_scope(grouping, fields, row)]))
        if not calculations:
            pairs = [self.pair(fields[name], row[name]) for name in values]
            listed = join_words(pairs, self.language)
            if self.es:
                return f"Encontré un registro con {listed}."
            return f"I found one record with {listed}."

        parts = []
        for name in values:
            calculation = calculations.get(name)
            if calculation is None:
                parts.append(self.pair(fields[name], row[name]))
                continue
            subject = self.subject(calculation) + (f" {scope}" if scope and not parts else "")
            parts.append(self._calculated(subject, fields[name], row[name]))

        return sentence(join_words(parts, self.language)) + "."

    def _calculated(self, subject, field, value):
        if value is None:
            if self.es:
                return f"no se pudo obtener {subject} con los datos disponibles"
            return f"{subject} could not be obtained from the available data"

        measured = self.measure(field, value)

        return f"{subject} fue {measured}" if self.es else f"{subject} was {measured}"

    def _grouped(self, rows, field, calculation, grouping, fields, qualifier):
        missing = "sin valor calculable" if self.es else "no computable value"
        items = []
        for row in rows:
            value = row[field.name]
            shown = self.measure(field, value) if value is not None else missing
            items.append(f"{shown} {self.group_scope(grouping, fields, row)}")
        verb = "fue" if self.es else "was"

        return sentence(
            f"{self.subject(calculation)}{qualifier} {verb} {join_words(items, self.language)}."
        )

    def _listed(self, rows, field, fields, qualifier):
        label = self.label(field) + qualifier
        items = [self._described(row, field, fields) for row in rows]
        total = len(rows)
        if self.es:
            return (
                f"Encontré {self.count(total, 'registro', 'registros')} de {label}: "
                f"{join_words(items, 'es')}."
            )

        return (
            f"I found {self.count(total, 'record', 'records')} of {label}: "
            f"{join_words(items, 'en')}."
        )

    def _lead(self, rows, fields, values, calculations, grouping, direction, qualifier):
        total = len(rows)
        order = self._direction_text(direction == "desc") if direction else ""
        calculated = [self.subject(calculations[name]) for name in values if name in calculations]
        subjects = join_words(calculated, self.language)
        table = self._table_hint(order)

        if calculations and grouping:
            return self._grouped_lead(subjects, grouping, qualifier, total, table)
        if calculations and total == 1:
            return self._calculated_values(len(values))
        if calculations:
            return self._calculated_records(subjects, qualifier, total, table)
        if total == 1:
            return self._one_record(len(values))

        return self._found_records(fields, values, qualifier, total, table)

    def _table_hint(self, order):
        if self.es:
            if order:
                return f" y en la tabla aparecen ordenados {order}."
            return " y podés verlos completos en la tabla."

        if order:
            return f", listed in the table {order}."
        return "; you can see all of them in the table."

    def _grouped_lead(self, subjects, grouping, qualifier, total, table):
        groups = join_words([self._group_name(spec) for spec in grouping.values()], self.language)
        if self.es:
            results = self.count(total, "resultado", "resultados")
            return f"Calculé {subjects} por {groups}{qualifier}. Son {results}{table}"

        results = self.count(total, "result", "results")

        return f"I calculated {subjects} by {groups}{qualifier}. There are {results}{table}"

    def _group_name(self, spec):
        if spec.time_grain != "raw":
            return grain_label(spec.time_grain, self.language)

        return self.names.operand(spec.field)

    def _calculated_values(self, variables):
        if self.es:
            calculated = self.count(variables, "valor", "valores")
            return f"Calculé {calculated}; podés verlos en la tabla."

        calculated = self.count(variables, "value", "values")

        return f"I calculated {calculated}; you can see them in the table."

    def _calculated_records(self, subjects, qualifier, total, table):
        if self.es:
            records = self.count(total, "registro", "registros")
            return f"Calculé {subjects} para cada uno de los {records}{qualifier}{table}"

        records = self.count(total, "record", "records")

        return f"I calculated {subjects} for each of the {records}{qualifier}{table}"

    def _one_record(self, variables):
        if self.es:
            measured = self.count(variables, "variable", "variables")
            return f"Encontré un registro con {measured}; podés verlas en la tabla."

        measured = self.count(variables, "variable", "variables")

        return f"I found one record with {measured}; you can see them in the table."

    def _found_records(self, fields, values, qualifier, total, table):
        numeric = [fields[name] for name in values if fields[name].sql_type in NUMERIC_TYPES]
        measured = ""
        if len(numeric) == 1:
            label = self.label(numeric[0])
            measured = f" de {label}" if self.es else f" of {label}"

        if self.es:
            records = self.count(total, "registro", "registros")
            return f"Encontré {records}{measured}{qualifier}{table}"

        records = self.count(total, "record", "records")

        return f"I found {records}{measured}{qualifier}{table}"

    def projections(self, record):
        """Columnas de texto leídas en su versión numérica; el recuento de valores no numéricos es del dataset."""
        texts = []
        for field in record.columns:
            for ref in field.sources:
                dataset = self._sources[ref.dataset_id]
                column = next(
                    item for item in dataset.columns_metadata if item.sql_name == ref.column
                )
                if column.derived_from and column.excluded_count:
                    texts.append(self._projection(dataset, column))

        return _unique(texts)

    def _projection(self, dataset, column):
        original = _column_metadata(dataset, column.derived_from)
        if original:
            label = column_label(original.original_name, original.unit)
        else:
            label = humanize(column.derived_from)
        excluded = column.excluded_count

        if self.es:
            unreadable = self.count(
                excluded,
                "un valor que no se pudo leer como número queda",
                "valores que no se pudieron leer como número quedan",
            )
            return (
                f"La columna {label} se usa en su versión numérica: en la tabla de origen, "
                f"{unreadable} sin dato."
            )

        unreadable = self.count(
            excluded,
            "value that could not be read as a number stays",
            "values that could not be read as numbers stay",
        )

        return f"{label} is used in its numeric version: in the source table, {unreadable} empty."

    def exclusions(self, record, origin, fields):
        calculations = {item.output_name: item for item in origin.calculations}
        totals = _exclusion_totals(record)
        texts = []
        for (valid, nulls, zeros, policy), names in _names_by_total(totals).items():
            text = self._exclusion(valid, nulls, zeros, policy, origin)
            subjects = [
                self.subject(calculations[name])
                for name in names
                if name in calculations and name in fields
            ]
            if len(calculations) > 1 and subjects:
                preposition = "Para" if self.es else "For"
                text = (
                    f"{preposition} {join_words(subjects, self.language)}, "
                    f"{_lowercase_first(text)}"
                )
            texts.append(text)

        return texts

    def _exclusion(self, valid, nulls, zeros, policy, origin):
        parts = []
        if nulls:
            parts.append(self._missing_values(valid, nulls, policy, origin))
        if zeros:
            parts.append(self._zero_divisions(zeros))

        return sentence(join_words(parts, self.language)) + "."

    def _missing_values(self, valid, nulls, policy, origin):
        if policy == "exclude":
            return self._excluded_missing(valid, nulls)
        if origin.operation == "aggregate":
            return self._uncalculated_missing(nulls, origin)

        return self._kept_missing(nulls)

    def _excluded_missing(self, valid, nulls):
        if self.es:
            used = self.count(valid, "registro", "registros")
            left_out = self.count(nulls, "registro sin dato", "registros sin dato")
            return f"el cálculo usó {used} con dato y dejó fuera {left_out}"

        used = self.count(valid, "record", "records")
        left_out = self.count(nulls, "record", "records")

        return f"the calculation used {used} with a value and left out {left_out} without one"

    def _uncalculated_missing(self, nulls, origin):
        # Conservar un faltante en una estadística la deja sin valor: no se calcula sobre datos incompletos.
        if self.es:
            kept = self.count(
                nulls, "registro sin dato no se descartó", "registros sin dato no se descartaron"
            )
            groups = "de los grupos que los incluyen " if origin.group_by else ""
            return f"{kept}, así que el resultado {groups}quedó sin calcular"

        kept = self.count(nulls, "record without a value was", "records without a value were")
        groups = "of the groups that include them " if origin.group_by else ""

        return f"{kept} not discarded, so the result {groups}was not calculated"

    def _kept_missing(self, nulls):
        if self.es:
            kept = self.count(
                nulls, "registro sin dato se conservó", "registros sin dato se conservaron"
            )
            return f"{kept} sin valor, en lugar de descartarse"

        kept = self.count(nulls, "record without a value was", "records without a value were")

        return f"{kept} kept empty instead of being discarded"

    def _zero_divisions(self, zeros):
        if self.es:
            cases = self.count(zeros, "caso", "casos")
            return (
                f"en {cases} el divisor fue cero y el cociente quedó sin valor "
                "(no se reemplazó por cero)"
            )

        cases = self.count(zeros, "case", "cases")

        return (
            f"in {cases} the divisor was zero, so the ratio has no value "
            "(it was not replaced with zero)"
        )


def _unique(items):
    return list(dict.fromkeys(items))


def _lowercase_first(text):
    return f"{text[:1].lower()}{text[1:]}"


def _source_key(ref):
    return ref.dataset_id, ref.column


def _single_source(field):
    if len(field.sources) != 1:
        return None

    return _source_key(field.sources[0])


def _ref_name(ref):
    return ref.field if ref.kind == "result" else ref.column


def _column_metadata(dataset, sql_name):
    return next((item for item in dataset.columns_metadata if item.sql_name == sql_name), None)


def _named_field(step, name):
    return next((item for item in step.output_fields if item.name == name), None)


def _source_field(step, ref):
    target = _source_key(ref)
    for field in step.output_fields:
        if field.sources and _source_key(field.sources[0]) == target:
            return field

    return None


def _goes_without_saying(predicate, shown):
    ref = predicate.field
    # Los filtros sobre resultados ya calculados los anuncia result_filters.
    if ref.kind != "column" or predicate.operator == "is_not_null":
        return True

    # Una igualdad sobre una columna visible ya se lee en sus valores; un rango nunca.
    return _source_key(ref) in shown and predicate.operator in _EQUALITY_OPERATORS


def _lists_places(predicate, label):
    return (
        predicate.operator in _EQUALITY_OPERATORS
        and is_place(label)
        and all(isinstance(value, str) for value in predicate.values)
    )


def _iso_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _shown(fields):
    return frozenset(_source_key(ref) for field in fields.values() for ref in field.sources)


def _has_value(rows, name):
    return any(row.get(name) is not None for row in rows)


def _lineage(step, steps):
    chain = []
    visit = [step]
    while visit:
        ancestor = visit.pop()
        chain.append(ancestor)
        visit.extend(steps[key] for key in ancestor.depends_on)

    return chain


def _trace_origin(step, steps):
    origin = step
    latest = None
    ranking = None
    while origin.operation in _ORDERING_OPERATIONS:
        if origin.operation == "latest":
            latest = origin
        elif ranking is None and origin.order_by:
            ranking = origin
        origin = steps[origin.depends_on[0]]

    return origin, latest, ranking


def _sort_direction(ranking, fields):
    if not ranking:
        return None

    order = ranking.order_by[0]
    sorted_field = fields.get(_ref_name(order.field))
    if sorted_field and sorted_field.sql_type in NUMERIC_TYPES:
        return order.direction

    return None


def _file_count(evidence, sources):
    names = set()
    for item in evidence:
        for key in item.dataset_ids:
            names.add(sources[key].provenance.source_name)

    return len(names)


def _exclusion_totals(record):
    totals = {}
    for item in record.exclusions:
        valid, nulls, zeros, _ = totals.get(item.output_name, (0, 0, 0, item.policy))
        totals[item.output_name] = (
            valid + item.valid_rows,
            nulls + item.null_rows,
            zeros + item.zero_divisions,
            item.policy,
        )

    return totals


def _names_by_total(totals):
    names = {}
    for name, total in totals.items():
        _, nulls, zeros, _ = total
        if nulls or zeros:
            names.setdefault(total, []).append(name)

    return names
