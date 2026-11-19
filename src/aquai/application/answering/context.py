import json
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from itertools import count as counter

from aquai.application.answering.naming import FieldNames
from aquai.application.answering.patterns import extremes, modal_bin, pearson, plotted_points
from aquai.application.answering.presentation import format_number
from aquai.application.answering.wording import (
    INLINE_FIELDS,
    NUMERIC_TYPES,
    at,
    calculation_meaning,
    column_label,
    count_text,
    dataset_label,
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
    spoken_label,
    unit_suffix,
    with_article,
)
from aquai.application.contracts.answering import ResponseContextRecord
from aquai.application.execution import decode_rows, execution_records
from aquai.application.planning.followUp import period_fields
from aquai.application.planning.temporalText import period_bounds
from aquai.application.matching import singular_words
from aquai.application.Rules.language import ENGLISH_CUES, SPANISH_CUES, WORD
from aquai.application.Rules.vocabulary import TREND_CUES, YEAR_COLUMN_NAMES

_EQUALITY_OPERATORS = {"eq", "ieq", "in"}
_RANGE_OPERATORS = {"lt", "le", "gt", "ge"}
_GROUPED_DIGITS = 5
_DATE_TYPES = {"DATE", "TIMESTAMP"}
_ORDERING_OPERATIONS = {"sort", "latest"}
_ANSWERING_CHARTS = {"histogram", "scatter"}
_MAX_PATTERN_MEASURES = 3
_MAX_LOCATOR_PARTS = 2
_CORRELATION_BANDS = (
    (0.1, "prácticamente nula", "virtually absent"),
    (0.3, "débil", "weak"),
    (0.5, "moderada", "moderate"),
    (1.01, "fuerte", "strong"),
)
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


#
# Devuelve el idioma de la respuesta: el preferido o el que se deduce de la pregunta.
#
def response_language(question, preferred=None):
    if preferred is not None:
        return preferred

    words = set(WORD.findall(question.lower()))
    english = len(words & ENGLISH_CUES)
    spanish = len(words & SPANISH_CUES)

    return "en" if english > spanish else "es"


#
# Construye el contexto de evidencia y oraciones con que se redacta la respuesta.
#
def build_response_context(state, budget_bytes, max_preview_rows=3, max_sentences=24):
    return _ContextBuilder(state, budget_bytes, max_preview_rows, max_sentences).build()


class _ContextBuilder:
    #
    # Guarda el estado, el presupuesto y los datos del plan que usa el constructor.
    #
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
        periods = [
            tuple(_iso_date(item[side]) if item.get(side) else None for side in ("start", "end"))
            for item in plan.context.evidence.get("requested_periods", [])
        ]
        self._writer = _Writer(self._sources, self._steps, self._language, temporal, periods)
        self._records = execution_records(state)
        self._period = state.routing.requested_period if state.routing else None

        self._catalog = []
        self._body = self._initial_body()
        self._sentence_ids = counter(1)
        self._summaries = {}
        self._omitted = {}
        self._meanings = {}
        self._patterned = set()

    #
    # Arma el contexto completo de la respuesta y lo devuelve junto con el idioma.
    #
    def build(self):
        self._add_references()
        if self._period:
            self._add_period_sentences()
            return self._finish()

        if self._multi_source:
            self._body["introduction"] = self._writer.introduction(self._files)
        self._add_table_sentences()
        self._add_source_comparison()
        self._add_meanings()

        return self._finish()

    #
    # Responde, cuando la pregunta busca una tendencia en el tiempo y los resultados vienen de archivos distintos,
    # que la diferencia entre ellos no la muestra por sí sola.
    #
    def _add_source_comparison(self):
        routing = self._state.routing
        question = routing.original_question if routing else self._state.question
        tables = [
            item for item in self._validation.usable_evidence if item.kind == "table" and item.is_requested_output
        ]
        if not self._multi_source or len(tables) < 2 or TREND_CUES.search(question) is None:
            return

        entry = self._add(tables[-1], self._writer.source_comparison(), required=True, attribute=False)
        if entry is not None:
            entry["evidence_ids"] = [item.evidence_id for item in tables]

    #
    # Agrega las oraciones que responden con el período de cada resultado.
    #
    def _add_period_sentences(self):
        periods = self._state.planning.temporal_text_columns
        for evidence in self._validation.usable_evidence:
            if evidence.kind != "table":
                continue

            summary = self._summaries.get(evidence.evidence_id)
            if not evidence.is_requested_output:
                self._catalog.remove(summary)
                continue

            step = self._steps[evidence.operation_id]
            record = self._records[evidence.operation_id]
            fields = {field.name: field for field in record.columns}
            origin, _latest, ranking = _trace_origin(step, self._steps)
            grains = {item.output_name: item.time_grain for item in origin.group_by}
            dated = [
                (fields[name], grains.get(name, "raw"))
                for name in period_fields(step, self._steps, periods)
            ]
            rows = _top_ranked(decode_rows(record, record.columns), ranking, fields)
            span = _date_span(origin)
            if span is not None:
                text = self._writer.period_span(rows, *span, self._period)
            else:
                text = self._writer.period_answer(rows, dated, self._period)
            self._drop_summary(evidence, self._add(evidence, text, attach=True, required=True))

    #
    # Crea el cuerpo inicial del contexto con pregunta, fuentes, avisos y filtros.
    #
    def _initial_body(self):
        state = self._state
        validation = self._validation
        plan = state.planning

        return {
            "question": state.question,
            "language": self._language,
            "status": validation.status,
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

    #
    # Devuelve el cuerpo del contexto serializado como JSON en bytes.
    #
    def _serialized(self):
        return json.dumps(self._body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    #
    # Indica si el contexto serializado supera el presupuesto de bytes.
    #
    def _over_budget(self):
        return len(self._serialized()) > self._budget_bytes

    #
    # Agrega una oración al catálogo si respeta el tope y el presupuesto, salvo que sea obligatoria.
    #
    def _add(self, evidence, text, *, kind="result", attach=False, required=False, attribute=True):
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

    #
    # Agrega una referencia resumida por cada evidencia autorizada.
    #
    def _add_references(self):
        for evidence in self._validation.usable_evidence:
            if evidence.kind == "visualization":
                self._add_chart(evidence)
                continue

            summary = self._writer.summary(evidence.rows)
            self._summaries[evidence.evidence_id] = self._add(
                evidence, summary, kind="summary", attach=True, required=True
            )

    #
    # Agrega la oración que describe un gráfico, con su límite y filas excluidas.
    #
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
        self._add_chart_pattern(evidence, chart)

    #
    # Agrega lo que muestran los datos del gráfico cuando la oración de su tabla no los enumera: extremos,
    # primer y último punto, intervalo más frecuente o correlación, con la definición que haga falta.
    #
    def _add_chart_pattern(self, evidence, chart):
        origin, _latest, _ranking = _trace_origin(self._steps[chart.input_operation_ids[0]], self._steps)
        grouping = {item.output_name: item for item in origin.group_by}
        calculation = next(
            (item for item in origin.calculations if item.output_name == chart.y_axis.field.name), None
        )
        pattern = self._writer.chart_pattern(chart, grouping, calculation, self._max_preview_rows)
        if pattern is None:
            return

        answers = chart.chart_type in _ANSWERING_CHARTS
        self._add(evidence, pattern, kind="result" if answers else "pattern", required=True)
        self._patterned.update(chart.input_operation_ids)
        if chart.chart_type == "scatter":
            self._add(evidence, self._writer.correlation_meaning(), kind="meaning")

    #
    # Agrega las oraciones de las tablas, primero los resultados derivados y luego las lecturas.
    #
    def _add_table_sentences(self):
        ordered = sorted(
            self._validation.usable_evidence,
            key=lambda item: self._steps[item.operation_id].operation == "read",
        )
        for evidence in ordered:
            if evidence.kind == "table":
                self._add_table(evidence)

    #
    # Agrega las oraciones de una tabla: alcance, resultado, definiciones y detalles.
    #
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

        pattern = self._table_pattern(evidence, rows, fields, origin, latest, ranking)
        if pattern is not None:
            self._add(evidence, pattern, kind="pattern")
        strength = self._correlation_fact(origin, rows)
        if strength is not None:
            self._add(evidence, strength, required=True)
        self._collect_meanings(origin, rows, evidence)
        self._add_details(evidence, record, origin, fields)

    #
    # Devuelve qué tan fuerte es una correlación calculada y con cuántos pares, que responden si hay una asociación:
    # los pares se cuentan sólo si la lectura excluyó los nulos de ambas variables.
    #
    def _correlation_fact(self, origin, rows):
        correlations = [item for item in origin.calculations if item.function == "correlation"]
        if origin.operation != "calculate" or len(correlations) != 1 or len(rows) != 1:
            return None
        value = rows[0].get(correlations[0].output_name)
        if value is None:
            return None

        source = self._steps.get(origin.depends_on[0]) if origin.depends_on else None
        record = self._records.get(source.id) if source is not None else None
        excluded = {
            getattr(item.field, "column", None)
            for item in (source.filters if source else [])
            if item.operator == "is_not_null"
        }
        complete = record is not None and {ref.field for ref in correlations[0].inputs} <= excluded
        return self._writer.correlation_summary(float(value), record.rows_returned if complete else None)

    #
    # Devuelve lo que destaca un resultado extenso que la tabla muestra completo: el rango de cada medición con
    # dónde y cuándo se registraron sus extremos, o el grupo con el valor más alto y el más bajo. Así la
    # respuesta puede sintetizar sin enumerar; None si la oración ya lo dice o un gráfico ya lo describe.
    #
    def _table_pattern(self, evidence, rows, fields, origin, latest, ranking):
        if len(rows) <= self._max_preview_rows or latest is not None or evidence.operation_id in self._patterned:
            return None

        calculations = {item.output_name: item for item in origin.calculations}
        grouping = {item.output_name: item for item in origin.group_by}
        if grouping:
            return self._writer.grouped_pattern(rows, fields, calculations, grouping)
        if calculations or self._writer.ranks_values(ranking, fields):
            return None

        return self._writer.records_pattern(rows, fields)

    #
    # Elige y redacta la oración que describe el resultado de una tabla.
    #
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

        if latest and len(numeric) == 1:
            date_field = fields[latest.order_by[0].field.field]
            return writer.latest(
                rows, numeric[0], date_field, fields, latest, chain, self._max_preview_rows
            )

        if ranking and not calculations and not grouping:
            ranked = writer.ranking(rows, ranking, fields, self._max_preview_rows, chain)
            if ranked is not None:
                return ranked

        direction = _sort_direction(ranking, fields) if writer.ranks_values(ranking, fields) else None

        return writer.rows(
            rows, fields, calculations, grouping, direction, self._max_preview_rows, chain
        )

    #
    # Quita el resumen genérico de una evidencia cuando ya tiene una oración propia.
    #
    def _drop_summary(self, evidence, entry):
        summary = self._summaries.get(evidence.evidence_id)
        if entry is not None and summary in self._catalog:
            self._catalog.remove(summary)

    #
    # Registra el significado de las estadísticas que produjeron algún valor.
    #
    def _collect_meanings(self, origin, rows, evidence):
        for calculation in origin.calculations:
            meaning = self._writer.meaning(calculation)
            if meaning and _has_value(rows, calculation.output_name):
                self._meanings.setdefault(meaning, evidence)

    #
    # Agrega los detalles de filas excluidas y, una sola vez, los de columnas numéricas derivadas.
    #
    def _add_details(self, evidence, record, origin, fields):
        written = {entry["direct"] for entry in self._catalog}
        projections = [item for item in self._writer.projections(record) if item not in written]
        exclusions = self._writer.exclusions(record, origin, fields)
        for detail in [*projections, *exclusions]:
            self._add(evidence, detail, kind="detail")

    #
    # Agrega al final las definiciones de las estadísticas usadas, si queda lugar.
    #
    def _add_meanings(self):
        for meaning, evidence in self._meanings.items():
            self._add(evidence, meaning, kind="meaning")

    #
    # Cierra el contexto: recorta el historial al presupuesto y calcula su huella.
    #
    def _finish(self):
        self._body["omitted_rows"] = self._omitted
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
    #
    # Guarda las fuentes, los pasos y el idioma con que se redactan las oraciones.
    #
    def __init__(self, sources, steps, language, temporal, periods=()):
        self._sources = sources
        self._steps = steps
        self._temporal = temporal
        self._periods = periods
        self.language = language
        self.es = language == "es"
        self.names = FieldNames(steps, self.column, language)

    #
    # Devuelve la etiqueta legible de un campo.
    #
    def label(self, field):
        source = _single_source(field)
        label = self.column(*source) if source else None

        return label or humanize(field.name)

    #
    # Devuelve la etiqueta de una columna de un dataset, o None si no existe.
    #
    def column(self, dataset_id, sql_name):
        dataset = self._sources.get(dataset_id)
        column = _column_metadata(dataset, sql_name) if dataset else None
        if not column:
            return None

        named = _column_metadata(dataset, column.derived_from) or column
        label = spoken_label(column_label(named.original_name, named.unit))

        return label if len(label) <= _MAX_LABEL_LENGTH else column.sql_name

    #
    # Devuelve un valor formateado para el texto, con opción de conservarlo exacto.
    #
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
            return _exact_number(value, self.language)

        return format_number(value, self.language)

    #
    # Devuelve «sí» o «no» (o «yes»/«no») para un valor booleano.
    #
    def _yes_no(self, value):
        if self.es:
            return "sí" if value else "no"

        return "yes" if value else "no"

    #
    # Devuelve un valor con la unidad documentada de su campo.
    #
    def measure(self, field, value):
        shown = self.value(value)
        if value is None:
            return shown

        return shown + unit_suffix(field, self.language)

    #
    # Devuelve una cantidad con su sustantivo en singular o plural.
    #
    def count(self, total, singular, plural):
        return count_text(total, singular, plural, self.language)

    #
    # Devuelve el nombre de un cálculo precedido por su artículo.
    #
    def subject(self, calculation):
        return with_article(*self.names.calculation(calculation), self.language)

    #
    # Devuelve la fecha o el período de un valor con su precisión, o None si no es temporal.
    #
    def when(self, field, value, *, temporal=False):
        if isinstance(value, date):
            return date_text(value, self.language), "day"

        key = _single_source(field)
        is_period = isinstance(value, str) and (temporal or key in self._temporal)
        if not is_period:
            return None

        return period(value, self.language, self._temporal.get(key))

    #
    # Devuelve el complemento que ubica un valor: lugar, fecha u otra categoría.
    #
    def attribute(self, field, value):
        label = self.label(field)

        moment = self.when(field, value)
        if moment is not None:
            return at(*moment, self.language)
        if is_place(label) and isinstance(value, str):
            return self._located(place_text(value))

        shown = self._displayed(field, value)

        return f"con {label} {shown}" if self.es else f"with {label} {shown}"

    #
    # Devuelve el complemento «en» seguido de los lugares indicados.
    #
    def _located(self, places):
        return ("en " if self.es else "in ") + places

    #
    # Devuelve un valor tal como se muestra: texto tal cual o medida formateada.
    #
    def _displayed(self, field, value):
        return value.strip() if isinstance(value, str) else self.measure(field, value)

    #
    # Devuelve los atributos de una fila que la ubican, separados por comas.
    #
    def context(self, row, fields, skip):
        attributes = [
            self.attribute(field, row[name])
            for name, field in fields.items()
            if name not in skip and row[name] is not None
        ]

        return ", ".join(attributes)

    #
    # Devuelve el complemento que describe el grupo de una fila según su agrupación.
    #
    def scope(self, spec, field, value):
        if value is None:
            label = self.label(field)
            return f"sin dato de {label}" if self.es else f"without a {label} value"
        if spec.time_grain != "raw":
            return at(*grain(value, spec.time_grain, self.language), self.language)

        return self.attribute(field, value)

    #
    # Devuelve el complemento que describe todas las agrupaciones de una fila.
    #
    def group_scope(self, grouping, fields, row):
        scopes = [self.scope(spec, fields[name], row[name]) for name, spec in grouping.items()]

        return join_words(scopes, self.language)

    #
    # Devuelve el par «etiqueta valor» de un campo, o su ausencia de dato.
    #
    def pair(self, field, value):
        label = self.label(field)
        if value is None:
            return f"{label} sin dato" if self.es else f"{label} without a value"

        moment = self.when(field, value)
        if moment is not None:
            return f"{label} {moment[0]}"

        return f"{label} {self._displayed(field, value)}"

    #
    # Antepone el nombre del archivo a una oración cuando hay varias fuentes.
    #
    def attributed(self, evidence, text):
        preposition = "En" if self.es else "In"

        return f"{preposition} {self.files(evidence)}, {_lowercase_first(text)}"

    #
    # Devuelve el nombre de los archivos de una evidencia con su artículo.
    #
    def files(self, evidence):
        names = _unique(self._file_name(key) for key in evidence.dataset_ids)
        single = len(names) == 1
        if self.es:
            return ("el archivo " if single else "los archivos ") + join_words(names, "es")

        return ("the file " if single else "the files ") + join_words(names, "en")

    #
    # Devuelve el nombre del archivo, con la hoja si otra tabla del mismo archivo participa.
    #
    def _file_name(self, dataset_id):
        dataset = self._sources[dataset_id]
        source = dataset.provenance.source_name
        siblings = [item for item in self._sources.values() if item.provenance.source_name == source]
        if len(siblings) == 1:
            return file_stem(source)

        sheet = "hoja" if self.es else "sheet"
        return f"{file_stem(source)} ({sheet} {dataset_label(dataset.name)})"

    #
    # Devuelve la introducción que avisa que hay resultados en varios archivos.
    #
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

    #
    # Devuelve por qué la diferencia entre resultados de archivos distintos no muestra por sí sola una tendencia.
    #
    def source_comparison(self):
        if self.es:
            return (
                "Como los resultados vienen de archivos distintos, la diferencia entre ellos no muestra por sí sola "
                "una tendencia en el tiempo: también puede deberse a diferencias de período, de puntos de muestreo "
                "o de método de medición."
            )

        return (
            "Since the results come from different files, the difference between them does not by itself show a "
            "trend over time: it can also come from differences in period, sampling points or measurement method."
        )

    #
    # Devuelve la definición de la estadística usada en un cálculo.
    #
    def meaning(self, calculation):
        return calculation_meaning(
            calculation.function, self.language, quantile=calculation.quantile
        )

    #
    # Devuelve el resumen genérico que remite a la tabla de resultados.
    #
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

    #
    # Devuelve el aviso de que no hay registros que cumplan los criterios.
    #
    def empty(self, evidence):
        if self.es:
            return f"No encontré registros que cumplan esos criterios en {self.files(evidence)}."

        return f"No records match those criteria in {self.files(evidence)}."

    #
    # Describe un gráfico, con la nota de límite y de filas excluidas si corresponde.
    #
    def chart(self, chart, limit):
        rows = chart.input_rows
        excluded = chart.excluded_rows
        text = self._chart_description(chart.chart_type, rows, excluded)

        if limit and rows >= limit:
            text += self._limit_note(limit)
        if excluded:
            text += self._excluded_note(excluded)

        return text

    #
    # Devuelve la descripción de un gráfico según su tipo.
    #
    def _chart_description(self, chart_type, rows, excluded):
        if self.es:
            shown = (
                "el resultado"
                if rows == 1
                else f"los {self.count(rows, 'resultado', 'resultados')}"
            )
            values = self.count(rows - excluded, "valor", "valores")
            distribution = (
                "El histograma muestra un único valor, así que no hay una distribución para comparar."
                if rows - excluded == 1
                else f"El histograma muestra cómo se distribuyen {values}: "
                "cada barra indica cuántos valores caen dentro de un intervalo."
            )
            descriptions = {
                "histogram": distribution,
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
            distribution = (
                "The histogram shows a single value, so there is no distribution to compare."
                if rows - excluded == 1
                else f"The histogram shows how {values} distributed: "
                "each bar counts the values that fall within an interval."
            )
            descriptions = {
                "histogram": distribution,
                "bar": f"The bar chart compares {shown}: the height of each bar is its value.",
                "line": f"The line chart shows how {shown} change along the horizontal axis.",
                "scatter": f"The scatter plot shows {shown} as points, "
                "to see whether the two variables are related.",
            }

        return descriptions[chart_type]

    #
    # Describe lo que muestran los puntos de un gráfico, o None si no agrega nada a lo ya dicho: la oración
    # de la tabla enumera hasta inline_rows valores y un único valor no tiene patrón.
    #
    def chart_pattern(self, chart, grouping, calculation, inline_rows):
        if chart.chart_type == "histogram":
            return self._histogram_pattern(chart)

        points = plotted_points(chart)
        if chart.chart_type == "scatter":
            return self._correlation(chart, points)
        if len(points) <= inline_rows:
            return None

        return self._extremes(chart, points, grouping, calculation)

    #
    # Redacta el valor más alto y el más bajo del gráfico y, en una línea de una sola serie, el primer y el
    # último punto.
    #
    def _extremes(self, chart, points, grouping, calculation):
        found = extremes(points)
        if found is None:
            return None

        measured = self.subject(calculation) if calculation else self.label(chart.y_axis.field)
        highest, lowest = (self._plotted_value(chart, item, grouping) for item in found)
        if self.es:
            text = f"En el gráfico, el valor máximo {_of(measured)} fue {highest} y el mínimo fue {lowest}."
        else:
            text = f"In the chart, the maximum value of {measured} was {highest} and the minimum was {lowest}."

        if chart.chart_type != "line" or len(chart.series) != 1:
            return text

        first, last = (self._plotted_value(chart, item, grouping) for item in (points[0], points[-1]))
        if self.es:
            return f"{text} Del primer al último punto pasó de {first} a {last}."

        return f"{text} From the first to the last point it went from {first} to {last}."

    #
    # Devuelve el valor de un punto del gráfico con su unidad y su ubicación.
    #
    def _plotted_value(self, chart, point, grouping):
        measured = self.measure(chart.y_axis.field, point.y)

        return f"{measured} {self._plotted_at(chart, point, grouping)}"

    #
    # Devuelve dónde está un punto del gráfico: su serie y su posición en el eje horizontal.
    #
    def _plotted_at(self, chart, point, grouping):
        field = chart.x_axis.field
        value = _typed(field, point.x)
        spec = grouping.get(field.name)
        if self._period_kind(field) == "year":
            place = at(str(value), "year", self.language)
        elif spec is not None:
            place = self.scope(spec, field, value)
        else:
            place = self.attribute(field, value)
        if chart.series_field is None or point.series is None:
            return place

        series = self.attribute(chart.series_field, _typed(chart.series_field, point.series))

        return f"{series}, {place}"

    #
    # Redacta el rango de un histograma y el intervalo con más valores.
    #
    def _histogram_pattern(self, chart):
        bins = [item for series in chart.series for item in series.bins]
        total = sum(item.count for item in bins)
        if total < 2 or bins[0].lower == bins[-1].upper:
            return None

        field = chart.x_axis.field
        label = self.label(field)
        low, high = self.measure(field, bins[0].lower), self.measure(field, bins[-1].upper)
        values = self.count(total, "valor", "valores") if self.es else self.count(total, "value", "values")
        top = modal_bin(chart)
        if self.es:
            text = f"Los {values} de {label} van de {low} a {high}."
            if top is None:
                return text
            bounds = f"de {self.measure(field, top.lower)} a {self.measure(field, top.upper)}"
            return f"{text} El intervalo más frecuente, {bounds}, reúne {self.count(top.count, 'valor', 'valores')}."

        text = f"The {values} of {label} range from {low} to {high}."
        if top is None:
            return text
        bounds = f"from {self.measure(field, top.lower)} to {self.measure(field, top.upper)}"

        return f"{text} The most frequent interval, {bounds}, holds {self.count(top.count, 'value', 'values')}."

    #
    # Redacta el coeficiente de correlación de Pearson entre los ejes de una dispersión.
    #
    def _correlation(self, chart, points):
        found = pearson(points)
        if found is None:
            return None

        coefficient, total = found
        x, y = (self.label(axis.field) for axis in (chart.x_axis, chart.y_axis))
        shown = f"{coefficient:.2f}".replace(".", "," if self.es else ".")
        strength = self._correlation_strength(coefficient)
        if self.es:
            return (
                f"El coeficiente de correlación de Pearson entre {x} y {y} fue {shown}, "
                f"calculado con {self.count(total, 'punto', 'puntos')}: {strength}."
            )

        return (
            f"The Pearson correlation coefficient between {x} and {y} was {shown}, "
            f"computed from {self.count(total, 'point', 'points')}: {strength}."
        )

    #
    # Redacta qué tan fuerte es una correlación calculada y, si se conocen, con cuántos pares de valores.
    #
    def correlation_summary(self, coefficient, pairs):
        strength = self._correlation_strength(coefficient)
        if self.es:
            counted = f"Se calculó con {self.count(pairs, 'par de valores', 'pares de valores')}: " if pairs else "Indica "
            return sentence(f"{counted}{strength}") + "."

        counted = f"It was computed from {self.count(pairs, 'pair of values', 'pairs of values')}: " if pairs else "It shows "
        return sentence(f"{counted}{strength}") + "."

    #
    # Describe la fuerza de una correlación según su valor absoluto (convención de Cohen: menos de 0,1
    # prácticamente nula, hasta 0,3 débil, hasta 0,5 moderada y desde 0,5 fuerte) y, si no es nula, su sentido.
    #
    def _correlation_strength(self, coefficient):
        band = next(item for item in _CORRELATION_BANDS if abs(coefficient) < item[0])
        absent = band is _CORRELATION_BANDS[0]
        if self.es:
            direction = "" if absent else f" y {'positiva' if coefficient > 0 else 'negativa'}"
            return f"una relación lineal {band[1]}{direction}"
        if absent:
            return "virtually no linear relationship"

        return f"a {band[2]} {'positive' if coefficient > 0 else 'negative'} linear relationship"

    #
    # Devuelve qué indica el coeficiente de correlación de Pearson.
    #
    def correlation_meaning(self):
        if self.es:
            return (
                "La correlación de Pearson va de -1 a 1: cerca de 1, las dos variables aumentan juntas; "
                "cerca de -1, una disminuye cuando la otra aumenta; y cerca de 0, no hay una relación lineal "
                "entre ellas. No indica que una variable cause la otra."
            )

        return (
            "Pearson correlation ranges from -1 to 1: near 1, both variables increase together; near -1, one "
            "decreases as the other increases; near 0, there is no linear relationship between them. It does "
            "not mean that one variable causes the other."
        )

    #
    # Devuelve la nota de que la consulta se limitó a cierta cantidad de resultados.
    #
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

    #
    # Devuelve la nota de registros sin datos válidos que quedaron fuera del gráfico.
    #
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

    #
    # Devuelve los filtros que acotan el resultado y que la oración no muestra aún.
    #
    def qualifier(self, chain, shown=frozenset()):
        parts = []
        periods = self._date_periods(chain)
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                if _goes_without_saying(predicate, shown):
                    continue
                period = periods.get(_source_key(predicate.field))
                ref = predicate.field
                field = _source_field(ancestor, ref) if ref.kind == "column" else None
                if period is not None and field and self._period_bounds(predicate, field):
                    parts.append(period)
                else:
                    parts.append(self._restriction(predicate, ancestor))

        parts = _unique(parts)

        return (" " + ", ".join(parts)) if parts else ""

    #
    # Agrupa las cotas de cada fecha filtrada y las expresa como período.
    #
    def _date_periods(self, chain):
        bounds = {}
        for ancestor in chain:
            for predicate in ancestor.filters:
                ref = predicate.field
                field = _source_field(ancestor, ref) if ref.kind == "column" else None
                interval = self._period_bounds(predicate, field) if field and predicate.values else None
                if interval is None:
                    continue

                found = bounds.setdefault(_source_key(ref), [None, None])
                found[0] = interval[0] if interval[0] is not None else found[0]
                found[1] = interval[1] if interval[1] is not None else found[1]

        return {key: self._date_period(low, high) for key, (low, high) in bounds.items()}

    #
    # Devuelve el primer y último día que admite un filtro de fecha, año o fecha en texto.
    #
    def _period_bounds(self, predicate, field):
        kind = self._period_kind(field)
        operator = predicate.operator
        if kind is None or (kind == "date" and operator not in _RANGE_OPERATORS):
            return None
        if kind == "text":
            order = self._temporal.get(_single_source(field))
            spans = [period_bounds(value, order) for value in predicate.values]
            if operator not in {"eq", "in"} or None in spans:
                return None
            return self._requested_span(min(span[0] for span in spans), max(span[1] for span in spans))

        days = [self._period_days(value, kind) for value in predicate.values]
        if None in days or (operator == "in" and not _contiguous_days(days)):
            return None
        first, last, day = min(item[0] for item in days), max(item[1] for item in days), timedelta(days=1)
        bounds = {
            "eq": (first, last),
            "in": (first, last),
            "ge": (first, None),
            "gt": (last + day, None),
            "le": (None, last),
            "lt": (None, first - day),
        }
        return bounds.get(operator)

    #
    # Devuelve el tipo de período de un campo: fecha, texto o año, o None.
    #
    def _period_kind(self, field):
        if field.sql_type in _DATE_TYPES:
            return "date"

        source = _single_source(field)
        if field.sql_type == "TEXT" and source in self._temporal:
            return "text"
        label = self.column(*source) if source else None
        if field.sql_type == "BIGINT" and label and singular_words(label) in YEAR_COLUMN_NAMES:
            return "year"
        return None

    #
    # Devuelve el primer y último día de un valor de año o de fecha.
    #
    @staticmethod
    def _period_days(value, kind):
        if kind == "year":
            return (date(value, 1, 1), date(value, 12, 31)) if type(value) is int else None

        moment = _iso_date(value)
        return (moment, moment) if moment else None

    #
    # Devuelve el período pedido que contiene los valores de texto filtrados, si existe.
    #
    def _requested_span(self, low, high):
        for start, end in self._periods:
            if (start is None or start <= low) and (end is None or high <= end):
                return start, end
        return None

    #
    # Redacta un período a partir de sus extremos, abierto o cerrado.
    #
    def _date_period(self, low, high):
        if high is None:
            return ("desde " if self.es else "from ") + self._period_bound(low, first=True)
        if low is None:
            return ("hasta " if self.es else "until ") + self._period_bound(high, first=False)

        if low == high:
            return ("el " if self.es else "on ") + date_text(low, self.language)
        whole_years = (low.month, low.day, high.month, high.day) == (1, 1, 12, 31)
        if whole_years and low.year == high.year:
            return ("en " if self.es else "in ") + str(low.year)
        if whole_years:
            return f"de {low.year} a {high.year}" if self.es else f"from {low.year} to {high.year}"

        whole_month = low.day == 1 and (high + timedelta(days=1)).day == 1
        if whole_month and (low.year, low.month) == (high.year, high.month):
            return ("en " if self.es else "in ") + grain(low, "month", self.language)[0]

        first, last = date_text(low, self.language), date_text(high, self.language)
        return f"del {first} al {last}" if self.es else f"from {first} to {last}"

    #
    # Redacta un extremo de un período abierto.
    #
    def _period_bound(self, day, first):
        if (day.month, day.day) == ((1, 1) if first else (12, 31)):
            return str(day.year)
        if (day.day == 1) if first else ((day + timedelta(days=1)).day == 1):
            return grain(day, "month", self.language)[0]
        return ("el " if self.es else "") + date_text(day, self.language)

    #
    # Redacta la restricción que impone un filtro sobre una columna.
    #
    def _restriction(self, predicate, ancestor):
        ref = predicate.field
        label = self.names.column(ref.dataset_id, ref.column)
        if _lists_places(predicate, label):
            places = [place_text(value) for value in predicate.values]
            return self._located(join_words(places, self.language))

        field = _source_field(ancestor, ref)

        return self.condition(predicate, label, field)

    #
    # Devuelve la oración que anuncia los filtros aplicados sobre resultados ya calculados.
    #
    def result_filters(self, chain):
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

    #
    # Redacta la condición de un filtro sobre un resultado calculado.
    #
    def _result_condition(self, predicate):
        ref = predicate.field
        field = _named_field(self._steps[ref.step_id], ref.field)
        label = self.names.result(ref.step_id, ref.field, field)

        return self.condition(predicate, label, field)

    #
    # Redacta la condición de un filtro según su operador y sus valores.
    #
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

    #
    # Devuelve los valores de un filtro formateados, con la unidad documentada.
    #
    def _condition_values(self, predicate, field):
        unit, documented = (
            display_unit(field, self.language) if field is not None else (None, False)
        )
        suffix = f" {unit}" if documented else ""

        return [
            value if isinstance(value, str) else self.value(value, exact=True) + suffix
            for value in predicate.values
        ]

    #
    # Redacta un límite sobre una fecha, o None si el filtro no es un rango de fechas.
    #
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

    #
    # Devuelve la relación textual del operador de un filtro (mayor que, igual a, etc.).
    #
    def _relation(self, predicate):
        relations = _SPANISH_RELATIONS if self.es else _ENGLISH_RELATIONS
        relation = relations.get(predicate.operator)
        if relation is not None:
            return relation
        if isinstance(predicate.values[0], str):
            return ""

        return "igual a " if self.es else "equal to "

    #
    # Responde con el período que cubren las filas de un resultado, con la precisión pedida.
    #
    def period_answer(self, rows, dated, asked):
        moments = [
            moment
            for row in rows
            for field, time_grain in dated
            if (moment := self._moment(field, row[field.name], time_grain)) is not None
        ]
        if not moments:
            if self.es:
                return (
                    "El resultado anterior no tiene una fecha asociada: "
                    "se calculó sin separar los registros por fecha."
                )
            return "The previous result has no date: it was calculated without separating records by date."

        labels = dict(
            label
            for start, end, precision in moments
            for label in self._labels(start, end, precision, asked)
        )
        texts = [labels[key] for key in sorted(labels)]

        return self._covered(texts, asked) + self._exact_day(rows, moments, asked) + "."

    #
    # Indica el día exacto si se preguntó año o mes y todo corresponde a un mismo día.
    #
    def _exact_day(self, rows, moments, asked):
        distinct = set(moments)
        if asked == "day" or len(distinct) != 1:
            return ""

        start, end, precision = next(iter(distinct))
        if precision != "day" or start != end:
            return ""

        day = date_text(start, self.language)
        if self.es:
            return f" ({'registro' if len(rows) == 1 else 'registros'} del {day})"

        return f" ({'record' if len(rows) == 1 else 'records'} from {day})"

    #
    # Responde con el período de los registros usados, de su primera a su última fecha.
    #
    def period_span(self, rows, first, last, asked):
        starts = [_as_day(row[first]) for row in rows if row[first] is not None]
        ends = [_as_day(row[last]) for row in rows if row[last] is not None]
        if not starts or not ends:
            return self.period_answer(rows, [], asked)

        start, end = min(starts), max(ends)
        days = self._day_span(start, end)
        if asked == "day":
            period = days
        else:
            labels = dict.fromkeys(self._labels(day, day, "day", asked)[0][1] for day in (start, end))
            joined = (" a " if self.es else " to ").join(labels)
            period = f"de {joined} ({days})" if self.es else f"from {joined} ({days})"

        if self.es:
            return f"El resultado anterior se calculó con registros {period}."
        return f"The previous result was calculated with records {period}."

    #
    # Devuelve el texto de un día o de un rango de días.
    #
    def _day_span(self, start, end):
        first, last = date_text(start, self.language), date_text(end, self.language)
        if self.es:
            return f"del {first}" if start == end else f"del {first} al {last}"

        return f"on {first}" if start == end else f"from {first} to {last}"

    #
    # Devuelve el intervalo y la precisión que nombra un valor temporal, o None.
    #
    def _moment(self, field, value, time_grain):
        if value is None:
            return None
        if time_grain == "year" and isinstance(value, int):
            return date(value, 1, 1), date(value, 12, 31), "year"
        if isinstance(value, date):
            precision = "month" if time_grain == "month" else "day"
            return value, value, precision
        if isinstance(value, str):
            return period_bounds(value, self._temporal.get(_single_source(field)))

        return None

    #
    # Devuelve las etiquetas ordenables de un intervalo con la precisión pedida.
    #
    def _labels(self, start, end, precision, asked):
        if asked == "year" or precision == "year":
            return [((year,), str(year)) for year in range(start.year, end.year + 1)]
        if asked == "month" or precision in {"month", "months"}:
            return [
                ((day.year, day.month), grain(day, "month", self.language)[0])
                for day in (start, end)
            ]

        return [((start.year, start.month, start.day), date_text(start, self.language))]

    #
    # Redacta qué período abarca o corresponde al resultado anterior.
    #
    def _covered(self, texts, asked):
        subject = "El resultado anterior" if self.es else "The previous result"
        if len(texts) > 4:
            first, last = texts[0], texts[-1]
            if not self.es:
                return f"{subject} spans from {first} to {last}"
            span = f"del {first} al {last}" if asked == "day" else f"de {first} a {last}"
            return f"{subject} abarca {span}"

        listed = join_words(texts, self.language)
        if not self.es:
            return f"{subject} corresponds to {listed}"
        if asked == "day":
            return f"{subject} corresponde " + ("al " if len(texts) == 1 else "a las fechas ") + listed

        return f"{subject} corresponde a {listed}"

    #
    # Redacta el valor más reciente de una medición y su fecha.
    #
    def latest(self, rows, field, date_field, fields, step, chain, inline_values):
        measured = self.names.result(step.id, field.name, field)
        label = measured + self.qualifier(chain, _shown(fields))
        when = rows[0][date_field.name]
        moment = self.when(date_field, when, temporal=True) or (self.value(when), "period")
        shared = self._shared_attributes(rows, fields, {field.name, date_field.name})
        where = (" " + ", ".join(shared)) if shared else ""

        if len(rows) == 1:
            return self._latest_value(rows[0][field.name], field, label, where, moment), False

        return self._latest_records(rows, field, label, where, moment, inline_values), True

    #
    # Devuelve los atributos de texto que comparten todas las filas.
    #
    def _shared_attributes(self, rows, fields, skip):
        shared = []
        for name, item in fields.items():
            if name in skip or item.sql_type != "TEXT":
                continue
            values = {row[name] for row in rows}
            if len(values) == 1 and None not in values:
                shared.append(self.attribute(item, next(iter(values))))

        return shared

    #
    # Redacta la medición más reciente cuando hay un único registro.
    #
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

    #
    # Redacta los registros más recientes cuando hay varios en la misma fecha.
    #
    def _latest_records(self, rows, field, label, where, moment, inline_values):
        counts = Counter(row[field.name] for row in rows)
        origin = dated(*moment, self.language)
        if len(counts) > inline_values:
            if self.es:
                total = self.count(len(rows), "registro", "registros")
                return (
                    f"Los registros más recientes de {label}{where} son {origin}: "
                    f"son {total} con valores distintos y podés verlos en la tabla."
                )
            total = self.count(len(rows), "record", "records")
            return (
                f"The most recent {label} records{where} are {origin}: "
                f"{total} with different values, listed in the table."
            )

        parts = [self._tally(field, value, times) for value, times in counts.items()]
        if self.es:
            return (
                f"Los registros más recientes de {label}{where} son {origin}: "
                f"{join_words(parts, 'es')}."
            )

        return f"The most recent {label} records{where} are {origin}: {join_words(parts, 'en')}."

    #
    # Devuelve cuántos registros tienen un valor, o ninguno.
    #
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

    #
    # Redacta el valor extremo de un ranking y los que le siguen, o None si no aplica.
    #
    def ranking(self, rows, step, fields, inline_rows, chain):
        order = step.order_by[0]
        name = _ref_name(order.field)
        field = fields.get(name)
        if not self.ranks_values(step, fields) or rows[0][name] is None:
            return None

        highest = order.direction == "desc"
        top = rows[0][name]
        label = self.names.result(step.id, name, field) + self.qualifier(chain, _shown(fields))
        tied = [row for row in rows if row[name] == top]
        listed = len(tied) <= inline_rows
        contexts = [self.context(row, fields, {name}) for row in tied] if listed else []
        places = [place for place in contexts if place]
        text = self._extreme(field, label, top, highest, len(tied), places)

        rest = [row for row in rows if row[name] != top]
        if not rest:
            return (text, False) if listed else (text + self._tied_table(), True)
        if len(rows) <= inline_rows:
            return text + self._following(rest, field, fields), False

        return text + self._ranked_table(len(rows), highest), True

    #
    # Indica si un orden clasifica por el valor de una medición: ordenar por año o por fecha no da un máximo.
    #
    def ranks_values(self, ranking, fields):
        if ranking is None or not ranking.order_by:
            return False

        field = fields.get(_ref_name(ranking.order_by[0].field))

        return field is not None and field.sql_type in NUMERIC_TYPES and self._period_kind(field) is None

    #
    # Devuelve la frase que remite a la tabla con los registros empatados en el extremo.
    #
    def _tied_table(self):
        return " Podés ver esos registros en la tabla." if self.es else " Those records are listed in the table."

    #
    # Redacta lo que destaca un listado extenso: el rango de cada medición, con dónde y cuándo se registró cada
    # extremo, o None si ninguna medición varía.
    #
    def records_pattern(self, rows, fields):
        measures = [field for field in fields.values() if self._measured(field)][:_MAX_PATTERN_MEASURES]
        varying = {name: field for name, field in fields.items() if len({row[name] for row in rows}) > 1}
        spreads = [item for item in (self._spread(rows, field, varying) for field in measures) if item]
        if not spreads:
            return None

        if self.es:
            return f"En esos registros, {join_words(spreads, 'es')}."

        return f"Across those records, {join_words(spreads, 'en')}."

    #
    # Redacta el grupo con el valor calculado más alto y el más bajo de un resultado agrupado extenso.
    #
    def grouped_pattern(self, rows, fields, calculations, grouping):
        values = [name for name in fields if name not in grouping]
        if len(values) != 1 or values[0] not in calculations:
            return None

        name = values[0]
        valued = [row for row in rows if row[name] is not None]
        if len(valued) < 2:
            return None

        high = max(valued, key=lambda row: row[name])
        low = min(valued, key=lambda row: row[name])
        if high[name] == low[name]:
            return None

        field, measured = fields[name], self.subject(calculations[name])
        highest, lowest = (
            f"{self.measure(field, row[name])} {self.group_scope(grouping, fields, row)}" for row in (high, low)
        )
        if self.es:
            return f"El valor máximo {_of(measured)} fue {highest} y el mínimo fue {lowest}."

        return f"The maximum value of {measured} was {highest} and the minimum was {lowest}."

    #
    # Indica si un campo es una medición numérica, no un año, una fecha ni un código.
    #
    def _measured(self, field):
        return field.sql_type in NUMERIC_TYPES and field.numeric_eligible and self._period_kind(field) is None

    #
    # Redacta el rango de una medición con dónde y cuándo se registraron su mínimo y su máximo; los ubica sólo
    # con los campos que varían entre las filas (un año común a todas no distingue nada).
    #
    def _spread(self, rows, field, fields):
        valued = [row for row in rows if row[field.name] is not None]
        if len(valued) < 2:
            return None

        low = min(valued, key=lambda row: row[field.name])
        high = max(valued, key=lambda row: row[field.name])
        if low[field.name] == high[field.name]:
            return None

        label = self.label(field)
        lowest, highest = (self._located_measure(row, field, fields) for row in (low, high))
        if self.es:
            return f"{label} va de {lowest} a {highest}"

        return f"{label} ranges from {lowest} to {highest}"

    #
    # Devuelve el valor de una medición en una fila con el lugar y el momento en que se registró.
    #
    def _located_measure(self, row, field, fields):
        measured = self.measure(field, row[field.name])
        where = self._locator(row, fields, field.name)

        return f"{measured} ({where})" if where else measured

    #
    # Devuelve dónde y cuándo se registró una fila: su lugar y su fecha, período o año, sin otras mediciones.
    #
    def _locator(self, row, fields, skip):
        parts = []
        for name, field in fields.items():
            value = row[name]
            if name == skip or value is None:
                continue

            moment = self.when(field, value)
            if moment is None and self._period_kind(field) == "year" and type(value) is int:
                moment = (str(value), "year")
            if moment is not None:
                parts.append(at(*moment, self.language))
            elif isinstance(value, str) and is_place(self.label(field)):
                parts.append(self._located(place_text(value)))

        return ", ".join(parts[:_MAX_LOCATOR_PARTS])

    #
    # Redacta el valor más alto o más bajo y dónde se registró.
    #
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

    #
    # Redacta los valores que siguen al extremo en un ranking.
    #
    def _following(self, rest, field, fields):
        following = [self._described(row, field, fields) for row in rest]
        if self.es:
            verb = "Le sigue" if len(rest) == 1 else "Le siguen"
            return f" {verb} {join_words(following, 'es')}."

        return f" Next: {join_words(following, 'en')}."

    #
    # Devuelve la frase que remite a la tabla con los registros ordenados.
    #
    def _ranked_table(self, total, highest):
        direction = self._direction_text(highest)
        if self.es:
            return (
                f" En la tabla podés ver los {self.count(total, 'registro', 'registros')} "
                f"ordenados {direction}."
            )

        return f" The table lists all {self.count(total, 'record', 'records')} {direction}."

    #
    # Devuelve el texto del sentido de un orden: de mayor a menor o al revés.
    #
    def _direction_text(self, descending):
        if self.es:
            return "de mayor a menor" if descending else "de menor a mayor"

        return "from highest to lowest" if descending else "from lowest to highest"

    #
    # Describe una fila con su medida y su contexto.
    #
    def _described(self, row, field, fields):
        measured = self.measure(field, row[field.name])
        context = self.context(row, fields, {field.name})

        return " ".join(filter(None, [measured, context]))

    #
    # Elige y redacta la oración que describe las filas de un resultado.
    #
    def rows(self, rows, fields, calculations, grouping, direction, inline_rows, chain):
        values = [name for name in fields if name not in grouping]
        grouped = self.qualifier(chain, _shown({name: fields[name] for name in grouping}))
        said = len(values) - len(_date_pairs(values, calculations, fields))

        if len(rows) == 1 and said <= INLINE_FIELDS:
            return self._single(rows[0], fields, values, calculations, grouping, grouped), False
        shown = self._record_essentials(rows[0], fields, values) if len(rows) == 1 and not calculations else []
        if shown:
            return self._single(rows[0], fields, shown, calculations, grouping, grouped), True

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
        several_calculations = bool(grouping) and len(values) <= INLINE_FIELDS and all(
            name in calculations for name in values
        )
        if few_rows and several_calculations:
            return self._grouped_values(rows, fields, values, calculations, grouping, grouped), False
        if few_rows and one_measure:
            qualifier = self.qualifier(chain, _shown(fields))
            return self._listed(rows, fields[numeric[0]], fields, qualifier), False

        text = self._lead(rows, fields, values, calculations, grouping, direction, grouped)

        return text, True

    #
    # Devuelve los campos que dicen lo esencial de un registro con muchos campos: sus mediciones y su fecha, si
    # caben en una oración; la tabla muestra el resto. Vacío si no tiene mediciones o son demasiadas.
    #
    def _record_essentials(self, row, fields, values):
        measured = [name for name in values if self._measured(fields[name])]
        dated = [name for name in values if name not in measured and self.when(fields[name], row[name]) is not None]
        if not measured or len(measured) > INLINE_FIELDS:
            return []

        return [*dated[:1], *measured]

    #
    # Redacta un resultado de una sola fila, con cálculos o campos.
    #
    def _single(self, row, fields, values, calculations, grouping, qualifier):
        scope = " ".join(filter(None, [qualifier.strip(), self.group_scope(grouping, fields, row)]))
        if not calculations:
            pairs = [self.pair(fields[name], row[name]) for name in values]
            listed = join_words(pairs, self.language)
            if self.es:
                return f"Encontré un registro con {listed}."
            return f"I found one record with {listed}."

        parts = []
        spans = _date_pairs(values, calculations, fields)
        for name in values:
            calculation = calculations.get(name)
            if name in spans.values():
                continue
            if name in spans:
                parts.append(self._records_span(row[name], row[spans[name]]))
                continue
            if calculation is None:
                parts.append(self.pair(fields[name], row[name]))
                continue
            subject = self.subject(calculation) + (f" {scope}" if scope and not parts else "")
            parts.append(self._calculated(subject, fields[name], row[name]))

        return sentence(join_words(parts, self.language)) + "."

    #
    # Redacta el período de los registros con su primera y su última fecha.
    #
    def _records_span(self, first, last):
        if first is None or last is None:
            if self.es:
                return "no se pudo obtener el período de los registros"
            return "the period of the records could not be obtained"

        start, end = _as_day(first), _as_day(last)
        if start == end:
            day = date_text(start, self.language)
            return f"todos los registros son del {day}" if self.es else f"all records are from {day}"

        return ("los registros van " if self.es else "the records go ") + self._day_span(start, end)

    #
    # Redacta el valor de un cálculo, o que no se pudo obtener.
    #
    def _calculated(self, subject, field, value):
        if value is None:
            if self.es:
                return f"no se pudo obtener {subject} con los datos disponibles"
            return f"{subject} could not be obtained from the available data"

        measured = self.measure(field, value)

        return f"{subject} fue {measured}" if self.es else f"{subject} was {measured}"

    #
    # Redacta un cálculo agrupado con su valor por cada grupo.
    #
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

    #
    # Redacta los cálculos de cada grupo cuando son pocos grupos: una oración por grupo con sus valores.
    #
    def _grouped_values(self, rows, fields, values, calculations, grouping, qualifier):
        texts = []
        for row in rows:
            scope = " ".join(filter(None, [self.group_scope(grouping, fields, row), qualifier.strip()]))
            parts = [
                self._calculated(self.subject(calculations[name]), fields[name], row[name]) for name in values
            ]
            texts.append(f"{sentence(scope)}, {join_words(parts, self.language)}.")

        return " ".join(texts)

    #
    # Redacta la lista de registros de una medición con su contexto.
    #
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

    #
    # Redacta la oración inicial de un resultado que se completa en la tabla.
    #
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

    #
    # Devuelve la frase que remite a la tabla, con su orden si lo hay.
    #
    def _table_hint(self, order):
        if self.es:
            if order:
                return f" y en la tabla aparecen ordenados {order}."
            return " y podés verlos completos en la tabla."

        if order:
            return f", listed in the table {order}."
        return "; you can see all of them in the table."

    #
    # Redacta la introducción de cálculos agrupados con la cantidad de resultados.
    #
    def _grouped_lead(self, subjects, grouping, qualifier, total, table):
        groups = join_words([self._group_name(spec) for spec in grouping.values()], self.language)
        if self.es:
            results = self.count(total, "resultado", "resultados")
            return f"Calculé {subjects} por {groups}{qualifier}. Son {results}{table}"

        results = self.count(total, "result", "results")

        return f"I calculated {subjects} by {groups}{qualifier}. There are {results}{table}"

    #
    # Devuelve el nombre de una agrupación, temporal o por campo.
    #
    def _group_name(self, spec):
        if spec.time_grain != "raw":
            return grain_label(spec.time_grain, self.language)

        return self.names.operand(spec.field)

    #
    # Redacta la oración de varios valores calculados en un solo registro.
    #
    def _calculated_values(self, variables):
        if self.es:
            calculated = self.count(variables, "valor", "valores")
            return f"Calculé {calculated}; podés verlos en la tabla."

        calculated = self.count(variables, "value", "values")

        return f"I calculated {calculated}; you can see them in the table."

    #
    # Redacta la oración de cálculos hechos para cada registro.
    #
    def _calculated_records(self, subjects, qualifier, total, table):
        if self.es:
            records = self.count(total, "registro", "registros")
            return f"Calculé {subjects} para cada uno de los {records}{qualifier}{table}"

        records = self.count(total, "record", "records")

        return f"I calculated {subjects} for each of the {records}{qualifier}{table}"

    #
    # Redacta la oración de un registro con varias variables.
    #
    def _one_record(self, variables):
        if self.es:
            measured = self.count(variables, "variable", "variables")
            return f"Encontré un registro con {measured}; podés verlas en la tabla."

        measured = self.count(variables, "variable", "variables")

        return f"I found one record with {measured}; you can see them in the table."

    #
    # Redacta la oración de los registros encontrados.
    #
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

    #
    # Devuelve las notas de columnas de texto leídas en su versión numérica.
    #
    def projections(self, record):
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

    #
    # Redacta la nota de una columna usada en su versión numérica y sus valores sin leer.
    #
    def _projection(self, dataset, column):
        original = _column_metadata(dataset, column.derived_from)
        if original:
            label = column_label(original.original_name, original.unit)
        else:
            label = humanize(column.derived_from)
        excluded = column.excluded_count
        censored = min(getattr(column, "censored_count", None) or 0, excluded)
        if censored:
            return self._censored_projection(label, excluded, censored)

        if self.es:
            unreadable = self.count(
                excluded,
                "valor que no se pudo leer como número queda",
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

    #
    # Redacta la nota de una columna con valores bajo el límite de detección: no entran en los cálculos, así
    # que un mínimo o un promedio describen sólo los valores detectados.
    #
    def _censored_projection(self, label, excluded, censored):
        others = excluded - censored
        if self.es:
            below = self.count(
                censored,
                "valor está informado como menor que un límite de detección o cuantificación («<…»)",
                "valores están informados como menores que un límite de detección o cuantificación («<…»)",
            )
            rest = f" y {self.count(others, 'valor más no se pudo leer', 'valores más no se pudieron leer')}" if others else ""
            return (
                f"En la columna {label}, {below}{rest}: no entran en los cálculos, así que el mínimo y el "
                "promedio describen sólo los valores cuantificados."
            )

        below = self.count(
            censored,
            "value is reported as below a detection or quantification limit («<…»)",
            "values are reported as below a detection or quantification limit («<…»)",
        )
        rest = f" and {self.count(others, 'more value could not be read', 'more values could not be read')}" if others else ""
        return (
            f"In {label}, {below}{rest}: they are left out of the calculations, so the minimum and the mean "
            "describe only the quantified values."
        )

    #
    # Devuelve las notas sobre registros excluidos o divisiones por cero en los cálculos.
    #
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

    #
    # Redacta una nota de exclusión con faltantes y divisiones por cero.
    #
    def _exclusion(self, valid, nulls, zeros, policy, origin):
        parts = []
        if nulls:
            parts.append(self._missing_values(valid, nulls, policy, origin))
        if zeros:
            parts.append(self._zero_divisions(zeros))

        return sentence(join_words(parts, self.language)) + "."

    #
    # Redacta qué se hizo con los valores faltantes según la política y la operación.
    #
    def _missing_values(self, valid, nulls, policy, origin):
        if policy == "exclude":
            return self._excluded_missing(valid, nulls)
        if origin.operation == "aggregate":
            return self._uncalculated_missing(nulls, origin)

        return self._kept_missing(nulls)

    #
    # Redacta que el cálculo dejó fuera los registros sin dato.
    #
    def _excluded_missing(self, valid, nulls):
        if self.es:
            used = self.count(valid, "registro", "registros")
            left_out = self.count(nulls, "registro sin dato", "registros sin dato")
            return f"el cálculo usó {used} con dato y dejó fuera {left_out}"

        used = self.count(valid, "record", "records")
        left_out = self.count(nulls, "record", "records")

        return f"the calculation used {used} with a value and left out {left_out} without one"

    #
    # Redacta que conservar faltantes dejó el resultado sin calcular.
    #
    def _uncalculated_missing(self, nulls, origin):
        if self.es:
            kept = self.count(
                nulls, "registro sin dato no se descartó", "registros sin dato no se descartaron"
            )
            groups = "de los grupos que los incluyen " if origin.group_by else ""
            return f"{kept}, así que el resultado {groups}quedó sin calcular"

        kept = self.count(nulls, "record without a value was", "records without a value were")
        groups = "of the groups that include them " if origin.group_by else ""

        return f"{kept} not discarded, so the result {groups}was not calculated"

    #
    # Redacta que los registros sin dato se conservaron vacíos.
    #
    def _kept_missing(self, nulls):
        if self.es:
            kept = self.count(
                nulls, "registro sin dato se conservó", "registros sin dato se conservaron"
            )
            return f"{kept} sin valor, en lugar de descartarse"

        kept = self.count(nulls, "record without a value was", "records without a value were")

        return f"{kept} kept empty instead of being discarded"

    #
    # Redacta los casos en que el divisor fue cero y el cociente quedó sin valor.
    #
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


#
# Devuelve los elementos sin repetidos, en su orden original.
#
def _unique(items):
    return list(dict.fromkeys(items))


#
# Devuelve el texto con la primera letra en minúscula.
#
def _lowercase_first(text):
    return f"{text[:1].lower()}{text[1:]}"


#
# Antepone «de» a un sujeto, contraído con su artículo masculino («del promedio»).
#
def _of(subject):
    return f"del {subject[3:]}" if subject.startswith("el ") else f"de {subject}"


#
# Devuelve un valor de un eje del gráfico con su tipo: las fechas vuelven a ser fechas y los enteros, que el
# gráfico guarda como float de presentación, vuelven a ser enteros.
#
def _typed(field, value):
    if field.sql_type == "BIGINT" and type(value) is float and value.is_integer():
        return int(value)
    if not isinstance(value, str):
        return value
    if field.sql_type == "DATE":
        return date.fromisoformat(value[:10])
    if field.sql_type == "TIMESTAMP":
        return datetime.fromisoformat(value)

    return value


#
# Devuelve la clave (dataset y columna) de una referencia a columna.
#
def _source_key(ref):
    return ref.dataset_id, ref.column


#
# Devuelve la clave de la única columna de origen de un campo, o None.
#
def _single_source(field):
    if len(field.sources) != 1:
        return None

    return _source_key(field.sources[0])


#
# Devuelve el nombre del campo o columna al que apunta una referencia.
#
def _ref_name(ref):
    return ref.field if ref.kind == "result" else ref.column


#
# Devuelve los metadatos de una columna de un dataset, o None.
#
def _column_metadata(dataset, sql_name):
    return next((item for item in dataset.columns_metadata if item.sql_name == sql_name), None)


#
# Devuelve el campo de salida de un paso con ese nombre, o None.
#
def _named_field(step, name):
    return next((item for item in step.output_fields if item.name == name), None)


#
# Devuelve el campo de salida de un paso que proviene de una columna de origen.
#
def _source_field(step, ref):
    target = _source_key(ref)
    for field in step.output_fields:
        if field.sources and _source_key(field.sources[0]) == target:
            return field

    return None


#
# Indica si un filtro es obvio para el lector y no hace falta mencionarlo.
#
def _goes_without_saying(predicate, shown):
    ref = predicate.field
    if ref.kind != "column" or predicate.operator == "is_not_null":
        return True

    return _source_key(ref) in shown and predicate.operator in _EQUALITY_OPERATORS


#
# Indica si un filtro enumera lugares por igualdad.
#
def _lists_places(predicate, label):
    return (
        predicate.operator in _EQUALITY_OPERATORS
        and is_place(label)
        and all(isinstance(value, str) for value in predicate.values)
    )


#
# Indica si los años o días forman una secuencia consecutiva.
#
def _contiguous_days(days) -> bool:
    ordered = sorted(days)
    return all(after[0] == before[1] + timedelta(days=1) for before, after in zip(ordered, ordered[1:]))


#
# Escribe un valor exacto, sin redondear, con los separadores del idioma y de miles en los enteros largos.
#
def _exact_number(value, language) -> str:
    text = format(Decimal(str(value)), "f")
    integer, _, fraction = text.partition(".")
    decimal, thousands = (",", ".") if language == "es" else (".", ",")
    if len(integer.lstrip("-")) >= _GROUPED_DIGITS:
        integer = f"{int(integer):,}".replace(",", thousands)

    return integer + (decimal + fraction if fraction else "")


#
# Devuelve la fecha ISO de un valor de filtro, o None si no lo es.
#
def _iso_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


#
# Devuelve las claves de origen de las columnas que ya se muestran.
#
def _shown(fields):
    return frozenset(_source_key(ref) for field in fields.values() for ref in field.sources)


#
# Indica si alguna fila tiene valor en la columna indicada.
#
def _has_value(rows, name):
    return any(row.get(name) is not None for row in rows)


#
# Devuelve el paso y todos los pasos de los que depende.
#
def _lineage(step, steps):
    chain = []
    visit = [step]
    while visit:
        ancestor = visit.pop()
        chain.append(ancestor)
        visit.extend(steps[key] for key in ancestor.depends_on)

    return chain


#
# Devuelve las filas del valor extremo de un ranking y sus empates.
#
def _top_ranked(rows, ranking, fields):
    order = ranking.order_by[0] if ranking is not None and ranking.order_by else None
    name = _ref_name(order.field) if order is not None else None
    field = fields.get(name)
    if field is None or field.sql_type not in NUMERIC_TYPES or not rows:
        return rows

    return [row for row in rows if row[name] == rows[0][name]]


#
# Devuelve el día de una fecha u hora.
#
def _as_day(value):
    return value.date() if isinstance(value, datetime) else value


#
# Empareja el mínimo y el máximo calculados sobre una misma fecha, que juntos dicen el período de los registros.
#
def _date_pairs(values, calculations, fields) -> dict[str, str]:
    extremes = {}
    for name in values:
        calculation = calculations.get(name)
        dated = fields[name].sql_type in _DATE_TYPES
        if calculation is None or not dated or calculation.function not in {"min", "max"}:
            continue
        if len(calculation.inputs) == 1:
            extremes.setdefault(calculation.inputs[0].field, {})[calculation.function] = name

    return {pair["min"]: pair["max"] for pair in extremes.values() if pair.keys() == {"min", "max"}}


#
# Devuelve los campos de primera y última fecha de una agregación de mínimo y máximo.
#
def _date_span(origin):
    if origin.operation != "aggregate" or origin.group_by or len(origin.calculations) != 2:
        return None

    names = {item.function: item.output_name for item in origin.calculations}
    fields = {field.name: field for field in origin.output_fields}
    if set(names) != {"min", "max"}:
        return None
    if any(fields[name].sql_type not in _DATE_TYPES for name in names.values()):
        return None

    return names["min"], names["max"]


#
# Devuelve el paso de origen, el de más reciente y el de ranking tras recorrer ordenamientos.
#
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
    if ranking is None and origin.operation == "read" and origin.order_by:
        ranking = origin

    return origin, latest, ranking


#
# Devuelve el sentido de orden de un ranking numérico, o None.
#
def _sort_direction(ranking, fields):
    if not ranking:
        return None

    order = ranking.order_by[0]
    sorted_field = fields.get(_ref_name(order.field))
    if sorted_field and sorted_field.sql_type in NUMERIC_TYPES:
        return order.direction

    return None


#
# Cuenta cuántos archivos distintos aportan evidencia.
#
def _file_count(evidence, sources):
    names = set()
    for item in evidence:
        for key in item.dataset_ids:
            names.add(sources[key].provenance.source_name)

    return len(names)


#
# Suma por cálculo las exclusiones de valores de un resultado.
#
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


#
# Agrupa los nombres de cálculos que comparten los mismos totales de exclusión.
#
def _names_by_total(totals):
    names = {}
    for name, total in totals.items():
        _, nulls, zeros, _ = total
        if nulls or zeros:
            names.setdefault(total, []).append(name)

    return names
