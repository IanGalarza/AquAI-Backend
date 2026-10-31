"""Contexto acotado y enunciados en lenguaje claro cuyos valores provienen de evidencia autorizada."""

from collections import Counter
from datetime import date
from decimal import Decimal
from hashlib import sha256
from itertools import count as counter
import json
import re

from aquai.application.contracts.answering import ResponseContextRecord
from aquai.application.execution import decode_rows, execution_records
from aquai.application.answering.naming import FieldNames
from aquai.application.answering.presentation import format_number
from aquai.application.answering.wording import (
    INLINE_FIELDS, NUMERIC_TYPES, at, calculation_meaning, column_label, count_text, date_text, dated, display_unit,
    file_stem, grain, grain_label, humanize, is_place, join_words, number_words, period, place_text, sentence,
    unit_suffix, with_article,
)


def response_language(question, preferred=None):
    if preferred is not None:
        return preferred
    words = set(re.findall(r"[a-záéíóúñ]+", question.lower()))
    english = len(words & {"what", "which", "show", "average", "mean", "please", "plot", "the", "rainfall", "how"})
    spanish = len(words & {"qué", "cuál", "mostrá", "promedio", "media", "gráfico", "graficá", "generá", "por", "los", "las", "del"})
    return "en" if english > spanish else "es"


class _Writer:
    """Redacta los enunciados con los metadatos verificados de los datasets seleccionados."""

    def __init__(self, sources, steps, language, temporal):
        self._sources, self._steps, self._temporal = sources, steps, temporal
        self.language, self.es = language, language == "es"
        self.names = FieldNames(steps, self.column, language)

    def label(self, field):
        if len(field.sources) == 1:
            ref = field.sources[0]
            label = self.column(ref.dataset_id, ref.column)
            if label:
                return label
        return humanize(field.name)

    def column(self, dataset_id, sql_name):
        dataset = self._sources.get(dataset_id)
        column = dataset and next((item for item in dataset.columns_metadata if item.sql_name == sql_name), None)
        if not column:
            return None
        label = column_label(column.original_name, column.unit)
        return label if len(label) <= 200 else column.sql_name

    def value(self, value, *, exact=False):
        if value is None:
            return "sin dato" if self.es else "no value"
        if isinstance(value, date):
            return date_text(value, self.language)
        if type(value) is bool:
            return ("sí" if value else "no") if self.es else ("yes" if value else "no")
        if isinstance(value, (Decimal, int, float)):
            if exact:
                # Los umbrales definen el alcance exacto: > 7,149 no es > 7,15.
                return str(value).replace(".", ",") if self.es else str(value)
            return format_number(value, self.language)
        return str(value)

    def measure(self, field, value):
        return self.value(value) + ("" if value is None else unit_suffix(field, self.language))

    def count(self, total, singular, plural):
        return count_text(total, singular, plural, self.language)

    def subject(self, calculation):
        return with_article(*self.names.calculation(calculation), self.language)

    def when(self, field, value, *, temporal=False):
        """Fecha tipada o período de texto; sólo las columnas temporales verificadas se interpretan."""
        if isinstance(value, date):
            return date_text(value, self.language), "day"
        key = (field.sources[0].dataset_id, field.sources[0].column) if len(field.sources) == 1 else None
        if isinstance(value, str) and (temporal or key in self._temporal):
            return period(value, self.language, self._temporal.get(key))
        return None

    def attribute(self, field, value):
        """Lugar, fecha u otra categoría que ubica un valor: en Punta Lara, el 12/03/2008."""
        label = self.label(field)
        if (moment := self.when(field, value)) is not None:
            return at(*moment, self.language)
        if is_place(label) and isinstance(value, str):
            return ("en " if self.es else "in ") + place_text(value)
        shown = value if isinstance(value, str) else self.measure(field, value)
        return f"con {label} {shown}" if self.es else f"with {label} {shown}"

    def context(self, row, fields, skip):
        return ", ".join(self.attribute(field, row[name]) for name, field in fields.items()
                         if name not in skip and row[name] is not None)

    def scope(self, spec, field, value):
        """Complemento de un grupo: en Punta Lara, en 2008, con Especie Microcystis aeruginosa."""
        if value is None:
            label = self.label(field)
            return f"sin dato de {label}" if self.es else f"without a {label} value"
        if spec.time_grain != "raw":
            return at(*grain(value, spec.time_grain, self.language), self.language)
        return self.attribute(field, value)

    def group_scope(self, grouping, fields, row):
        return join_words([self.scope(spec, fields[name], row[name]) for name, spec in grouping.items()], self.language)

    def pair(self, field, value):
        label = self.label(field)
        if value is None:
            return f"{label} sin dato" if self.es else f"{label} without a value"
        if (moment := self.when(field, value)) is not None:
            return f"{label} {moment[0]}"
        return f"{label} {value if isinstance(value, str) else self.measure(field, value)}"

    def attributed(self, evidence, text):
        """Con varias fuentes, cada resultado nombra la suya para no atribuirle un valor a otra."""
        return f"{'En' if self.es else 'In'} {self.files(evidence)}, {text[:1].lower()}{text[1:]}"

    def files(self, evidence):
        names = list(dict.fromkeys(file_stem(self._sources[key].provenance.source_name) for key in evidence.dataset_ids))
        if self.es:
            return ("el archivo " if len(names) == 1 else "los archivos ") + join_words(names, "es")
        return ("the file " if len(names) == 1 else "the files ") + join_words(names, "en")

    def introduction(self, files):
        if self.es:
            return f"Encontré información en {number_words(files)} archivos distintos, así que te muestro el resultado de cada uno por separado."
        return f"I found information in {number_words(files, 'en')} different files, so I show the result of each one separately."

    def meaning(self, calculation):
        return calculation_meaning(calculation.function, self.language, quantile=calculation.quantile)

    def summary(self, rows):
        if self.es:
            return ("Encontré un resultado; podés verlo en la tabla." if rows == 1 else
                    f"Encontré {self.count(rows, 'resultado', 'resultados')}; podés verlos en la tabla.")
        return ("I found one result; you can see it in the table." if rows == 1 else
                f"I found {self.count(rows, 'result', 'results')}; you can see them in the table.")

    def empty(self, evidence):
        if self.es:
            return f"No encontré registros que cumplan esos criterios en {self.files(evidence)}."
        return f"No records match those criteria in {self.files(evidence)}."

    def chart(self, chart, limit):
        es, rows, excluded = self.es, chart.input_rows, chart.excluded_rows
        if es:
            shown = "el resultado" if rows == 1 else f"los {self.count(rows, 'resultado', 'resultados')}"
            text = {"histogram": f"El histograma muestra cómo se distribuyen {self.count(rows - excluded, 'valor', 'valores')}: "
                                 "cada barra indica cuántos valores caen dentro de un intervalo.",
                    "bar": f"El gráfico de barras compara {shown}: la altura de cada barra es su valor.",
                    "line": f"El gráfico de líneas muestra cómo cambian {shown} a lo largo del eje horizontal.",
                    "scatter": f"El gráfico de dispersión ubica {shown} como puntos, para ver si las dos variables se relacionan.",
                    }[chart.chart_type]
        else:
            shown = self.count(rows, "result", "results")
            text = {"histogram": f"The histogram shows how {self.count(rows - excluded, 'value is', 'values are')} distributed: "
                                 "each bar counts the values that fall within an interval.",
                    "bar": f"The bar chart compares {shown}: the height of each bar is its value.",
                    "line": f"The line chart shows how {shown} change along the horizontal axis.",
                    "scatter": f"The scatter plot shows {shown} as points, to see whether the two variables are related.",
                    }[chart.chart_type]
        if limit and rows >= limit:
            text += (f" La consulta se limitó a {self.count(limit, 'resultado', 'resultados')}, así que no incluye todos los registros."
                     if es else f" The query was limited to {self.count(limit, 'result', 'results')}, so it does not include every record.")
        if excluded:
            text += ((f" {'Quedó' if excluded == 1 else 'Quedaron'} fuera {self.count(excluded, 'registro', 'registros')} sin "
                      f"datos válidos para graficar; no se {'reemplazó' if excluded == 1 else 'reemplazaron'} por cero.") if es else
                     (f" {self.count(excluded, 'record', 'records')} without valid data {'was' if excluded == 1 else 'were'} "
                      "left out; missing values were not replaced with zero."))
        return text

    def qualifier(self, chain, shown=frozenset()):
        """Filtros que acotan el resultado y la oración no muestra: en Punta Lara, con Turbidez mayor que 10 NTU.

        Que un registro tenga dato de la variable medida es obvio para el lector y no se repite."""
        parts = []
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                ref = predicate.field
                # Una igualdad sobre una columna visible ya se lee en sus valores; un rango nunca.
                shown_value = (ref.dataset_id, ref.column) in shown and predicate.operator in {"eq", "ieq", "in"}
                if ref.kind != "column" or predicate.operator == "is_not_null" or shown_value:
                    continue
                label = self.names.column(ref.dataset_id, ref.column)
                if predicate.operator in {"eq", "ieq", "in"} and is_place(label) and all(isinstance(value, str) for value in predicate.values):
                    parts.append(("en " if self.es else "in ") + join_words([place_text(value) for value in predicate.values], self.language))
                    continue
                field = next((item for item in ancestor.output_fields if item.sources
                              and (item.sources[0].dataset_id, item.sources[0].column) == (ref.dataset_id, ref.column)), None)
                parts.append(self.condition(predicate, label, field))
        parts = list(dict.fromkeys(parts))
        return (" " + ", ".join(parts)) if parts else ""

    def result_filters(self, chain):
        """Un filtro sobre resultados ya calculados cambia qué filas se ven: se dice siempre."""
        conditions = []
        for ancestor in reversed(chain):
            for predicate in ancestor.filters:
                ref = predicate.field
                if ref.kind == "result":
                    field = next((item for item in self._steps[ref.step_id].output_fields if item.name == ref.field), None)
                    conditions.append(self.condition(predicate, self.names.result(ref.step_id, ref.field, field), field))
        conditions = list(dict.fromkeys(conditions))
        if not conditions:
            return None
        if self.es:
            return f"Sólo incluí los resultados {join_words(conditions, 'es')}."
        return f"I only included the results {join_words(conditions, 'en')}."

    def condition(self, predicate, label, field):
        es, operator, name = self.es, predicate.operator, label
        preposition = "con" if es else "with"
        unit, documented = display_unit(field, self.language) if field is not None else (None, False)
        values = [value if isinstance(value, str) else self.value(value, exact=True) + (f" {unit}" if documented else "")
                  for value in predicate.values]
        if field is not None and field.sql_type in {"DATE", "TIMESTAMP"} and operator in {"lt", "le", "gt", "ge"}:
            moment = _iso_date(predicate.values[0])
            if moment is not None:
                shown = date_text(moment, self.language)
                return ({"ge": f"desde el {shown}", "gt": f"después del {shown}", "le": f"hasta el {shown}",
                         "lt": f"antes del {shown}"} if es else
                        {"ge": f"from {shown}", "gt": f"after {shown}", "le": f"until {shown}", "lt": f"before {shown}"})[operator]
        if operator == "is_null":
            return f"sin dato de {name}" if es else f"without a {name} value"
        if operator == "is_not_null":
            return f"con un valor de {name}" if es else f"with a {name} value"
        if operator == "in":
            return f"{preposition} {name} {join_words(values, self.language, 'o' if es else 'or')}"
        if operator == "not_in":
            return f"{'excluyendo' if es else 'excluding'} {name} {join_words(values, self.language)}"
        relation = ({"ne": "distinto de ", "lt": "menor que ", "le": "menor o igual que ", "gt": "mayor que ",
                     "ge": "mayor o igual que "} if es else
                    {"ne": "other than ", "lt": "below ", "le": "at most ", "gt": "above ", "ge": "at least "}).get(operator)
        if relation is None:
            relation = "" if isinstance(predicate.values[0], str) else ("igual a " if es else "equal to ")
        return f"{preposition} {name} {relation}{values[0]}"

    def latest(self, rows, field, date_field, fields, step, chain):
        """Última medición; si la fecha tiene varios registros, la tabla muestra a qué corresponde cada valor."""
        es, label = self.es, self.names.result(step.id, field.name, field) + self.qualifier(chain, _shown(fields))
        when = rows[0][date_field.name]
        moment = self.when(date_field, when, temporal=True) or (self.value(when), "period")
        # Atributos de texto compartidos por todos los registros (p. ej. el sitio) dan contexto.
        shared = [self.attribute(item, next(iter(values))) for name, item in fields.items()
                  if name not in {field.name, date_field.name} and item.sql_type == "TEXT"
                  and len(values := {row[name] for row in rows}) == 1 and None not in values]
        where = (" " + ", ".join(shared)) if shared else ""
        if len(rows) == 1:
            value = rows[0][field.name]
            if value is None:
                return (f"La medición más reciente de {label}{where}, {at(*moment, self.language)}, no tiene un valor registrado."
                        if es else f"The most recent {label} record{where}, {at(*moment, self.language)}, has no value."), False
            return (f"La medición más reciente de {label}{where} fue {self.measure(field, value)}, {at(*moment, self.language)}."
                    if es else f"The most recent {label} measurement{where} was {self.measure(field, value)}, {at(*moment, self.language)}."), False
        parts = [(self.count(times, "registro sin dato", "registros sin dato") if es else self.count(times, "record without a value", "records without a value"))
                 if value is None else
                 f"{self.count(times, 'registro', 'registros') if es else self.count(times, 'record', 'records')} "
                 f"{'con' if es else 'with'} {self.measure(field, value)}"
                 for value, times in Counter(row[field.name] for row in rows).items()]
        if es:
            return f"Los registros más recientes de {label}{where} son {dated(*moment, self.language)}: {join_words(parts, 'es')}.", True
        return f"The most recent {label} records{where} are {dated(*moment, self.language)}: {join_words(parts, 'en')}.", True

    def ranking(self, rows, step, fields, inline_rows, chain):
        """Resultado ordenado: el extremo pedido primero, con sus atributos y posibles empates."""
        order = step.order_by[0]
        name = order.field.field if order.field.kind == "result" else order.field.column
        field = fields.get(name)
        if field is None or field.sql_type not in NUMERIC_TYPES or rows[0][name] is None:
            return None
        es, highest, top = self.es, order.direction == "desc", rows[0][name]
        label = self.names.result(step.id, name, field) + self.qualifier(chain, _shown(fields))
        tied = [row for row in rows if row[name] == top]
        places = [place for place in (self.context(row, fields, {name}) for row in tied) if place]
        if es:
            text = f"El valor más {'alto' if highest else 'bajo'} de {label} fue {self.measure(field, top)}"
            if len(tied) > 1:
                text += f" y se registró {join_words(places, 'es')}" if places else f" y aparece en {len(tied)} registros"
        else:
            text = f"The {'highest' if highest else 'lowest'} {label} value was {self.measure(field, top)}"
            if len(tied) > 1:
                text += f", recorded {join_words(places, 'en')}" if places else f", in {len(tied)} records"
        if len(tied) == 1 and places:
            text += (" y se registró " if es else ", recorded ") + places[0]
        text += "."
        rest = [row for row in rows if row[name] != top]
        if not rest:
            return text, False
        if len(rows) <= inline_rows:
            following = [" ".join(filter(None, [self.measure(field, row[name]), self.context(row, fields, {name})])) for row in rest]
            if es:
                return text + f" {'Le sigue' if len(rest) == 1 else 'Le siguen'} {join_words(following, 'es')}.", False
            return text + f" Next: {join_words(following, 'en')}.", False
        direction = ("de mayor a menor" if highest else "de menor a mayor") if es else ("from highest to lowest" if highest else "from lowest to highest")
        if es:
            return text + f" En la tabla podés ver los {self.count(len(rows), 'registro', 'registros')} ordenados {direction}.", True
        return text + f" The table lists all {self.count(len(rows), 'record', 'records')} {direction}.", True

    def rows(self, rows, fields, calculations, grouping, direction, inline_rows, chain):
        """Pocos resultados se describen en una oración; los demás se presentan en la tabla."""
        values = [name for name in fields if name not in grouping]
        # Los grupos se nombran en la oración; las columnas de una lectura, en sus valores.
        grouped = self.qualifier(chain, _shown({name: fields[name] for name in grouping}))
        if len(rows) == 1 and len(values) <= INLINE_FIELDS:
            return self._single(rows[0], fields, values, calculations, grouping, grouped), False
        if 1 < len(rows) <= inline_rows:
            numeric = [name for name in values if fields[name].sql_type in NUMERIC_TYPES]
            if grouping and len(values) == 1 and values[0] in calculations:
                return self._grouped(rows, fields[values[0]], calculations[values[0]], grouping, fields, grouped), False
            if not grouping and not calculations and len(numeric) == 1 and len(values) <= INLINE_FIELDS:
                return self._listed(rows, fields[numeric[0]], fields, self.qualifier(chain, _shown(fields))), False
        return self._lead(rows, fields, values, calculations, grouping, direction, grouped), True

    def _single(self, row, fields, values, calculations, grouping, qualifier):
        es, scope = self.es, " ".join(filter(None, [qualifier.strip(), self.group_scope(grouping, fields, row)]))
        if not calculations:
            pairs = join_words([self.pair(fields[name], row[name]) for name in values], self.language)
            return f"Encontré un registro con {pairs}." if es else f"I found one record with {pairs}."
        parts = []
        for name in values:
            field, value, calculation = fields[name], row[name], calculations.get(name)
            if calculation is None:
                parts.append(self.pair(field, value))
                continue
            subject = self.subject(calculation) + (f" {scope}" if scope and not parts else "")
            if value is None:
                parts.append(f"no se pudo obtener {subject} con los datos disponibles" if es else
                             f"{subject} could not be obtained from the available data")
            else:
                parts.append(f"{subject} fue {self.measure(field, value)}" if es else f"{subject} was {self.measure(field, value)}")
        return sentence(join_words(parts, self.language)) + "."

    def _grouped(self, rows, field, calculation, grouping, fields, qualifier):
        missing = "sin valor calculable" if self.es else "no computable value"
        items = [f"{self.measure(field, row[field.name]) if row[field.name] is not None else missing} "
                 f"{self.group_scope(grouping, fields, row)}" for row in rows]
        verb = "fue" if self.es else "was"
        return sentence(f"{self.subject(calculation)}{qualifier} {verb} {join_words(items, self.language)}.")

    def _listed(self, rows, field, fields, qualifier):
        label = self.label(field) + qualifier
        items = [" ".join(filter(None, [self.measure(field, row[field.name]), self.context(row, fields, {field.name})])) for row in rows]
        if self.es:
            return f"Encontré {self.count(len(rows), 'registro', 'registros')} de {label}: {join_words(items, 'es')}."
        return f"I found {self.count(len(rows), 'record', 'records')} of {label}: {join_words(items, 'en')}."

    def _lead(self, rows, fields, values, calculations, grouping, direction, qualifier):
        es, total = self.es, len(rows)
        order = ""
        if direction:
            order = ("de mayor a menor" if direction == "desc" else "de menor a mayor") if es else \
                ("from highest to lowest" if direction == "desc" else "from lowest to highest")
        subjects = join_words([self.subject(calculations[name]) for name in values if name in calculations], self.language)
        table = ((f" y en la tabla aparecen ordenados {order}." if order else " y podés verlos completos en la tabla.") if es else
                 (f", listed in the table {order}." if order else "; you can see all of them in the table."))
        if calculations and grouping:
            groups = join_words([grain_label(spec.time_grain, self.language) if spec.time_grain != "raw"
                                 else self.names.operand(spec.field) for spec in grouping.values()], self.language)
            if es:
                return f"Calculé {subjects} por {groups}{qualifier}. Son {self.count(total, 'resultado', 'resultados')}{table}"
            return f"I calculated {subjects} by {groups}{qualifier}. There are {self.count(total, 'result', 'results')}{table}"
        if calculations and total == 1:
            return (f"Calculé {self.count(len(values), 'valor', 'valores')}; podés verlos en la tabla." if es else
                    f"I calculated {self.count(len(values), 'value', 'values')}; you can see them in the table.")
        if calculations:
            return (f"Calculé {subjects} para cada uno de los {self.count(total, 'registro', 'registros')}{qualifier}{table}" if es else
                    f"I calculated {subjects} for each of the {self.count(total, 'record', 'records')}{qualifier}{table}")
        if total == 1:
            return (f"Encontré un registro con {self.count(len(values), 'variable', 'variables')}; podés verlas en la tabla." if es else
                    f"I found one record with {self.count(len(values), 'variable', 'variables')}; you can see them in the table.")
        numeric = [fields[name] for name in values if fields[name].sql_type in NUMERIC_TYPES]
        measured = ""
        if len(numeric) == 1:
            measured = f" de {self.label(numeric[0])}" if es else f" of {self.label(numeric[0])}"
        if es:
            return f"Encontré {self.count(total, 'registro', 'registros')}{measured}{qualifier}{table}"
        return f"I found {self.count(total, 'record', 'records')}{measured}{qualifier}{table}"

    def projections(self, record):
        """Columnas de texto leídas en su versión numérica; el recuento de valores no numéricos es del dataset."""
        texts = []
        for field in record.columns:
            for ref in field.sources:
                dataset = self._sources[ref.dataset_id]
                column = next(item for item in dataset.columns_metadata if item.sql_name == ref.column)
                if not column.derived_from or not column.excluded_count:
                    continue
                original = next((item for item in dataset.columns_metadata if item.sql_name == column.derived_from), None)
                label = column_label(original.original_name, original.unit) if original else humanize(column.derived_from)
                excluded = column.excluded_count
                texts.append(
                    f"La columna {label} se usa en su versión numérica: en la tabla de origen, "
                    f"{self.count(excluded, 'un valor que no se pudo leer como número queda', 'valores que no se pudieron leer como número quedan')} "
                    "sin dato." if self.es else
                    f"{label} is used in its numeric version: in the source table, "
                    f"{self.count(excluded, 'value that could not be read as a number stays', 'values that could not be read as numbers stay')} empty.")
        return list(dict.fromkeys(texts))

    def exclusions(self, record, origin, fields):
        """Registros sin dato o divisiones por cero, sumados por cálculo sobre todos sus grupos."""
        calculations = {item.output_name: item for item in origin.calculations}
        totals = {}
        for item in record.exclusions:
            valid, nulls, zeros, _ = totals.get(item.output_name, (0, 0, 0, item.policy))
            totals[item.output_name] = (valid + item.valid_rows, nulls + item.null_rows, zeros + item.zero_divisions, item.policy)
        distinct = {}
        for name, total in totals.items():
            if total[1] or total[2]:
                distinct.setdefault(total, []).append(name)
        texts = []
        for (valid, nulls, zeros, policy), names in distinct.items():
            text = self._exclusion(valid, nulls, zeros, policy, origin)
            subjects = [self.subject(calculations[name]) for name in names if name in calculations and name in fields]
            # Con varios cálculos, la exclusión nombra a cuál corresponde.
            if len(calculations) > 1 and subjects:
                text = f"{'Para' if self.es else 'For'} {join_words(subjects, self.language)}, {text[:1].lower()}{text[1:]}"
            texts.append(text)
        return texts

    def _exclusion(self, valid, nulls, zeros, policy, origin):
        es, parts = self.es, []
        if nulls and policy == "exclude":
            parts.append(f"el cálculo usó {self.count(valid, 'registro', 'registros')} con dato y dejó fuera "
                         f"{self.count(nulls, 'registro sin dato', 'registros sin dato')}" if es else
                         f"the calculation used {self.count(valid, 'record', 'records')} with a value and left out "
                         f"{self.count(nulls, 'record', 'records')} without one")
        elif nulls and origin.operation == "aggregate":
            # Conservar un faltante en una estadística la deja sin valor: no se calcula sobre datos incompletos.
            groups = "de los grupos que los incluyen " if origin.group_by else ""
            parts.append(f"{self.count(nulls, 'registro sin dato no se descartó', 'registros sin dato no se descartaron')}, "
                         f"así que el resultado {groups}quedó sin calcular" if es else
                         f"{self.count(nulls, 'record without a value was', 'records without a value were')} not discarded, "
                         f"so the result {'of the groups that include them ' if origin.group_by else ''}was not calculated")
        elif nulls:
            parts.append(f"{self.count(nulls, 'registro sin dato se conservó', 'registros sin dato se conservaron')} "
                         "sin valor, en lugar de descartarse" if es else
                         f"{self.count(nulls, 'record without a value was', 'records without a value were')} kept empty "
                         "instead of being discarded")
        if zeros:
            parts.append(f"en {self.count(zeros, 'caso', 'casos')} el divisor fue cero y el cociente quedó sin valor "
                         "(no se reemplazó por cero)" if es else
                         f"in {self.count(zeros, 'case', 'cases')} the divisor was zero, so the ratio has no value "
                         "(it was not replaced with zero)")
        return sentence(join_words(parts, self.language)) + "."


def _iso_date(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _shown(fields):
    """Columnas de origen cuyos valores la oración ya muestra."""
    return frozenset((ref.dataset_id, ref.column) for field in fields.values() for ref in field.sources)


def _lineage(step, steps):
    chain, visit = [], [step]
    while visit:
        ancestor = visit.pop()
        chain.append(ancestor)
        visit.extend(steps[key] for key in ancestor.depends_on)
    return chain


def build_response_context(state, budget_bytes, max_preview_rows=3, max_sentences=24):
    """Catálogo de enunciados verificados; max_preview_rows filas se describen en texto y el resto en tablas."""
    validation, plan = state.validation, state.planning
    language = response_language(state.question, state.response_language)
    steps = {step.id: step for step in plan.operations}
    sources = {item.dataset_id: item for item in plan.selected_datasets}
    files = len({sources[key].provenance.source_name for item in validation.usable_evidence for key in item.dataset_ids})
    multi_source = files > 1
    writer = _Writer(sources, steps, language,
                     {(item.dataset_id, item.column): item.day_first for item in plan.temporal_text_columns})
    records = execution_records(state)
    catalog = []
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

    if multi_source:
        body["introduction"] = writer.introduction(files)
    # Identificadores únicos aunque luego se retiren las referencias genéricas de tablas descritas.
    sentence_ids = counter(1)

    def add(evidence, text, *, kind="result", attach=False, required=False, attribute=True):
        # Alcances y referencias obligatorias nunca se descartan por el tope ni por el presupuesto.
        if not required and len(catalog) >= max_sentences:
            return None
        if multi_source and kind == "result" and attribute:
            text = writer.attributed(evidence, text)
        entry = {"sentence_id": f"s{next(sentence_ids)}", "evidence_ids": [evidence.evidence_id],
                 "operation_id": evidence.operation_id, "kind": kind, "direct": text, "conversational": text,
                 "attachment": evidence.evidence_id if attach else None}
        catalog.append(entry)
        if not required and len(serialized()) > budget_bytes:
            catalog.pop()
            return None
        return entry

    # Todas las ramas autorizadas tienen una referencia aunque no quepan sus enunciados.
    summaries = {}
    for evidence in validation.usable_evidence:
        if evidence.kind == "visualization":
            chart = next(item for item in state.python_execution.visualizations if item.operation_id == evidence.operation_id)
            limits = [step.limit for step in _lineage(steps[evidence.operation_id], steps) if step.limit]
            add(evidence, writer.chart(chart, min(limits) if limits else None), kind="chart", attach=True, required=True)
        else:
            summaries[evidence.evidence_id] = add(evidence, writer.summary(evidence.rows), kind="summary", attach=True, required=True)

    omitted, meanings = {}, {}

    # Mismo orden que la respuesta: los resultados derivados antes que las filas leídas que los respaldan.
    ordered = sorted(validation.usable_evidence, key=lambda item: steps[item.operation_id].operation == "read")
    for evidence in ordered:
        if evidence.kind != "table":
            continue
        step = steps[evidence.operation_id]
        chain = _lineage(step, steps)
        if (filtered := writer.result_filters(chain)) is not None:
            add(evidence, filtered, kind="scope", required=True)
        origin = step
        latest = ranking = None
        while origin.operation in {"sort", "latest"}:
            if origin.operation == "latest":
                latest = origin
            elif ranking is None and origin.order_by:
                ranking = origin
            origin = steps[origin.depends_on[0]]
        calculations = {item.output_name: item for item in origin.calculations}
        grouping = {item.output_name: item for item in origin.group_by}
        record = records[evidence.operation_id]
        rows = decode_rows(record, record.columns)
        fields = {field.name: field for field in record.columns}
        numeric = [field for field in record.columns if field.numeric_eligible and field.sql_type in NUMERIC_TYPES]
        result = None
        if not rows:
            result = writer.empty(evidence), False
        elif latest and len(rows) <= max_preview_rows and len(numeric) == 1:
            result = writer.latest(rows, numeric[0], fields[latest.order_by[0].field.field], fields, latest, chain)
        elif ranking and not calculations and not grouping:
            result = writer.ranking(rows, ranking, fields, max_preview_rows, chain)
        if result is None:
            sort = ranking.order_by[0].field if ranking else None
            sorted_field = fields.get(sort.field if sort.kind == "result" else sort.column) if sort else None
            direction = ranking.order_by[0].direction if sorted_field and sorted_field.sql_type in NUMERIC_TYPES else None
            result = writer.rows(rows, fields, calculations, grouping, direction, max_preview_rows, chain)
        text, attach = result
        entry = add(evidence, text, attach=attach, attribute=bool(rows))
        if entry is not None and summaries.get(evidence.evidence_id) in catalog:
            # El enunciado específico reemplaza la referencia genérica de la tabla.
            catalog.remove(summaries[evidence.evidence_id])
        if entry is None or attach:
            omitted[evidence.operation_id] = len(rows)
        # Una estadística sólo se define si produjo algún valor; se explica junto al primer resultado que la usa.
        for calculation in origin.calculations:
            meaning = writer.meaning(calculation)
            if meaning and any(row.get(calculation.output_name) is not None for row in rows):
                meanings.setdefault(meaning, evidence)
        for detail in [*writer.projections(record), *writer.exclusions(record, origin, fields)]:
            add(evidence, detail, kind="detail")
    # Las definiciones son contexto: entran al final y sólo si queda lugar, sin desplazar resultados ni exclusiones.
    for meaning, evidence in meanings.items():
        add(evidence, meaning, kind="meaning")
    body["omitted_rows"] = omitted
    # El historial es contexto lingüístico opcional, nunca sustituye evidencia.
    while body["conversation_context"] and len(serialized()) > budget_bytes:
        body["conversation_context"].pop(0)
    content = serialized()
    return ResponseContextRecord(evidence=body, sha256=sha256(content).hexdigest(), size_bytes=len(content),
        budget_bytes=budget_bytes, omitted_rows=omitted), language
