import re
from decimal import Decimal

from aquai.application.answering.hypotheses import documented_values
from aquai.application.answering.measures import shown_value
from aquai.application.answering.presentation import format_unit
from aquai.application.answering.readings import (
    Reading,
    number_readings,
    sentences,
    written_nearby,
)
from aquai.application.answering.verification import RequiredValue, rows_of_place
from aquai.application.answering.wording import join_words
from aquai.application.contracts.answering import ComparisonFact, ReferenceLine, ReferenceValue
from aquai.application.contracts.retrieval import ReferenceDraft
from aquai.application.matching import (
    contains_phrase,
    named_variable_groups,
    names_subject,
    requires_chemical_basis,
)
from aquai.application.Rules.documents import normalized_text
from aquai.application.Rules.evidence import (
    DETECTION_LIMIT,
    DOCUMENT_YEAR,
    GUIDELINE_CUES,
    MEAN_WITH_DEVIATION,
    REFERENCE_LINE_CUES,
    WATER_USES,
)
from aquai.application.units import comparable_unit, concentration_basis, counts_organisms, unit_key

_MAX_SCALE_GAP = 10
_TABLE_CELL = "|"
_MAX_LINES = 3
_MIN_CONDITION_WORDS = 2
_MAX_CONDITION_CHARS = 120
_MAX_GUIDELINES = 4
_CITED_NUMBER = re.compile(
    r"\b(?:tablas?|tables?|fig(?:ura|ure)?s?|cap[ií]tulos?|chapters?|secci[oó]n|sections?|p|pp|vol|n[º°o])\.?\s*$",
    re.IGNORECASE,
)
_CONTRACTIONS = re.compile(r"\b(de|a) el\b ", re.IGNORECASE)
_USE_STEMS = {
    "drinking": ("consumo", "potable", "bebida", "drinking", "tap water"),
    "recreational": ("recrea", "bano", "baño", "bathing", "swimming"),
    "irrigation": ("riego", "irrigation"),
    "aquatic_life": ("vida acuatica", "vida acuática", "biota", "aquatic"),
}
_USE_LABELS = {
    "es": {
        "drinking": "agua de consumo",
        "recreational": "uso recreativo",
        "irrigation": "riego",
        "aquatic_life": "vida acuática",
    },
    "en": {
        "drinking": "drinking water",
        "recreational": "recreational use",
        "irrigation": "irrigation",
        "aquatic_life": "aquatic life",
    },
}
_KIND_LABELS = {
    "es": {
        "guideline": "el valor guía",
        "limit": "el límite",
        "typical_range": "el rango típico",
        "observed": "lo reportado en ese lugar",
    },
    "en": {
        "guideline": "the guideline value",
        "limit": "the limit",
        "typical_range": "the typical range",
        "observed": "what was reported there",
    },
}
_COMPARABLE_KINDS = ("guideline", "limit", "typical_range")
_LINE_KINDS = ("guideline", "limit")
_INCOMPARABLE = ("other_unit", "other_basis")


#
# Devuelve los valores guía o límites que los fragmentos confiables fijan para lo preguntado: oraciones que
# dicen una guía o un límite, nombran el tema, no son de otro uso del agua y dan una cifra que no es un año
# ni el número de una tabla o figura.
#
def documented_guidelines(fragments, subject, question) -> list[RequiredValue]:
    asked = water_uses(question)
    found = {}
    for fragment in fragments:
        heading = (fragment.section or [""])[-1]
        for line in (part for row in fragment.text.splitlines() for part in sentences(row)):
            if fragment.doubts or not _states_guideline(line, subject):
                continue
            uses = water_uses(f"{line} {heading}")
            reading = _leading_value(line, fragment.language)
            if reading is None or (asked and uses and not uses & asked):
                continue
            item = RequiredValue(fragment.evidence_id, reading.values, reading.text, line)
            found.setdefault(min(reading.values), item)

    return list(found.values())[:_MAX_GUIDELINES]


#
# Indica si una oración fija una guía o un límite para el tema preguntado.
#
def _states_guideline(line, subject) -> bool:
    return (
        GUIDELINE_CUES.search(line) is not None
        and DETECTION_LIMIT.search(line) is None
        and names_subject(line, subject)
    )


#
# Devuelve la primera cifra de una oración que no es un año ni el número de una tabla, figura o página.
#
def _leading_value(line, language):
    for reading in number_readings(line, language):
        year = reading.unit is None and DOCUMENT_YEAR.fullmatch(reading.text) is not None
        if not year and not _CITED_NUMBER.search(line[: reading.start]):
            return reading

    return None


#
# Devuelve los usos del agua que nombra un texto, según el vocabulario cerrado.
#
def water_uses(text: str) -> set[str]:
    return {
        use
        for use, phrases in WATER_USES.items()
        if any(contains_phrase(text, phrase, plural_insensitive=True) for phrase in phrases)
    }


#
# Devuelve los valores que los fragmentos reportan para el mismo lugar, leídos sin el modelo.
#
def local_references(fragments, results, question, places) -> list[ReferenceValue]:
    asked = water_uses(question)
    found = {}
    for fragment in fragments:
        heading = " ".join([*fragment.section[-2:], fragment.element_caption or ""])
        for result in results:
            for value in documented_values(result, [fragment]):
                if not any(contains_phrase(f"{heading} {value.quote}", place) for place in places):
                    continue
                reference = _reference(_read_draft(value, result), fragment, result, asked, places)
                found.setdefault(_identity(reference), reference)

    return list(found.values())


#
# Expresa un valor leído del fragmento con la forma de una referencia propuesta.
#
def _read_draft(value, result) -> ReferenceDraft:
    return ReferenceDraft(
        document_id=value.evidence_id,
        variable=result.variable,
        quote=value.quote[:240],
        low=float(value.low),
        high=float(value.high),
        unit=value.unit,
        kind="observed",
    )


#
# Conserva las referencias propuestas que se comprueban contra su fragmento y su resultado.
#
def verified_references(drafts, fragments, results, question, places=()) -> list[ReferenceValue]:
    documents = {item.evidence_id: item for item in fragments}
    asked = water_uses(question)
    verified = {}
    for draft in drafts:
        fragment = documents.get(draft.document_id)
        result = _target(draft, results)
        if fragment is None or result is None or fragment.doubts:
            continue
        draft = _with_stated_mean(_with_counted_unit(draft))
        if not _literal(draft, fragment) or not _names_variable(draft, fragment, result):
            continue
        if not _in_variable_cell(draft, fragment, result) or _detection_limit(draft, fragment):
            continue

        _keep(verified, _reference(draft, fragment, result, asked, places))

    return list(verified.values())


#
# Guarda una referencia sin repetir su valor; entre dos iguales prefiere la que dice su condición.
#
def _keep(verified, reference) -> None:
    key = _identity(reference)
    current = verified.get(key)
    if current is None or (current.condition is None and reference.condition is not None):
        verified[key] = reference


#
# Deja en la unidad propuesta sólo la unidad: en «cyanobacterial cells/ml» el organismo no es parte de ella.
#
def _with_counted_unit(draft):
    words = (draft.unit or "").split()
    for index in range(1, len(words)):
        unit = " ".join(words[index:])
        if counts_organisms(unit) and not counts_organisms(draft.unit):
            return draft.model_copy(update={"unit": unit})

    return draft


#
# Indica si el valor es un límite de detección o de cuantificación, que no es un valor de referencia.
#
def _detection_limit(draft, fragment) -> bool:
    return DETECTION_LIMIT.search(f"{_lead_in(draft.quote, fragment.text)} {draft.quote}") is not None


#
# Usa la media que figura en una cita «media ± desvío»: los extremos no son un dato del documento.
#
def _with_stated_mean(draft):
    for found in MEAN_WITH_DEVIATION.finditer(draft.quote):
        for reading in number_readings(found.group("mean"))[:1]:
            mean = float(min(reading.values))
            if _within_bounds(mean, draft):
                return draft.model_copy(update={"low": mean, "high": mean})

    return draft


#
# Indica si un valor cae entre las cotas que propuso el modelo.
#
def _within_bounds(value, draft) -> bool:
    return (draft.low is None or draft.low <= value) and (draft.high is None or value <= draft.high)


#
# Indica si, en una tabla, el valor está en la columna o en la fila de la variable.
#
def _in_variable_cell(draft, fragment, result) -> bool:
    rows = [line for line in fragment.text.splitlines() if _TABLE_CELL in line]
    quote = normalized_text(draft.quote)
    found = [line for line in rows if quote in normalized_text(line)]
    if not found:
        return _TABLE_CELL not in draft.quote

    return any(_cell_of_variable(row, rows[0], draft, result) for row in found)


#
# Indica si la fila nombra la variable, o si el valor está en una columna que la nombra.
#
def _cell_of_variable(row, first_row, draft, result) -> bool:
    if _names_any(row, result.names):
        return True

    header = first_row.split(_TABLE_CELL)
    wanted = {_decimal(item) for item in (draft.low, draft.high) if item is not None}

    return any(
        index < len(header)
        and _names_any(header[index], result.names)
        and wanted & {value for item in number_readings(cell) for value in item.values}
        for index, cell in enumerate(row.split(_TABLE_CELL))
    )


#
# Devuelve el resultado al que se refiere una referencia propuesta.
#
def _target(draft, results):
    wanted = normalized_text(draft.variable)
    for result in results:
        names = (result.variable, result.column, result.label)
        if any(normalized_text(name) == wanted for name in names):
            return result

    return results[0] if len(results) == 1 else None


#
# Indica si la cita, sus números y su unidad figuran tal cual en el fragmento.
#
def _literal(draft, fragment) -> bool:
    quote = normalized_text(draft.quote)
    if not quote or quote not in normalized_text(fragment.text):
        return False

    readings = number_readings(draft.quote)
    shown = {value for reading in readings for value in reading.values}
    bounds = [item for item in (draft.low, draft.high) if item is not None]
    if not bounds or any(_decimal(item) not in shown for item in bounds):
        return False

    return draft.unit is None or all(_bound_has_unit(item, draft) for item in bounds)


#
# Indica si la cita escribe esa cota con esa unidad, y no otra cifra cercana.
#
def _bound_has_unit(bound, draft) -> bool:
    value = _decimal(bound)

    return written_nearby(Reading(str(bound), frozenset({value}), draft.unit), draft.quote)


#
# Convierte un número del modelo en un decimal normalizado.
#
def _decimal(value) -> Decimal:
    return Decimal(str(value)).normalize()


#
# Indica si la cita trata de la variable: la nombra, o la nombra su fragmento y la cita no habla de otra.
#
def _names_variable(draft, fragment, result) -> bool:
    names = result.names
    if _names_any(draft.quote, names):
        return True

    own = set().union(*(named_variable_groups(name) for name in names))
    context = [fragment.element_caption or "", *fragment.section[-2:], fragment.text]
    quoted = f"{draft.quote} {_lead_in(draft.quote, fragment.text)}"

    return not named_variable_groups(quoted) - own and any(
        _names_any(place, names) for place in context
    )


#
# Devuelve lo que introduce a la cita dentro de su oración: ahí se nombra de qué variable habla.
#
def _lead_in(quote, text) -> str:
    wanted = normalized_text(quote)
    for item in sentences(text):
        plain = normalized_text(item)
        if wanted in plain:
            return plain[: plain.index(wanted)]

    return ""


#
# Indica si un texto contiene alguno de los nombres de una variable.
#
def _names_any(text, names) -> bool:
    return any(contains_phrase(text, name, plural_insensitive=True) for name in names)


#
# Construye la referencia con su estado de compatibilidad.
#
def _reference(draft, fragment, result, asked, places) -> ReferenceValue:
    context = " ".join([*fragment.section, fragment.element_caption or "", fragment.title or ""])
    around = f"{draft.quote} {context}"
    uses = _value_uses(draft, fragment, context)
    status, detail = _compatibility(draft, result, uses, asked)

    return ReferenceValue(
        evidence_id=fragment.evidence_id,
        result_field=result.field,
        variable=result.variable,
        kind=_stated_kind(draft, fragment),
        low=draft.low,
        high=draft.high,
        unit=draft.unit,
        basis=concentration_basis(draft.quote, draft.unit),
        water_use=next(iter(uses)) if len(uses) == 1 else None,
        condition=_stated_condition(draft, fragment),
        same_place=any(_reports_place(draft, fragment, place, around) for place in places),
        quote=draft.quote,
        status=status,
        detail=detail,
    )


#
# Devuelve el uso del agua del valor: el de su cita, el de su oración o, si no lo dicen, el de su sección;
# cuando el texto más cercano nombra varios usos, vale el que propuso el modelo sólo si es uno de ellos.
#
def _value_uses(draft, fragment, context) -> set[str]:
    proposed = _proposed_uses(draft.water_use)
    sentence = f"{_lead_in(draft.quote, fragment.text)} {draft.quote}"
    for named in (water_uses(draft.quote), water_uses(sentence), water_uses(context)):
        chosen = named & proposed
        if named:
            return chosen if len(chosen) == 1 else named

    return set()


#
# Devuelve el uso del agua que propuso el modelo, leído con el vocabulario cerrado.
#
def _proposed_uses(text) -> set[str]:
    plain = normalized_text(text or "")

    return {use for use, stems in _USE_STEMS.items() if any(stem in plain for stem in stems)}


#
# Devuelve la condición en que rige el valor (nivel, alerta, probabilidad de efectos, organismo), sólo si el
# fragmento la escribe tal cual y no es la misma cita del número.
#
def _stated_condition(draft, fragment) -> str | None:
    condition = " ".join((draft.condition or "").split()).strip(" .,;:")
    source = " ".join([fragment.text, fragment.element_caption or "", *fragment.section])
    if len(condition.split()) < _MIN_CONDITION_WORDS:
        return None
    if normalized_text(condition) not in normalized_text(source):
        return None

    bounds = {_decimal(item) for item in (draft.low, draft.high) if item is not None}
    if any(reading.values & bounds for reading in number_readings(condition)):
        return None

    return condition[:_MAX_CONDITION_CHARS]


#
# Indica si la cita es del lugar: su contexto lo nombra y, si la tabla tiene códigos de sitio, está en sus filas.
#
def _reports_place(draft, fragment, place, around) -> bool:
    if not contains_phrase(around, place):
        return False

    source = f"{fragment.element_caption or ''}\n{fragment.text}"
    own = rows_of_place(source, place)

    return own == source or normalized_text(draft.quote) in normalized_text(own)


#
# Devuelve el tipo de referencia: una guía o un límite tienen que decirse como tales en el texto.
#
def _stated_kind(draft, fragment) -> str:
    if draft.kind not in _LINE_KINDS:
        return draft.kind

    around = " ".join([draft.quote, *fragment.section[-2:], fragment.element_caption or ""])

    return draft.kind if GUIDELINE_CUES.search(around) else "observed"


#
# Decide si un valor se puede contrastar con el resultado, en orden de requisitos.
#
def _compatibility(draft, result, uses, asked):
    status, detail = _unit_compatibility(draft, result)
    if status != "compatible":
        return status, detail
    if len(uses) > 1 or (asked and not asked & uses):
        return "other_use", ", ".join(sorted(uses)) or None

    return status, detail


#
# Decide si la unidad y la base química del valor coinciden con las del resultado.
#
def _unit_compatibility(draft, result):
    if result.unit_status == "dimensionless":
        return ("compatible", None) if draft.unit is None else ("other_unit", draft.unit)
    if result.lacks_unit:
        return "unit_unknown", None
    if draft.unit is None:
        return "other_unit", None

    own = comparable_unit(result.column, result.unit)
    other = comparable_unit(draft.quote, draft.unit)
    if own[0] != other[0]:
        return "other_unit", f"{draft.unit} / {result.unit}"
    if own[1] != other[1] or (requires_chemical_basis(result.variable) and own[1] is None):
        return "other_basis", f"{other[1]} / {own[1]}"

    return "compatible", None


#
# Devuelve la identidad de una referencia, para no repetirla.
#
def _identity(reference):
    return reference.result_field, reference.low, reference.high, reference.unit, reference.water_use


#
# Devuelve los contrastes verificados entre los resultados y sus referencias compatibles.
#
def comparison_facts(references, results, language) -> list[ComparisonFact]:
    by_field = {item.field: item for item in results}
    comparable = most_specified(references)
    ambiguous = {id(item) for group in _unconditioned_groups(comparable) for item in group}
    facts = []
    for reference in comparable:
        result = by_field.get(reference.result_field)
        if id(reference) not in ambiguous and result is not None and result.values:
            facts.append(_comparison(reference, result, language))

    return facts


#
# Devuelve las referencias comparables sin repetir un mismo valor citado por varios documentos: se queda con
# la que dice su uso del agua y su condición, y conserva los usos distintos.
#
def most_specified(references) -> list:
    comparable = [item for item in references if item.status == "compatible" and _comparable(item)]
    groups = {}
    for item in comparable:
        key = (item.result_field, item.kind, item.low, item.high, unit_key(item.unit or ""))
        groups.setdefault(key, []).append(item)

    kept = set()
    for group in groups.values():
        named = [item for item in group if item.water_use] or group
        for use in dict.fromkeys(item.water_use for item in named):
            same = [item for item in named if item.water_use == use]
            kept.add(id(next((item for item in same if item.condition), same[0])))

    return [item for item in comparable if id(item) in kept]


#
# Devuelve los grupos de guías o límites del mismo tipo, unidad y uso con valores distintos en los que alguno
# no dice su condición: presentarlos como un mismo valor guía confundiría criterios distintos.
#
def _unconditioned_groups(references) -> list[list]:
    groups = {}
    for item in references:
        if item.kind in _COMPARABLE_KINDS:
            key = (item.result_field, item.kind, unit_key(item.unit or ""), item.water_use)
            groups.setdefault(key, []).append(item)

    return [
        group
        for group in groups.values()
        if len({(item.low, item.high) for item in group}) > 1 and any(item.condition is None for item in group)
    ]


#
# Explica qué valores de referencia no se compararon por ser varios sin una condición verificable.
#
def ambiguity_notes(references, results, language) -> list[str]:
    comparable = most_specified(references)
    labels = {item.field: item.variable for item in results}
    notes = []
    for group in _unconditioned_groups(comparable):
        ordered = sorted(group, key=lambda item: item.low if item.low is not None else item.high)
        values = join_words(list(dict.fromkeys(_bounds_text(item, language) for item in ordered)), language)
        unit = f" {format_unit(group[0].unit)}" if group[0].unit else ""
        variable = labels.get(group[0].result_field, group[0].variable)
        notes.append(
            f"The documents give several reference values for {variable} ({values}{unit}) and I could not "
            "verify which condition each one applies to, so I did not compare the result with them."
            if language == "en"
            else f"Los documentos dan varios valores de referencia para {variable} ({values}{unit}) y no pude "
            "verificar a qué condición corresponde cada uno, así que no comparé el resultado con ellos."
        )

    return notes


#
# Indica si una referencia sirve de contraste: un valor observado sólo vale para su mismo lugar.
#
def _comparable(reference) -> bool:
    return reference.kind in _COMPARABLE_KINDS or reference.same_place


#
# Construye el contraste de un resultado con una referencia.
#
def _comparison(reference, result, language) -> ComparisonFact:
    text = (
        _single_value(reference, result, language)
        if len(result.values) == 1
        else _several_values(reference, result, language)
    )

    direction = _position(result.values[0], reference) if len(result.values) == 1 else "mixed"

    return ComparisonFact(
        text=text,
        evidence_id=result.evidence_id,
        document_id=reference.evidence_id,
        direction=direction,
    )


#
# Redacta el contraste de un único valor.
#
def _single_value(reference, result, language) -> str:
    unit = f" {format_unit(result.unit)}" if result.unit else ""
    value = f"{shown_value(result.values[0], language)}{unit}"
    position = _position(result.values[0], reference)
    target = _reference_text(reference, language, result.unit)
    if language == "en":
        words = {"below": "is below", "above": "is above", "within": "is within"}
        return f"{result.label} ({value}) {words[position]} {target}."

    words = {"below": "está por debajo de", "above": "está por encima de", "within": "está dentro de"}

    return _contracted(f"{result.label} ({value}) {words[position]} {target}.")


#
# Redacta el contraste de varios valores contando los que quedan por encima.
#
def _several_values(reference, result, language) -> str:
    above = sum(_position(value, reference) == "above" for value in result.values)
    total = len(result.values)
    target = _reference_text(reference, language, result.unit)
    if language == "en":
        if not above:
            return f"None of the {total} values of {result.label} is above {target}."
        return f"{above} of {total} values of {result.label} are above {target}."
    if not above:
        return _contracted(f"Ninguno de los {total} valores de {result.label} está por encima de {target}.")

    return _contracted(f"{above} de {total} valores de {result.label} están por encima de {target}.")


#
# Contrae «de el» y «a el» como corresponde en español.
#
def _contracted(text) -> str:
    return _CONTRACTIONS.sub(lambda match: "del " if match.group(1).lower() == "de" else "al ", text)


#
# Devuelve si un valor queda por debajo, dentro o por encima de una referencia.
#
def _position(value, reference) -> str:
    low = _bound(reference.low, reference.high)
    high = _bound(reference.high, reference.low)
    if value > high:
        return "above"
    if value < low:
        return "below"

    return "within"


#
# Devuelve el extremo de una referencia, o el otro si es un valor único.
#
def _bound(value, other) -> Decimal:
    return Decimal(str(value if value is not None else other))


#
# Devuelve la referencia en palabras: tipo, valor, unidad y uso del agua. Un contraste compatible
# usa la grafía de la unidad del resultado.
#
def _reference_text(reference, language, result_unit=None) -> str:
    kind = _KIND_LABELS[language][reference.kind]
    written = result_unit or reference.unit
    unit = f" {format_unit(written)}" if written else ""
    details = [item for item in (_USE_LABELS[language].get(reference.water_use), reference.condition) if item]
    suffix = f" ({'; '.join(details)})" if details else ""

    joiner = {"es": "de", "en": "of"}[language] if reference.kind != "observed" else ":"

    return f"{kind} {joiner} {_bounds_text(reference, language)}{unit}{suffix}".replace(" :", ":")


#
# Devuelve los extremos de una referencia, o su único valor.
#
def _bounds_text(reference, language) -> str:
    low, high = reference.low, reference.high
    if low is None or high is None or low == high:
        return shown_value(Decimal(str(low if low is not None else high)), language)

    return f"{shown_value(Decimal(str(low)), language)}-{shown_value(Decimal(str(high)), language)}"


#
# Explica qué guías o límites documentados no se compararon por estar en otra unidad o base química.
#
def incompatibility_notes(references, language) -> list[str]:
    units = sorted(
        {
            format_unit(item.unit)
            for item in references
            if item.unit and item.kind in _COMPARABLE_KINDS and item.status in _INCOMPARABLE
        }
    )
    if not units:
        return []

    named = join_words(units, language)
    if language == "en":
        return [
            f"I did not compare this result with the documented reference values: they are in {named}, "
            "a different unit or chemical basis, and I did not convert them."
        ]

    return [
        f"No comparé este resultado con los valores de referencia documentados: están en {named}, otra "
        "unidad o base química, y no los convertí."
    ]


#
# Devuelve las guías y límites verificados que sirven de umbral: compatibles y de un único valor.
#
def threshold_references(references) -> list[ReferenceValue]:
    return [
        item
        for item in references
        if item.status == "compatible" and item.kind in _LINE_KINDS and _single_bound(item) is not None
    ]


#
# Devuelve el único valor de una referencia que sirve de umbral.
#
def threshold_value(reference) -> Decimal:
    return _decimal(_single_bound(reference))


#
# Describe un valor de referencia con su unidad, su uso y su condición.
#
def reference_option(reference, language) -> str:
    unit = f" {format_unit(reference.unit)}" if reference.unit else ""
    details = [item for item in (_USE_LABELS[language].get(reference.water_use), reference.condition) if item]
    suffix = f" ({'; '.join(details)})" if details else ""

    return f"{_bounds_text(reference, language)}{unit}{suffix}"


#
# Devuelve las líneas de referencia verificadas para los gráficos y las notas de lo que no se dibujó.
#
def chart_lines(state, references, results, question, language):
    charts = state.python_execution.visualizations if state.python_execution else []
    if not charts or REFERENCE_LINE_CUES.search(question) is None:
        return [], []

    lines, notes = [], []
    for chart in charts:
        placed, reason = _chart_line(chart, references, question, language)
        lines.extend(placed)
        if reason:
            notes.append(reason)

    return lines, list(dict.fromkeys(notes))


#
# Elige la línea de un gráfico, o explica por qué no corresponde dibujarla.
#
def _chart_line(chart, references, question, language):
    axis, field = _measured_axis(chart)
    candidates = [
        item
        for item in references
        if item.kind in _LINE_KINDS
        and _single_bound(item) is not None
        and item.result_field == field.name
    ]
    compatible = [item for item in candidates if item.status == "compatible"]
    if not compatible:
        return [], _no_line_note(candidates, language)

    asked = water_uses(question)
    chosen = [item for item in compatible if item.water_use in asked]
    if not chosen:
        return [], _ambiguous_use_note(compatible, language)

    drawable = [item for item in chosen if not _out_of_scale(chart, _single_bound(item), axis)]
    if not drawable:
        return [], _scale_note(chosen[0], language)

    return [_line(chart, axis, field, item, language) for item in drawable[:_MAX_LINES]], None


#
# Construye la línea de una referencia verificada.
#
def _line(chart, axis, field, reference, language) -> ReferenceLine:
    label = _reference_text(reference, language)

    return ReferenceLine(
        operation_id=chart.operation_id,
        evidence_id=reference.evidence_id,
        axis=axis,
        value=_single_bound(reference),
        unit=reference.unit or (field.unit or ""),
        label=label[:1].upper() + label[1:],
    )


#
# Devuelve el eje del gráfico que representa la variable medida.
#
def _measured_axis(chart):
    if chart.chart_type == "histogram":
        return "x", chart.x_axis.field

    return "y", chart.y_axis.field


#
# Devuelve el valor de una referencia que es un único número.
#
def _single_bound(reference):
    low, high = reference.low, reference.high
    if low is None or high is None or low == high:
        return low if low is not None else high

    return None


#
# Indica si el valor quedaría tan lejos de los datos que los aplastaría en el gráfico.
#
def _out_of_scale(chart, value, axis) -> bool:
    values = [
        abs(point.y if axis == "y" else point.x)
        for series in chart.series
        for point in series.points
        if isinstance(point.y if axis == "y" else point.x, (int, float))
    ]
    if chart.chart_type == "histogram":
        bins = [item for series in chart.series for item in series.bins]
        if all(item.lower == item.upper for item in bins):
            return True
        values = [abs(item.upper) for item in bins]

    return bool(values) and abs(value) > _MAX_SCALE_GAP * max(values)


#
# Explica que no hay un valor compatible para dibujar.
#
def _no_line_note(candidates, language) -> str:
    other_units = sorted({item.unit for item in candidates if item.status == "other_unit" and item.unit})
    if other_units:
        units = join_words(other_units, language)
        return (
            f"I did not add a reference line: the documented values are in {units}, a different unit "
            "from this variable, and I did not convert them."
            if language == "en"
            else f"No agregué una línea de referencia: los valores documentados están en {units}, una "
            "unidad distinta de la de esta variable, y no los convertí."
        )

    if language == "en":
        return (
            "I did not add a reference line: the documented values do not match this variable's "
            "unit, chemical basis or use."
            if candidates
            else "I did not add a reference line: I found no documented value for this variable."
        )

    return (
        "No agregué una línea de referencia: los valores documentados no coinciden con la unidad, "
        "la base química o el uso de esta variable."
        if candidates
        else "No agregué una línea de referencia: no encontré un valor documentado para esta variable."
    )


#
# Explica que hay valores para distintos usos y hace falta elegir uno.
#
def _ambiguous_use_note(references, language) -> str:
    uses = [_USE_LABELS[language].get(item.water_use) for item in references]
    named = join_words(sorted({item for item in uses if item}), language)
    if language == "en":
        return (
            "I did not add a reference line because the value depends on the water use"
            + (f" ({named})" if named else "")
            + ": tell me which one applies."
        )

    return (
        "No agregué una línea de referencia porque el valor depende del uso del agua"
        + (f" ({named})" if named else "")
        + ": indicame cuál corresponde."
    )


#
# Explica que el valor de referencia queda fuera de la escala de los datos.
#
def _scale_note(reference, language) -> str:
    target = _reference_text(reference, language)
    if language == "en":
        return f"I did not draw {target} because it is far above the charted values."

    return f"No dibujé {target} porque queda muy por encima de los valores graficados."
