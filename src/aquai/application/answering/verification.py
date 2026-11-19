import re
from dataclasses import dataclass, field

from aquai.application.answering.readings import (
    HIDDEN_VALUE,
    Reading,
    folded,
    group_values,
    number_readings,
    sentences,
    unread_tokens,
    written_nearby,
)
from aquai.application.matching import contains_phrase, named_variable_groups, normalize_text
from aquai.application.Rules.documents import language_scores
from aquai.application.Rules.evidence import (
    ABSENCE_CLAIM,
    ASSESSMENT_FAMILIES,
    ATTRIBUTED_SOURCE,
    BOUND_WORDS,
    CAUSAL_CLAIMS,
    CAUSAL_LINKS,
    CHEMICAL_NAME,
    DATE_TOKEN,
    COMPARISON_DIRECTIONS,
    CONDITIONAL_CUES,
    DEICTIC_PLACE,
    DOCUMENT_YEAR,
    EMPTY_RESULT,
    FOREIGN_SCRIPT,
    FREQUENCY_WORDS,
    FULL_DATE,
    GUIDELINE_CUES,
    HEDGE_CUES,
    INLINE_REFERENCE,
    INTERNAL_TERMS,
    MARKUP,
    NEGATION,
    OBSERVED_CLASSIFICATION,
    OBSERVED_COMPARISON,
    OBSERVED_REFERENCE,
    OBSERVED_VERDICTS,
    ORGANIZATION_NAMES,
    QUALIFIER_FAMILIES,
    RECOMMENDATIONS,
    REFERENCE_LABELS,
    RELATION_LANGUAGE,
    RELATIVE_QUANTITIES,
    RESULT_WORDS,
    SITUATED_FAMILIES,
    SOURCE_ALIASES,
    UNSUPPORTED_STATISTICS,
    UNSUPPORTED_VERDICTS,
    VERDICT_LIMITS,
    WATER_USES,
)
from aquai.application.Rules.variables import PH_ALIASES
from aquai.application.units import comparable_unit, spelled_symbol

_TABLE_CELL = "|"
_NAME_STEM = 5
_CAPITALIZED = re.compile(r"(?<![\w.°])[A-ZÁÉÍÓÚÑÜ][\w-]+")
_FIRST_WORD = re.compile(r"\w")
_NAMED_NUMBER = re.compile(r"[A-ZÁÉÍÓÚÑÜ][\w-]*\s+$")
_CLAUSE_BREAK = re.compile(r"[,;:()\n]")
_CLAUSE_WORD = 6
_PROPER_NAME = re.compile(
    r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:(?:de|del|la|las|el|los|of|the)\s+)*[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)*"
)
_HEDGE_WINDOW = 40
_BOUND_WINDOW = 40
_BOUNDS = {"<", ">", "≤", "≥"}
_LABEL_WINDOW = 8
_MIN_ROOT = 4
_FLEXION_CHARS = 2
_COUNTED_SAMPLES = re.compile(r"(?:\d[\d.,]*|\buna?)\s+(?:muestras?|samples?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RequiredValue:
    """Valor de referencia que un fragmento fija para lo preguntado y que la respuesta tiene que presentar."""

    evidence_id: str
    values: frozenset
    shown: str
    sentence: str


@dataclass
class DraftEvidence:
    """Evidencia autorizada para un borrador; un párrafo sólo puede usar lo que cita."""

    facts: dict[str, str] = field(default_factory=dict)
    fact_documents: dict[str, list[str]] = field(default_factory=dict)
    fact_directions: dict[str, str] = field(default_factory=dict)
    fact_values: dict[str, list[str]] = field(default_factory=dict)
    required_facts: set[str] = field(default_factory=set)
    documents: dict[str, str] = field(default_factory=dict)
    document_languages: dict[str, str | None] = field(default_factory=dict)
    doubtful_documents: set[str] = field(default_factory=set)
    document_labels: dict[str, str] = field(default_factory=dict)
    document_sources: dict[str, str] = field(default_factory=dict)
    document_owners: dict[str, str] = field(default_factory=dict)
    hypothesis_values: list[str] = field(default_factory=list)
    question: str = ""
    result_names: list[str] = field(default_factory=list)
    previous_answers: list[str] = field(default_factory=list)
    allows_hypotheses: bool = False
    documents_required: bool = False
    required_values: list[RequiredValue] = field(default_factory=list)
    language: str = "es"
    places: list[str] = field(default_factory=list)
    document_contexts: dict[str, str] = field(default_factory=dict)
    variable_names: list[str] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Rejection:
    code: str
    detail: str
    family: str | None = None


@dataclass
class _Authorized:
    """Números, unidades y comparadores que un párrafo puede usar."""

    values: set = field(default_factory=set)
    pairs: set = field(default_factory=set)
    comparators: set = field(default_factory=set)
    unitless: set = field(default_factory=set)
    bounded: set = field(default_factory=set)
    exact: set = field(default_factory=set)
    text: str = ""

    #
    # Agrega lo que autoriza un texto, leído según su idioma.
    #
    def add(self, text: str, language: str | None) -> None:
        self.text += f" {folded(text)}"
        for reading in number_readings(text, language):
            self.values |= reading.values | group_values(reading.text)
            if reading.unit:
                unit = comparable_unit("", reading.unit)
                self.pairs |= {(value, unit) for value in reading.values}
            if reading.comparator:
                self.comparators |= {(value, reading.comparator) for value in reading.values}
            if reading.comparator in _BOUNDS:
                self.bounded |= reading.values
            else:
                self.exact |= reading.values

    #
    # Agrega un hecho: sus números sin unidad no pueden tomar la de un fragmento.
    #
    def add_fact(self, text: str, language: str) -> None:
        self.add(text, language)
        readings = number_readings(text, language)
        bare = {value for item in readings if not item.unit for value in item.values}
        measured = {value for item in readings if item.unit for value in item.values}
        self.unitless |= bare - measured


class DraftVerifier:
    #
    # Guarda la evidencia autorizada contra la que se revisan los segmentos del borrador.
    #
    def __init__(self, evidence: DraftEvidence):
        self._evidence = evidence

    #
    # Devuelve los motivos por los que un segmento no se puede publicar.
    #
    def check_segment(self, segment) -> list[Rejection]:
        structural = self._structure(segment)
        if structural:
            return structural

        text = _full_text(segment)
        unnamed = _without_source_names(text, self._evidence.source_names)

        return [
            *self._form(text),
            *self._numbers(segment, unnamed),
            *self._sources(segment, text),
            *self._result_wording(segment, unnamed),
            *self._verdicts(segment, text),
            *self._comparisons(segment, text),
            *self._statistics(segment, text),
            *self._places(segment, text),
            *self._attributions(segment, text),
            *self._frequencies(segment, text),
            *self._reference_labels(segment, text),
            *self._conditions(segment, text),
        ]

    #
    # Devuelve las oraciones obligatorias cuyo valor no quedó en los párrafos de resultado.
    #
    def missing_facts(self, segments) -> list[str]:
        results = [segment for segment in segments if segment.role == "result"]
        cited = {key for segment in results for key in segment.fact_ids}
        published = "\n".join(_full_text(segment) for segment in results)
        shown = {
            value
            for reading in number_readings(published, self._evidence.language)
            for value in reading.values
        }

        return [
            key
            for key in self._evidence.required_facts
            if key not in cited or not self._shows_values(key, shown)
        ]

    #
    # Devuelve los valores de referencia documentados que ningún párrafo documental llegó a presentar.
    #
    def missing_values(self, segments) -> list[RequiredValue]:
        shown = {
            value
            for segment in segments
            if segment.role in {"documented", "relation"}
            for reading in number_readings(_full_text(segment), self._evidence.language)
            for value in reading.values
        }

        return [item for item in self._evidence.required_values if not item.values & shown]

    #
    # Indica si los valores de una oración obligatoria figuran en el texto publicado.
    #
    def _shows_values(self, key, shown):
        text = FULL_DATE.sub(r"\1", self._evidence.facts[key])
        readings = number_readings(text, self._evidence.language)

        return all(reading.values & shown for reading in readings)

    #
    # Revisa que el texto sea prosa limpia en el idioma de la respuesta.
    #
    def _form(self, text) -> list[Rejection]:
        rejections = []
        if HIDDEN_VALUE in text:
            rejections.append(Rejection("hidden_value", HIDDEN_VALUE))

        for code, pattern in (
            ("markup", MARKUP),
            ("foreign_script", FOREIGN_SCRIPT),
            ("inline_reference", INLINE_REFERENCE),
            ("internal_term", INTERNAL_TERMS),
        ):
            found = pattern.search(text)
            if found:
                rejections.append(Rejection(code, found.group()))

        scores = language_scores(text)
        other = "en" if self._evidence.language == "es" else "es"
        if scores.get(other, 0) > 2 * max(1, scores.get(self._evidence.language, 0)):
            rejections.append(Rejection("wrong_language", text[:60]))

        return rejections

    #
    # Revisa que el segmento cite evidencia existente, coherente con su función.
    #
    def _structure(self, segment) -> list[Rejection]:
        evidence = self._evidence
        unknown = [
            *(key for key in segment.fact_ids if key not in evidence.facts),
            *(key for key in segment.document_ids if key not in evidence.documents),
        ]
        if unknown:
            return [Rejection("unknown_evidence", ", ".join(unknown))]
        if not segment.text.strip():
            return [Rejection("empty_segment", segment.role)]

        if segment.role == "result" and not segment.fact_ids:
            return [Rejection("result_without_fact", segment.text[:80])]
        if segment.role in {"documented", "relation", "hypothesis"} and not segment.document_ids:
            return [Rejection("claim_without_document", segment.text[:80])]
        if segment.role == "hypothesis" and not evidence.allows_hypotheses:
            return [Rejection("hypothesis_without_ambiguity", segment.text[:80])]

        return []

    #
    # Revisa que cada oración con cifras de una hipótesis esté en condicional.
    #
    def _conditions(self, segment, text) -> list[Rejection]:
        if segment.role != "hypothesis":
            return []

        language = self._evidence.language

        return [
            Rejection("hypothesis_not_conditional", sentence[:80])
            for sentence in sentences(text)
            if number_readings(sentence, language) and not _conditional_first(sentence)
        ]

    #
    # Revisa que cada número y su unidad provengan de la evidencia que el segmento cita.
    #
    def _numbers(self, segment, text) -> list[Rejection]:
        allowed = self._allowed_for(segment)
        fragments = self._usable_documents(segment)
        readings = number_readings(text, self._evidence.language)
        rejections = [
            problem
            for reading in readings
            for problem in self._reading_problems(reading, allowed, fragments, text)
        ]
        unread = [
            Rejection("unread_number", token)
            for token in unread_tokens(text, readings)
            if folded(token) not in allowed.text and not CHEMICAL_NAME.fullmatch(token)
        ]

        return [*rejections, *unread]

    #
    # Devuelve los problemas de un número leído del borrador.
    #
    def _reading_problems(self, reading, allowed, fragments, text) -> list[Rejection]:
        if not reading.values & allowed.values:
            return [Rejection("unsupported_number", reading.text)]
        if reading.comparator and not self._comparator_allowed(reading, allowed):
            return [Rejection("unsupported_comparator", f"{reading.comparator}{reading.text}")]
        if _drops_bound(reading, allowed, text):
            return [Rejection("dropped_bound", reading.text)]
        if reading.spelled_unit and not self._spelled_unit_allowed(reading, allowed, fragments):
            return [Rejection("unsupported_unit", f"{reading.text} {reading.spelled_unit}")]
        unknown = reading.unit and reading.values & allowed.unitless
        if unknown or (reading.unit and not self._unit_matches(reading, allowed, fragments)):
            return [Rejection("unsupported_unit", f"{reading.text} {reading.unit}")]

        return []

    #
    # Indica si la evidencia escribe ese número con el mismo comparador.
    #
    @staticmethod
    def _comparator_allowed(reading, allowed) -> bool:
        return any((value, reading.comparator) in allowed.comparators for value in reading.values)

    #
    # Indica si una unidad escrita en palabras figura así en la evidencia o es la que acompaña al número allí.
    #
    def _spelled_unit_allowed(self, reading, allowed, fragments) -> bool:
        if folded(reading.spelled_unit) in allowed.text:
            return True

        symbol = spelled_symbol(reading.spelled_unit)
        written = Reading(reading.text, reading.values, symbol, start=reading.start, end=reading.end)

        return symbol is not None and self._unit_matches(written, allowed, fragments)

    #
    # Revisa que las cifras de cada oración documental salgan de un mismo documento, sin mezclar fuentes: los
    # fragmentos de un mismo documento (dos secciones de la misma guía) son una sola fuente.
    #
    def _sources(self, segment, text) -> list[Rejection]:
        evidence = self._evidence
        cited = [key for key in segment.document_ids if key not in evidence.doubtful_documents]
        owners = {}
        for key in cited:
            owners.setdefault(evidence.document_owners.get(key, key), []).append(key)
        if segment.role != "documented" or len(owners) < 2:
            return []

        shared = _Authorized()
        for key in segment.fact_ids:
            shared.add_fact(evidence.facts[key], evidence.language)
        for answer in evidence.previous_answers:
            shared.add(answer, evidence.language)
        holders = [set().union(*(self._document_values(key) for key in keys)) for keys in owners.values()]

        return [
            Rejection("mixed_sources", sentence[:80])
            for sentence in sentences(text)
            if not _single_source(sentence, holders, shared.values, evidence.language)
        ]

    #
    # Devuelve los números que aporta un fragmento, con los años de su cita.
    #
    def _document_values(self, key) -> set:
        evidence = self._evidence
        allowed = _Authorized()
        allowed.add(evidence.documents[key], evidence.document_languages.get(key))
        allowed.add(" ".join(DOCUMENT_YEAR.findall(evidence.document_labels.get(key, ""))), None)

        return allowed.values

    #
    # Devuelve los textos de los fragmentos confiables que aportan números al segmento.
    #
    def _usable_documents(self, segment) -> list[str]:
        evidence = self._evidence
        if segment.role in {"result", "relation"}:
            return []

        return [
            evidence.documents[key]
            for key in segment.document_ids
            if key not in evidence.doubtful_documents
        ]

    #
    # Devuelve lo que el segmento puede usar: sólo lo que cita, nunca la pregunta.
    #
    def _allowed_for(self, segment) -> _Authorized:
        evidence = self._evidence
        allowed = _Authorized()
        for key in segment.fact_ids:
            allowed.add_fact(evidence.facts[key], evidence.language)
            for value in evidence.fact_values.get(key, []):
                allowed.add(value, "en")
        for key in segment.document_ids:
            allowed.add(" ".join(DOCUMENT_YEAR.findall(evidence.document_labels.get(key, ""))), None)
            if segment.role in {"documented", "hypothesis"} and key not in evidence.doubtful_documents:
                allowed.add(evidence.documents[key], evidence.document_languages.get(key))
        if segment.role == "documented":
            for text in evidence.previous_answers:
                allowed.add(text, evidence.language)
        if segment.role == "hypothesis":
            allowed.unitless.clear()
            for text in evidence.hypothesis_values:
                allowed.add(text, evidence.language)

        return allowed

    #
    # Indica si el número aparece en la evidencia con esa misma unidad.
    #
    def _unit_matches(self, reading, allowed, fragments):
        unit = comparable_unit("", reading.unit)
        if any((value, unit) in allowed.pairs for value in reading.values):
            return True

        sources = fragments if unit[1] is None else [item for item in fragments if _TABLE_CELL in item]

        return any(written_nearby(reading, source) for source in sources)

    #
    # Revisa que un párrafo de resultado no agregue lo que sus hechos no dicen y deje cada cifra junto a lo que
    # mide. La redacción es libre: sólo se rechazan las valoraciones de lo observado que los hechos citados no
    # hacen, las variables y los nombres propios que no figuran en lo citado ni en la pregunta.
    #
    def _result_wording(self, segment, text) -> list[Rejection]:
        if segment.role != "result":
            return []

        cited = [self._evidence.facts[key] for key in segment.fact_ids]
        rivals = [self._evidence.facts[key] for key in self._evidence.required_facts]
        named = " ".join([*cited, self._evidence.question, *self._evidence.result_names])
        words = _without_numbers(text)
        unlicensed = [
            *_unlicensed_assessments(words, " ".join(cited)),
            *_unlicensed_variables(words, named),
            *_unlicensed_names(words, named),
        ]
        rejections = [Rejection("unsupported_result_word", word) for word in dict.fromkeys(unlicensed)]
        absent = ABSENCE_CLAIM.search(text)
        if absent and not any(EMPTY_RESULT.search(fact) for fact in cited):
            rejections.append(Rejection("unsupported_absence", absent.group()))
        rejections += self._unsupported_samples(text, cited)

        return [*rejections, *self._misaligned(text, list(dict.fromkeys([*cited, *rivals])))]

    #
    # Rechaza llamar muestras a lo que los hechos cuentan como registros o fechas: una fila por taxón o por
    # parámetro no es una muestra, aunque la pregunta diga «muestras».
    #
    @staticmethod
    def _unsupported_samples(text, cited) -> list[Rejection]:
        if any(_COUNTED_SAMPLES.search(fact) for fact in cited):
            return []

        return [Rejection("unsupported_count_noun", match.group()) for match in _COUNTED_SAMPLES.finditer(text)]

    #
    # Revisa que cada número de un resultado acompañe a la variable o el lugar que le corresponde. Las partes de
    # una fecha («01/09/2021») no son valores con etiqueta: su presencia en los hechos ya la revisa _numbers.
    #
    def _misaligned(self, text, facts) -> list[Rejection]:
        language = self._evidence.language
        labels = {fact: _value_labels(fact, language) for fact in facts}
        rejections = []
        for sentence in sentences(text):
            position = 0
            dates = [match.span() for match in DATE_TOKEN.finditer(sentence)]
            for reading in number_readings(sentence, language):
                dated = any(start <= reading.start < end for start, end in dates)
                if not dated and not _aligned(sentence[position : reading.start], reading, labels):
                    rejections.append(Rejection("misaligned_value", reading.text))
                position = reading.stop

        return rejections

    #
    # Revisa que el texto no emita veredictos ni atribuya causas a lo observado.
    #
    def _verdicts(self, segment, text) -> list[Rejection]:
        observed = _speaks_of_results(segment)
        rejections = []
        for sentence in sentences(text):
            rejections.extend(self._sentence_verdicts(segment, sentence, observed))

        return rejections

    #
    # Devuelve los veredictos, clasificaciones y causas sin respaldo de una oración.
    #
    def _sentence_verdicts(self, segment, sentence, observed) -> list[Rejection]:
        rejections = []
        verdict = UNSUPPORTED_VERDICTS.search(sentence)
        if verdict and not VERDICT_LIMITS.search(sentence[: verdict.start()]):
            rejections.append(Rejection("unsupported_verdict", verdict.group()))

        if segment.role == "documented" and self._evidence.facts:
            mention = OBSERVED_REFERENCE.search(sentence)
            if mention:
                rejections.append(Rejection("observed_in_documented", mention.group()))

        if not observed or segment.role == "hypothesis":
            return rejections

        for code, pattern in (
            ("unsupported_verdict", OBSERVED_VERDICTS),
            ("unsupported_classification", OBSERVED_CLASSIFICATION),
            ("unsupported_quantity", RELATIVE_QUANTITIES),
        ):
            found = pattern.search(sentence)
            limited = found and VERDICT_LIMITS.search(sentence[: found.start()])
            quoted = found and (self._quotes_facts(segment, found.group()) or self._names_variable(sentence, found))
            if found and not limited and not quoted:
                rejections.append(Rejection(code, found.group()))

        cause = CAUSAL_CLAIMS.search(sentence)
        if cause and not _hedged(sentence, cause.start()):
            rejections.append(Rejection("unsupported_causality", cause.group()))

        advice = RECOMMENDATIONS.search(sentence)
        if advice:
            rejections.append(Rejection("unsupported_recommendation", advice.group()))

        return rejections

    #
    # Indica si la palabra hallada es parte del nombre de una variable pedida que la oración escribe con la palabra
    # que la precede («cianobacterias potencialmente tóxicas»): nombrar la variable no es juzgar lo observado.
    #
    def _names_variable(self, sentence, found) -> bool:
        before = normalize_text(sentence[: found.start()]).split()[-1:]
        phrase = " ".join([*before, *normalize_text(found.group()).split()])

        return len(phrase.split()) > 1 and any(contains_phrase(name, phrase) for name in self._evidence.variable_names)

    #
    # Indica si unas palabras repiten lo que dicen los hechos citados, como la condición de un valor guía.
    #
    def _quotes_facts(self, segment, words) -> bool:
        cited = normalize_text(" ".join(self._evidence.facts[key] for key in segment.fact_ids)).split()
        roots = {_root(word) for word in cited}

        return all(_root(word) in roots for word in normalize_text(words).split())

    #
    # Revisa que una comparación de lo observado salga de un contraste y respete su sentido.
    #
    def _comparisons(self, segment, text) -> list[Rejection]:
        if not _speaks_of_results(segment) or segment.role == "hypothesis":
            return []

        directions = [
            self._evidence.fact_directions[key]
            for key in segment.fact_ids
            if key in self._evidence.fact_directions
        ]
        if segment.role in {"result", "relation"}:
            directions += self._filtered_directions(segment)
        rejections = []
        for sentence in sentences(text):
            compared = OBSERVED_COMPARISON.search(sentence)
            if compared is None:
                continue
            if directions:
                if not _direction_agrees(sentence, directions):
                    rejections.append(Rejection("wrong_direction", compared.group()))
                continue

            ordered = _ordered_values(sentence, compared, self._evidence.language) if segment.role == "result" else None
            if ordered is None:
                rejections.append(Rejection("unsupported_comparison", compared.group()))
            elif not ordered:
                rejections.append(Rejection("wrong_direction", compared.group()))

        return rejections

    #
    # Devuelve los sentidos de comparación que ya dicen los hechos citados como filtro («mayor que 20.000»):
    # un resultado puede repetirlos con otras palabras sin comparar nada nuevo.
    #
    def _filtered_directions(self, segment) -> list[str]:
        cited = " ".join(self._evidence.facts[key] for key in segment.fact_ids)

        return [name for name, pattern in COMPARISON_DIRECTIONS.items() if pattern.search(cited)]

    #
    # Revisa que un párrafo documental sólo atribuya algo a un lugar si sus fragmentos lo nombran.
    #
    def _places(self, segment, text) -> list[Rejection]:
        if not segment.document_ids:
            return []

        evidence = self._evidence
        contexts = [evidence.document_contexts.get(key, "") for key in segment.document_ids]
        facts = [evidence.facts[key] for key in segment.fact_ids]
        rejections = []
        for place in evidence.places:
            naming = [*facts, *(item for item in contexts if contains_phrase(item, place))]
            for sentence in sentences(text):
                if contains_phrase(sentence, place) or DEICTIC_PLACE.search(sentence):
                    rejections.extend(self._place_values(sentence, place, naming))

        return rejections

    #
    # Revisa que los números de una oración sobre un lugar salgan de fragmentos que lo nombran.
    #
    def _place_values(self, sentence, place, naming) -> list[Rejection]:
        if not naming and contains_phrase(sentence, place):
            return [Rejection("unsupported_place", place)]

        allowed = _Authorized()
        for text in naming:
            allowed.add(rows_of_place(text, place), None)

        return [
            Rejection("unsupported_place", f"{reading.text} ({place})")
            for reading in number_readings(sentence, self._evidence.language)
            if not reading.values & allowed.values
        ]

    #
    # Revisa que un párrafo sólo atribuya algo a una fuente que figure en lo que cita.
    #
    def _attributions(self, segment, text) -> list[Rejection]:
        evidence = self._evidence
        facts = [evidence.facts[key] for key in segment.fact_ids]
        rejections = []
        for sentence in sentences(text):
            names = _named_sources(sentence)
            for name in names:
                naming = self._naming_documents(segment, name, names)
                if not naming:
                    rejections.append(Rejection("unsupported_attribution", name))
                    continue

                texts = [self._source_text(key) for key in naming if key not in evidence.doubtful_documents]
                rejections.extend(self._attributed_values(sentence, name, [*texts, *facts]))

        return rejections

    #
    # Devuelve los documentos citados que son esa fuente, o que la citan si la oración también los nombra a ellos.
    #
    def _naming_documents(self, segment, name, names) -> list[str]:
        evidence = self._evidence
        own = [key for key in segment.document_ids if _names_source(name, self._metadata(key))]
        if own:
            return own

        others = [item for item in names if item != name]

        return [
            key
            for key in segment.document_ids
            if _names_source(name, self._source_text(key))
            and any(_names_source(item, self._metadata(key)) for item in others)
        ]

    #
    # Devuelve cómo se identifica un documento: su cita, su título y su archivo.
    #
    def _metadata(self, key) -> str:
        evidence = self._evidence

        return evidence.document_sources.get(key) or evidence.document_labels.get(key, "")

    #
    # Devuelve la cita de un documento junto con su texto, su sección y su leyenda.
    #
    def _source_text(self, key) -> str:
        evidence = self._evidence
        context = evidence.document_contexts.get(key, evidence.documents[key])

        return f"{evidence.document_labels.get(key, '')} {context}"

    #
    # Revisa que los números atribuidos a una fuente salgan de fragmentos que la nombran.
    #
    def _attributed_values(self, sentence, name, texts) -> list[Rejection]:
        allowed = _Authorized()
        for text in texts:
            allowed.add(text, None)

        return [
            Rejection("unsupported_attribution", f"{reading.text} ({name})")
            for reading in number_readings(sentence, self._evidence.language)
            if not reading.values & allowed.values
            or (reading.unit and not self._unit_matches(reading, allowed, texts))
        ]

    #
    # Revisa que un párrafo no agregue frecuencias o generalidades que sus fragmentos no expresan.
    #
    def _frequencies(self, segment, text) -> list[Rejection]:
        cited = " ".join(self._evidence.documents[key] for key in segment.document_ids)
        found = FREQUENCY_WORDS.search(text)
        if segment.role == "result" or found is None or FREQUENCY_WORDS.search(cited):
            return []

        return [Rejection("unsupported_frequency", found.group())]

    #
    # Revisa que no se llame valor de referencia, guía, umbral o límite a lo que los fragmentos citados sólo
    # reportan como medición de un estudio: el promedio de un sitio no es una norma.
    #
    def _reference_labels(self, segment, text) -> list[Rejection]:
        evidence = self._evidence
        found = REFERENCE_LABELS.search(text)
        if segment.role not in {"documented", "relation"} or found is None:
            return []

        cited = [evidence.documents[key] for key in segment.document_ids]
        cited += [evidence.facts[key] for key in segment.fact_ids]
        if any(GUIDELINE_CUES.search(item) for item in cited):
            return []

        return [Rejection("unsupported_reference_label", found.group())]

    #
    # Revisa que el texto no agregue conclusiones estadísticas ausentes de lo que cita.
    #
    def _statistics(self, segment, text) -> list[Rejection]:
        evidence = self._evidence
        cited = [evidence.facts[key] for key in segment.fact_ids]
        cited += [evidence.documents[key] for key in segment.document_ids]
        found = UNSUPPORTED_STATISTICS.search(text)
        if found and not UNSUPPORTED_STATISTICS.search(" ".join(cited)):
            return [Rejection("unsupported_statistic", found.group())]

        return []


#
# Devuelve el texto completo de un segmento con sus viñetas.
#
def _full_text(segment) -> str:
    return "\n".join([segment.text, *segment.items])


#
# Devuelve el texto sin los nombres de los archivos y hojas del análisis: sus dígitos («5MAR2024 (2)») no son cifras
# ni sus mayúsculas, nombres de lugares.
#
def _without_source_names(text, names) -> str:
    for name in names:
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)

    return text


#
# Devuelve lo que una paráfrasis pierde o agrega respecto de la oración del fragmento que más se le parece:
# una acotación (atenuación, alcance del estudio, condición, cuantificador, ambiente o uso del agua), una
# relación causal, una generalidad o una conclusión estadística que el fragmento no expresa. Un párrafo que
# relaciona lo observado con un valor documentado sólo debe conservar el ambiente y el uso del agua.
#
def paraphrase_drift(
    claim, passage, heading="", earlier="", question="", relation=False, same_language=False
) -> Rejection | None:
    situated = f"{passage} {heading}"
    for family, patterns in QUALIFIER_FAMILIES.items():
        if relation and family not in SITUATED_FAMILIES:
            continue

        source = situated if family in SITUATED_FAMILIES else passage
        kept = _kept_text(family, claim, earlier, question)
        found = [
            item
            for item in patterns[0].finditer(source)
            if not (same_language and _qualifies_another_clause(item, passage, claim))
            and not _names_qualified_place(item, passage, heading, f"{earlier} {claim} {question}")
        ]
        if found and not patterns[-1].search(kept):
            return Rejection("lost_qualifier", f"{family} «{found[0].group()}»: {claim[:70]}", family)

    for use in _water_uses(situated) - _water_uses(f"{earlier} {claim} {question}"):
        return Rejection("lost_qualifier", f"uso del agua «{use}»: {claim[:70]}")

    if relation:
        return None

    for code, added, expressed in (
        ("unsupported_link", CAUSAL_LINKS, RELATION_LANGUAGE),
        ("unsupported_frequency", FREQUENCY_WORDS, FREQUENCY_WORDS),
        ("unsupported_statistic", UNSUPPORTED_STATISTICS, UNSUPPORTED_STATISTICS),
    ):
        found = added.search(claim)
        if found and not expressed.search(passage):
            return Rejection(code, f"{found.group()}: {claim[:70]}")

    return None


#
# Indica si una acotación del encabezado describe un lugar que la oración nombra: en el título «… of a shallow
# temperate estuary (Río de la Plata)», decir «la costa del Río de la Plata» conserva que se trata del estuario. No
# cuenta la primera palabra del encabezado, que va en mayúscula sin ser un nombre.
#
def _names_qualified_place(found, passage, heading, said) -> bool:
    if found.start() <= len(passage) or not heading:
        return False

    names = [item.group() for item in _PROPER_NAME.finditer(heading) if item.start() > 0]
    return any(contains_phrase(said, name) for name in names)


#
# Indica si una acotación del fragmento pertenece a una cláusula que la paráfrasis no retoma: en «presentan una
# gran diversidad de formas, …, y habitan aguas dulces y salobres», «aguas dulces» acota dónde habitan, no la
# diversidad de formas. Sólo se usa con fragmentos en el idioma de la respuesta, porque compara palabras; una
# acotación que forma sola su cláusula («En este estudio, …») sitúa toda la oración y sigue siendo exigible.
#
def _qualifies_another_clause(found, passage, claim) -> bool:
    if found.start() >= len(passage):
        return False

    start = max((item.end() for item in _CLAUSE_BREAK.finditer(passage, 0, found.start())), default=0)
    after = _CLAUSE_BREAK.search(passage, found.end())
    clause = passage[start : after.start() if after else len(passage)]
    qualifier = set(normalize_text(found.group()).split())
    content = {_clause_stem(word) for word in normalize_text(clause).split() if len(word) >= _CLAUSE_WORD} - {
        _clause_stem(word) for word in qualifier
    }
    said = {_clause_stem(word) for word in normalize_text(claim).split() if len(word) >= _CLAUSE_WORD}

    return bool(content) and not content & said


#
# Devuelve la raíz con que se comparan las palabras de una cláusula y de una paráfrasis.
#
def _clause_stem(word) -> str:
    return word[:_CLAUSE_WORD]


#
# Devuelve dónde puede figurar una acotación: la oración misma o, si sitúa el estudio o el ambiente, también
# lo ya dicho en el párrafo y, para el ambiente, la pregunta.
#
def _kept_text(family, claim, earlier, question) -> str:
    if family in SITUATED_FAMILIES:
        return f"{earlier} {claim} {question}"
    if family in {"alcance del estudio", "condición experimental"}:
        return f"{earlier} {claim}"

    return claim


#
# Devuelve los usos del agua que nombra un texto.
#
def _water_uses(text) -> set[str]:
    return {
        use
        for use, phrases in WATER_USES.items()
        if any(contains_phrase(text, phrase, plural_insensitive=True) for phrase in phrases)
    }


#
# Indica si las cifras propias de una oración están todas en un mismo fragmento citado.
#
def _single_source(sentence, holders, shared, language) -> bool:
    readings = [item for item in number_readings(sentence, language) if not item.values & shared]
    if len(readings) < 2:
        return True

    return any(all(item.values & values for item in readings) for values in holders)


#
# Indica si una oración plantea la condición antes de comparar: «si fuera…, quedaría dentro de…».
#
def _conditional_first(sentence) -> bool:
    compared = OBSERVED_COMPARISON.search(sentence)

    return CONDITIONAL_CUES.search(sentence[: compared.end()] if compared else sentence) is not None


#
# Indica si un párrafo habla de los resultados: cita sus hechos o los relaciona con un documento.
#
def _speaks_of_results(segment) -> bool:
    return bool(segment.fact_ids) or segment.role == "relation"


#
# Indica si un número que la evidencia sólo da como cota se escribió como un valor exacto.
#
def _drops_bound(reading, allowed, text) -> bool:
    lost = not reading.comparator and reading.values & allowed.bounded and not reading.values & allowed.exact

    return bool(lost) and not BOUND_WORDS.search(text[max(0, reading.start - _BOUND_WINDOW) : reading.start])


#
# Devuelve las fuentes que una oración nombra como origen de lo que afirma.
#
def _named_sources(sentence) -> list[str]:
    attributed = [
        found.group("name")
        for found in ATTRIBUTED_SOURCE.finditer(sentence)
        if not NEGATION.search(sentence[: found.start()])
    ]

    return list(dict.fromkeys([*attributed, *ORGANIZATION_NAMES.findall(sentence)]))


#
# Devuelve el texto de un fragmento sin las filas de tabla de otros sitios, si define el código del lugar.
#
def rows_of_place(text, place) -> str:
    name = re.escape(place)
    code = re.search(rf"\b([A-Z]{{2,5}})\s*[:=]\s*{name}|{name}\s*\(([A-Z]{{2,5}})\)", text, re.IGNORECASE)
    rows = [line for line in text.splitlines() if _TABLE_CELL in line]
    if code is None or len(rows) < 2:
        return text

    prefix = (code.group(1) or code.group(2)).upper()
    foreign = [line for line in rows[1:] if not _row_label(line).upper().startswith(prefix)]

    return "\n".join(line for line in text.splitlines() if line not in foreign)


#
# Devuelve el rótulo de una fila de tabla: su primera celda con contenido.
#
def _row_label(line) -> str:
    return next((cell.strip() for cell in line.split(_TABLE_CELL) if cell.strip()), "")


#
# Indica si alguna palabra del nombre de una fuente, o un alias suyo, figura en lo citado.
#
def _names_source(name, known) -> bool:
    words = [word for word in name.split() if word[0].isupper()]
    plain = {folded(word) for word in words} | {folded(name)}
    aliases = [alias for group in SOURCE_ALIASES if plain & set(group) for alias in group]

    return any(contains_phrase(known, item) for item in [*words, *aliases])


#
# Devuelve la parte estable de una palabra, para comparar sin importar su flexión.
#
def _root(word: str) -> str:
    singular = word[:-1] if word.endswith("s") else word

    return singular[: max(_MIN_ROOT, len(singular) - _FLEXION_CHARS)]


#
# Devuelve un texto sin sus números, para comparar sólo las palabras.
#
def _without_numbers(text: str) -> str:
    pieces, position = [], 0
    for reading in number_readings(text):
        pieces.append(text[position : reading.start])
        position = reading.end

    return "".join([*pieces, text[position:]])


#
# Devuelve cada número de un hecho con las raíces de las palabras que lo presentan.
#
def _value_labels(fact, language) -> list[tuple]:
    return [
        (
            item.values,
            {
                _root(word)
                for word in normalize_text(fact[: item.start]).split()[-_LABEL_WINDOW:]
                if word not in RESULT_WORDS
            },
        )
        for item in number_readings(fact, language)
    ]


#
# Indica si las palabras previas a un número no lo presentan como el valor de otro hecho.
#
def _aligned(preceding, reading, labels) -> bool:
    near = {_root(word) for word in normalize_text(preceding).split()[-_LABEL_WINDOW:]}
    owners = [items for items in labels.values() if any(reading.values & item[0] for item in items)]
    if not owners:
        return True

    rivals = [items for items in labels.values() if items not in owners]
    own = {word for items in owners for values, label in items if reading.values & values for word in label}
    foreign = {word for items in rivals for _, label in items for word in label}

    return bool(near & (own - foreign)) or not near & (foreign - own)


#
# Indica si una afirmación causal está atenuada justo antes de la causa.
#
def _hedged(sentence, position) -> bool:
    return HEDGE_CUES.search(sentence[max(0, position - _HEDGE_WINDOW) : position]) is not None


#
# Indica si el sentido escrito coincide con el de algún contraste citado.
#
def _direction_agrees(sentence, directions) -> bool:
    if "mixed" in directions:
        return True

    written = {name for name, pattern in COMPARISON_DIRECTIONS.items() if pattern.search(sentence)}

    return not written or bool(written & set(directions))


#
# Indica si la comparación entre dos valores calculados respeta sus cifras («64,88 UNT en un sitio, superior a
# 41,72 UNT en el otro»): True si el sentido coincide, False si lo invierte y None si la oración no da los dos
# valores comparables, en la misma unidad, a cada lado de la comparación.
#
def _ordered_values(sentence, compared, language) -> bool | None:
    above = COMPARISON_DIRECTIONS["above"].search(compared.group()) is not None
    below = COMPARISON_DIRECTIONS["below"].search(compared.group()) is not None
    if above == below:
        return None

    readings = [
        item
        for item in number_readings(sentence, language)
        if not item.comparator and not _NAMED_NUMBER.search(sentence[: item.start])
    ]
    left = [item for item in readings if item.end <= compared.start()]
    right = [item for item in readings if item.start >= compared.end()]
    pairs = [*([(left[-1], right[0])] if left and right else []), *([(right[0], right[1])] if len(right) > 1 else [])]
    same_unit = [
        (first, second)
        for first, second in pairs
        if comparable_unit("", first.unit or "") == comparable_unit("", second.unit or "")
    ]
    if not same_unit:
        return None

    first, second = same_unit[0]
    if above:
        return min(first.values) > max(second.values)

    return max(first.values) < min(second.values)


#
# Devuelve las valoraciones de lo observado (intensidad, nivel, cambio, tipicidad o relación) que el texto hace
# y lo citado no autoriza: una familia de valoraciones sólo se puede usar si los hechos citados la usan.
#
def _unlicensed_assessments(text, licensed) -> list[str]:
    allowed = {family for word in normalize_text(licensed).split() for family in _assessment_families(word)}

    return [
        word
        for word in normalize_text(text).split()
        if any(family not in allowed for family in _assessment_families(word))
    ]


#
# Devuelve las familias de valoración a las que pertenece una palabra normalizada.
#
def _assessment_families(word) -> list[str]:
    return [family for family, pattern in ASSESSMENT_FAMILIES.items() if pattern.fullmatch(word)]


#
# Devuelve las variables que el texto nombra y que no figuran en lo citado, la pregunta ni los resultados: un
# párrafo de resultado no puede atribuir un valor a otra variable.
#
def _unlicensed_variables(text, licensed) -> list[str]:
    extra = named_variable_groups(text) - named_variable_groups(licensed)
    known = set(normalize_text(licensed).split())
    words = normalize_text(text).split()
    named = [word for word in words if named_variable_groups(word) & extra]
    acidity = [word for word in words if word in PH_ALIASES and word not in known]

    return [*(named or (["variable"] if extra else [])), *acidity]


#
# Devuelve los nombres propios (sitios, organismos, fuentes, instituciones) que el texto usa fuera del inicio de
# una oración y que no figuran en lo citado, la pregunta ni los resultados.
#
def _unlicensed_names(text, licensed) -> list[str]:
    known = set(normalize_text(licensed).split())
    found = []
    for item in sentences(text):
        first = _FIRST_WORD.search(item)
        for match in _CAPITALIZED.finditer(item):
            if first is not None and match.start() <= first.start():
                continue
            parts = normalize_text(match.group()).split()
            if any(not _known_name(part, known) for part in parts):
                found.append(match.group())

    return found


#
# Indica si una palabra de un nombre propio figura entre las conocidas, tal cual o con otra flexión.
#
def _known_name(part, known) -> bool:
    if part in known or len(part) < _NAME_STEM:
        return part in known or part.isdigit()

    return any(word[:_NAME_STEM] == part[:_NAME_STEM] for word in known if len(word) >= _NAME_STEM)
