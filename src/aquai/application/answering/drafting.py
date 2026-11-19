import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

from aquai.application.answering.measures import measured_results, shown_value
from aquai.application.answering.readings import folded, number_readings, sentences, without_numbers
from aquai.application.answering.references import documented_guidelines
from aquai.application.answering.verification import (
    DraftEvidence,
    DraftVerifier,
    Rejection,
    RequiredValue,
)
from aquai.application.answering.wording import dataset_label, file_stem, sentence
from aquai.application.contracts.answering import (
    AnswerSegment,
    DocumentReference,
    ResponseContextRecord,
)
from aquai.application.execution import decode_rows, execution_records
from aquai.application.matching import variable_name, variable_variants
from aquai.application.Rules.evidence import (
    ANAPHORIC_START,
    ATTRIBUTED_SOURCE,
    ATTRIBUTION_LEAD,
    COMPARISON_DIRECTIONS,
    EVIDENCE_GAP,
    EVIDENCE_NOUNS,
    GUIDELINE_CUES,
    INTERPRETATION_CUES,
    LEADING_CONNECTOR,
    LITERATURE_CLAUSE,
    ORGANIZATION_NAMES,
    PAST_NARRATION,
    PENDING_CONFIRMATION,
    REFERENCE_LINE_CUES,
    UNNAMED_STUDY_START,
)

_REQUIRED_KINDS = {"result"}
_KEPT_KINDS = {"scope", "detail"}
_INTERPRETING_ROLES = {"documented", "relation"}
_UNRETRIED_CODES = {"unbacked_citation", "unaudited_claim", "scoped_claim"}
_CITATION_MARGIN = 0.03
_MAX_EXCERPT = 600
_MISSING_INTERPRETATION = (
    "falta un párrafo documented: parafraseá de cerca lo que dicen los fragmentos sobre esta variable"
)
_UNEXPLORED_DOCUMENTS = (
    "documented: la interpretación usa un solo fragmento; revisá también estos y usá los que expliquen qué "
    "representa la variable, con qué factores se relaciona o cómo varía en el mismo sitio o sistema, cada uno "
    "en su párrafo y sólo si aporta"
)
_MIN_INTERPRETING_DOCUMENTS = 2
_MISSING_HYPOTHESIS = (
    "falta un párrafo hypothesis: con las lecturas de ambiguities, en condicional, citando sus documentos"
)
_MISSING_VALUES = (
    "documented: la pregunta pide valores de referencia; presentá cada uno de estos con la condición en que "
    "rige según su fragmento y citándolo, en un párrafo por fragmento, nombrando el valor y su condición en "
    "la misma oración, sin remitir a otra (nada de el primero o este nivel)"
)
_VALUE_NOTES = {
    "es": "Los documentos dan además el valor de referencia {value} ({source}), que no pude redactar de forma verificable.",
    "en": "The documents also give the reference value {value} ({source}), which I could not write in a verifiable way.",
}
_MIN_SENTENCE_WORDS = 6
_MIN_PASSAGE_CHARS = 10
_CONTRAST_SPLIT = re.compile(
    r";\s+|,\s+(?:while|whereas|but|mientras\s+que|en\s+tanto\s+que|pero|aunque)\s+", re.IGNORECASE
)
_LISTING_MARKS = 3
_PHRASE_BREAK = re.compile(r"\s*(?:[,;:()\[\]|·•]|\s[-–]\s|\band/or\b|\by/o\b)\s*", re.IGNORECASE)
_MIN_PHRASE_WORDS = 2
_MAX_RESTORED_COMPARISONS = 2
_MAX_FACT_VALUES = 400
_MAX_PREVIOUS_ANSWER = 1500
_WORD = re.compile(r"[^\W\d_]+")
_STEM = 5
_SHORT_WORD = 3
_REPEATED_SHARE = 0.7
_REPEATED_SENTENCE = 0.7
_MIN_REPEATED_WORDS = 4
_CORRECTIONS = {
    "unsupported_place": "el fragmento no nombra ese lugar: atribuí el dato al estudio o sistema que sí nombra",
    "unsupported_unit": "copiá la unidad tal como figura junto a ese número en la evidencia",
    "unsupported_number": (
        "ese número no está en lo que cita el párrafo (un párrafo relation sólo usa los de sus facts); "
        "escribilo con los separadores del idioma de la respuesta"
    ),
    "mixed_sources": "no mezcles en una oración cifras de fragmentos distintos: una oración por fuente",
    "lost_qualifier": (
        "agregá esa acotación del fragmento con tus palabras, en esa oración o antes en el mismo párrafo "
        "(por ejemplo: en agua dulce, en aguas marinas, en ese estudio, en laboratorio, algunas, puede); si "
        "viene de otra frase del fragmento (cómo es el sitio de estudio), no la mezcles: decí el hallazgo en "
        "una oración propia"
    ),
    "unsupported_link": "el fragmento no establece esa relación de causa o influencia: decí sólo lo que afirma",
    "unsupported_statistic": "no agregues conclusiones estadísticas que el fragmento no expresa",
    "hidden_value": "no copies [valor no confirmado]: si un valor no está confirmado, no lo menciones",
    "unsupported_result_word": (
        "esa valoración, variable o nombre no figura en los facts que cita el párrafo: decí sólo lo que esos "
        "facts dicen, con tus palabras"
    ),
    "wrong_direction": "esa comparación invierte el orden de los valores: revisá cuál es mayor",
    "unsupported_absence": (
        "no afirmes que no hay datos: que el análisis no los haya obtenido no prueba que no existan; las notas ya "
        "avisan qué parte no se pudo responder"
    ),
    "observed_in_documented": "un párrafo documented habla de los documentos, no de los datos analizados",
    "unsupported_claim": "reescribí esa oración más cerca del texto del fragmento",
    "unsupported_comparison": "no compares lo observado sin un fact de kind comparison",
    "unsupported_verdict": "no emitas un juicio sobre lo observado",
    "unsupported_recommendation": (
        "no recomiendes acciones a partir de lo observado; si un documento propone acciones para un nivel, "
        "decilo en un párrafo documented como lo que ese documento indica"
    ),
    "unsupported_classification": "no clasifiques lo observado",
    "unsupported_causality": "no afirmes causas de lo observado",
    "unsupported_attribution": "atribuí cada dato sólo a la fuente del fragmento que lo contiene",
    "unsupported_frequency": "no agregues frecuencia ni generalidad: decí de qué estudio y sitio es el dato",
    "unsupported_reference_label": (
        "ese valor es lo que midió o reportó el estudio, no un valor de referencia, guía, umbral ni límite: "
        "presentalo como el valor que reporta ese estudio"
    ),
    "internal_term": "no uses términos internos de la evidencia",
    "dangling_reference": "ese párrafo remitía a otro que se quitó: escribilo de forma autónoma",
}
_ROLE_LABELS = {"es": "Hipótesis, no confirmada:", "en": "Unconfirmed hypothesis:"}
ATTACHMENT_LEADS = {"es": "Detalle de ese resultado:", "en": "Detail of that result:"}


@dataclass
class WriterRequest:
    """Lo que necesita una redacción: el contexto del modelo y la evidencia para verificarla."""

    record: ResponseContextRecord
    evidence: DraftEvidence
    catalog: dict = field(default_factory=dict)
    ambiguities: list = field(default_factory=list)


@dataclass
class ReviewedDraft:
    accepted: list
    rejections: list[tuple[str, Rejection]]
    missing_facts: list[str]
    dropped: int = 0

    #
    # Indica si conviene publicar el borrador sin otro intento: resultados completos, los valores de
    # referencia documentados presentados, nada perdido en la verificación y la interpretación (con más de
    # un fragmento, si los hay) o la hipótesis que hagan falta.
    #
    def sufficient(self, evidence: DraftEvidence) -> bool:
        return (
            bool(self.accepted)
            and not self.missing_facts
            and not self.missing_values(evidence)
            and not self.lost
            and (self.interprets or not evidence.documents_required)
            and not self.unexplored(evidence)
            and (self.reads_ambiguity or not evidence.allows_hypotheses)
        )

    #
    # Devuelve los fragmentos sin usar cuando la interpretación pedida se apoya en uno solo habiendo varios.
    #
    def unexplored(self, evidence: DraftEvidence) -> list[str]:
        cited = {key for item in self.accepted if item.role in _INTERPRETING_ROLES for key in item.document_ids}
        enough = min(len(evidence.documents), _MIN_INTERPRETING_DOCUMENTS)
        if not evidence.documents_required or not cited or len(cited) >= enough:
            return []

        return [key for key in evidence.documents if key not in cited]

    #
    # Devuelve los valores de referencia documentados que el borrador no presentó.
    #
    def missing_values(self, evidence: DraftEvidence) -> list[RequiredValue]:
        return DraftVerifier(evidence).missing_values(self.accepted)

    #
    # Indica si el borrador explica o relaciona algo con documentos.
    #
    @property
    def interprets(self) -> bool:
        return any(item.document_ids and item.role in _INTERPRETING_ROLES for item in self.accepted)

    #
    # Devuelve los problemas que quitaron texto del borrador y que otro intento puede corregir.
    #
    @property
    def lost(self) -> list[Rejection]:
        return [problem for _, problem in self.rejections if problem.code not in _UNRETRIED_CODES]

    #
    # Indica si el borrador da una lectura condicional con sus valores, no sólo un aviso.
    #
    @property
    def reads_ambiguity(self) -> bool:
        return any(item.role == "hypothesis" and number_readings(item.text) for item in self.accepted)

    #
    # Devuelve qué tan útil es el borrador para elegir entre intentos: completo, con la interpretación
    # pedida, con los valores de referencia documentados, con más oraciones documentales verificadas y,
    # a igualdad, con menos texto perdido.
    #
    def score(self, evidence: DraftEvidence) -> tuple:
        verified = sum(
            len(sentences(item.text)) + len(item.items)
            for item in self.accepted
            if item.role in _INTERPRETING_ROLES
        )
        interpreted = self.interprets or not evidence.documents_required
        missing = len(self.missing_values(evidence))

        return not self.missing_facts, interpreted, -missing, verified, -len(self.lost)


#
# Devuelve la cita legible de un fragmento: archivo y páginas.
#
def fragment_label(fragment) -> str:
    pages = pages_text(fragment.page_start, fragment.page_end)

    return f"{file_stem(fragment.file_name)}{f', {pages}' if pages else ''}"


#
# Devuelve las páginas de un fragmento como texto.
#
def pages_text(start, end) -> str:
    if start is None:
        return ""

    return f"p. {start}" if end in {None, start} else f"pp. {start}-{end}"


#
# Devuelve los fragmentos que entran en la redacción, sin superar el presupuesto.
#
def _fitted_documents(body, fragments, budget) -> list:
    kept = []
    for fragment in fragments:
        candidate = [*body["documents"], _document_entry(fragment)]
        if _size({**body, "documents": candidate}) > budget:
            break
        body["documents"] = candidate
        kept.append(fragment)

    return kept


#
# Construye la entrada de un fragmento para el redactor.
#
def _document_entry(fragment) -> dict:
    return {
        "id": fragment.evidence_id,
        "source": fragment_label(fragment),
        "title": fragment.title,
        "section": " > ".join(fragment.section[-2:]) or None,
        "kind": fragment.kind,
        "text": without_numbers(fragment.text) if fragment.doubts else fragment.text,
        "extraction_warning": bool(fragment.doubts),
    }


#
# Devuelve el tamaño en bytes del contexto serializado.
#
def _size(body) -> int:
    return len(_serialized(body))


#
# Serializa el contexto del redactor.
#
def _serialized(body) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


#
# Cierra el registro del contexto del redactor con su huella.
#
def _record(body, budget, omitted) -> ResponseContextRecord:
    content = _serialized(body)

    return ResponseContextRecord(
        evidence=body,
        sha256=sha256(content).hexdigest(),
        size_bytes=len(content),
        budget_bytes=budget,
        omitted_rows=omitted,
    )


#
# Arma la redacción de una respuesta analítica a partir del catálogo verificado.
#
def analytical_request(state, context, language, budget, *, notes=(), comparisons=(), ambiguities=()):
    catalog = {item["sentence_id"]: item for item in context.evidence["sentence_catalog"]}
    question = state.routing.original_question if state.routing else state.question
    requested = asks_documents(state, question)
    analyzed = requested or _charted(state)
    body = {
        "question": question,
        "language": language,
        "status": context.evidence["status"],
        "interpretation_requested": analyzed,
        "facts": _facts(catalog),
        "ambiguities": [item.model_dump(mode="json") for item in ambiguities],
        "notes": list(notes),
        "documents": [],
    }
    kept = _fitted_documents(body, _usable_fragments(state), budget)
    compared = state.routing is not None and state.routing.reference_comparison is not None
    required = documented_guidelines(kept, _asked_subject(state), question) if compared else []
    available = {item.evidence_id for item in kept}
    contrasted = [item for item in comparisons if item.document_id in available]
    legible = [item for item in ambiguities if set(item.evidence_ids) & available]
    catalog.update(_comparison_entries(contrasted))
    body["facts"] = _facts(catalog, without_definitions=bool(kept) and requested)
    body["ambiguities"] = [item.model_dump(mode="json") for item in legible]
    evidence = DraftEvidence(
        facts={item["id"]: item["text"] for item in body["facts"]},
        fact_documents=_fact_documents(catalog),
        fact_directions=_fact_directions(catalog),
        fact_values=_fact_values(state, catalog),
        required_facts={key for key, item in catalog.items() if item["kind"] in _REQUIRED_KINDS},
        documents={item.evidence_id: item.text for item in kept},
        document_languages={item.evidence_id: item.language for item in kept},
        doubtful_documents={item.evidence_id for item in kept if item.doubts},
        document_labels={item.evidence_id: fragment_label(item) for item in kept},
        document_sources={item.evidence_id: _fragment_source(item) for item in kept},
        document_owners={item.evidence_id: str(item.document_id) for item in kept},
        hypothesis_values=_hypothesis_texts(legible),
        question=question,
        result_names=[name for item in measured_results(state, language) for name in item.names],
        allows_hypotheses=bool(legible),
        documents_required=requested and bool(kept),
        required_values=required,
        language=language,
        places=_asked_places(state),
        document_contexts={item.evidence_id: _fragment_context(item) for item in kept},
        variable_names=_asked_variables(state),
        source_names=_source_names(state),
    )

    return WriterRequest(_record(body, budget, context.omitted_rows), evidence, catalog, legible)


#
# Devuelve los nombres de los archivos y hojas del análisis, que un párrafo puede mencionar aunque tengan dígitos
# («BaseCiano_PARA BELEN_5MAR2024 (2)»), del más largo al más corto.
#
def _source_names(state) -> list[str]:
    plan = state.planning
    names = {
        name
        for item in (plan.selected_datasets if plan else [])
        for name in (item.provenance.source_name, file_stem(item.provenance.source_name), dataset_label(item.name))
        if name
    }

    return sorted(names, key=len, reverse=True)


#
# Indica si la pregunta pide interpretar el resultado o contrastarlo con valores de referencia.
#
def asks_interpretation(question: str) -> bool:
    return bool(INTERPRETATION_CUES.search(question) or REFERENCE_LINE_CUES.search(question))


#
# Indica si la pregunta pide que la respuesta se apoye en los documentos: interpretar o contrastar el resultado,
# lo que dice la bibliografía o lo que comunica un documento que nombra («lo que comunica Giannuzzi et al.»).
#
def asks_documents(state, question: str) -> bool:
    return asks_interpretation(question) or asks_documents_alone(state, question)


#
# Indica si la pregunta pide a los documentos algo que no depende del resultado calculado: lo que dice la
# bibliografía, lo que comunica un documento que nombra o un valor de referencia. Pedir que se interprete el
# resultado no alcanza: si el cálculo falló no hay nada que interpretar («decime si puede interpretarse como
# una tendencia»).
#
def asks_documents_alone(state, question: str) -> bool:
    named = state.routing is not None and bool(state.routing.document_ids)

    return named or bool(REFERENCE_LINE_CUES.search(question) or LITERATURE_CLAUSE.search(question))


#
# Indica si la respuesta publica un gráfico: mostrar un patrón invita a explicar qué significa, aunque los
# documentos sólo se exigen si la pregunta los pide.
#
def _charted(state) -> bool:
    return any(item.kind == "visualization" for item in state.validation.usable_evidence)


#
# Devuelve los documentos que respaldan a cada contraste del catálogo.
#
def _fact_documents(catalog) -> dict:
    return {key: entry["document_ids"] for key, entry in catalog.items() if entry.get("document_ids")}


#
# Devuelve el sentido de cada contraste del catálogo.
#
def _fact_directions(catalog) -> dict:
    return {key: entry["direction"] for key, entry in catalog.items() if entry.get("direction")}


#
# Devuelve los hechos del catálogo tal como los recibe el redactor; con documentos no hacen falta
# las definiciones genéricas de las estadísticas.
#
def _facts(catalog, *, without_definitions=False) -> list[dict]:
    return [
        {"id": key, "kind": entry["kind"], "text": entry["direct"]}
        for key, entry in catalog.items()
        if not (without_definitions and entry["kind"] == "meaning")
    ]


#
# Convierte los contrastes verificados en hechos citables con su documento.
#
def _comparison_entries(comparisons) -> dict:
    return {
        f"c{index}": {
            "sentence_id": f"c{index}",
            "kind": "comparison",
            "direct": item.text,
            "evidence_ids": [item.evidence_id],
            "document_ids": [item.document_id],
            "direction": item.direction,
            "attachment": None,
        }
        for index, item in enumerate(comparisons, 1)
    }


#
# Arma la redacción de una respuesta documental.
#
def documental_request(state, language, budget, previous, analyses=(), notes=()):
    body = {
        "question": state.question,
        "language": language,
        "facts": [],
        "comparisons": [],
        "ambiguities": [],
        "previous_answers": [item[:_MAX_PREVIOUS_ANSWER] for item in previous],
        "notes": list(notes),
        "documents": [],
    }
    kept = _fitted_documents(body, _usable_fragments(state), budget)
    catalog = _retaken_facts(analyses, kept)
    body["facts"] = _facts(catalog)
    retaken = [item.text for message in analyses for item in message.scope if item.kind == "value"]
    named = [item for item in _asked_places(state) if item[:1].isupper()]
    asked = REFERENCE_LINE_CUES.search(state.question) is not None
    required = documented_guidelines(kept, _asked_subject(state), state.question) if asked else []
    evidence = DraftEvidence(
        facts={item["id"]: item["text"] for item in body["facts"]},
        fact_documents=_fact_documents(catalog),
        fact_directions=_fact_directions(catalog),
        documents={item.evidence_id: item.text for item in kept},
        document_languages={item.evidence_id: item.language for item in kept},
        doubtful_documents={item.evidence_id for item in kept if item.doubts},
        document_labels={item.evidence_id: fragment_label(item) for item in kept},
        document_sources={item.evidence_id: _fragment_source(item) for item in kept},
        document_owners={item.evidence_id: str(item.document_id) for item in kept},
        question=state.question,
        previous_answers=body["previous_answers"],
        documents_required=bool(kept),
        required_values=required,
        language=language,
        places=list(dict.fromkeys([*named, *retaken])),
        document_contexts={item.evidence_id: _fragment_context(item) for item in kept},
        variable_names=_asked_variables(state),
    )

    return WriterRequest(_record(body, budget, {}), evidence)


#
# Convierte lo verificado en los análisis que un seguimiento retoma en hechos que puede citar.
#
def _retaken_facts(analyses, fragments) -> dict:
    current = {str(item.chunk_id): item.evidence_id for item in fragments}
    catalog = {}
    for message in analyses:
        documents = {item.get("evidence_id"): current.get(str(item.get("chunk_id"))) for item in message.documents}
        for segment in message.segments:
            entry = _retaken_entry(segment, documents)
            if entry is not None:
                key = f"p{len(catalog) + 1}"
                catalog[key] = {**entry, "sentence_id": key}

    return catalog


#
# Devuelve el hecho de un párrafo retomado: un resultado (también el párrafo determinista que cita evidencia),
# o un contraste con su documento a la vista.
#
def _retaken_entry(segment, documents):
    text = segment.get("text", "")
    deterministic = segment.get("role") is None and segment.get("kind", "paragraph") == "paragraph"
    if segment.get("role") == "result" or (deterministic and segment.get("evidence_ids")):
        return {"kind": "result", "direct": text, "evidence_ids": [], "attachment": None}

    written = [name for name, pattern in COMPARISON_DIRECTIONS.items() if pattern.search(text)]
    cited = [documents[key] for key in segment.get("document_ids") or [] if documents.get(key)]
    if segment.get("role") != "relation" or len(written) != 1 or not cited:
        return None

    return {
        "kind": "comparison",
        "direct": text,
        "evidence_ids": [],
        "document_ids": cited,
        "direction": written[0],
        "attachment": None,
    }


#
# Devuelve los nombres del tema preguntado: los de su variable o, si no se nombró una, los términos con que
# se buscó, sin organismos ni palabras de guía.
#
def _asked_subject(state) -> list[str]:
    mentions = state.routing.data_mentions if state.routing else []
    variables = [item.text for item in mentions if not isinstance(item, str) and item.kind == "variable"]
    named = [name for mention in variables for name in variable_variants(variable_name(mention))]
    retrieval = state.document_retrieval
    searched = [
        term
        for term in (retrieval.subject if retrieval is not None else [])
        if not ORGANIZATION_NAMES.search(term) and not GUIDELINE_CUES.search(term)
    ]

    return named or searched


#
# Devuelve los nombres de las variables que la consulta pide, tal como los escribió.
#
def _asked_variables(state) -> list[str]:
    mentions = state.routing.data_mentions if state.routing else []

    return [item.text for item in mentions if not isinstance(item, str) and item.kind == "variable"]


#
# Devuelve los lugares y valores por los que se acotó la consulta.
#
def _asked_places(state) -> list[str]:
    mentions = state.routing.data_mentions if state.routing else []

    return [item.text for item in mentions if not isinstance(item, str) and item.kind == "value"]


#
# Devuelve cómo se identifica la fuente de un fragmento: su cita, su título y su archivo.
#
def _fragment_source(fragment) -> str:
    return " ".join([fragment_label(fragment), fragment.title or "", fragment.file_name])


#
# Devuelve el texto de un fragmento con su sección y su leyenda.
#
def _fragment_context(fragment) -> str:
    heading = [fragment.title or "", fragment.file_name, *fragment.section, fragment.element_caption or ""]

    return " ".join([*heading, fragment.text])


#
# Devuelve los fragmentos recuperados para el turno.
#
def _usable_fragments(state) -> list:
    retrieval = state.document_retrieval

    return retrieval.fragments if retrieval is not None else []


#
# Devuelve los valores exactos que cada hecho puede citar: los de las tablas en las que se apoya.
#
def _fact_values(state, catalog) -> dict[str, list[str]]:
    tables = _table_values(state)

    return {
        key: [value for evidence in entry["evidence_ids"] for value in tables.get(evidence, [])]
        for key, entry in catalog.items()
    }


#
# Devuelve los valores de cada tabla autorizada.
#
def _table_values(state) -> dict[str, list[str]]:
    records = execution_records(state)
    usable = [item for item in state.validation.usable_evidence if item.kind == "table"]
    values = {}
    for evidence in usable:
        record = records[evidence.operation_id]
        rows = decode_rows(record, record.columns)
        found = [str(value) for row in rows for value in row.values() if value is not None]
        values[evidence.evidence_id] = found[:_MAX_FACT_VALUES]

    return values


#
# Devuelve los textos con las lecturas condicionales que una hipótesis puede citar.
#
def _hypothesis_texts(ambiguities) -> list[str]:
    return [
        " ; ".join([f"{item.observed} {item.unit}", *item.equivalents, *item.documented])
        for item in ambiguities
    ]


#
# Separa los párrafos publicables del borrador de los que la verificación rechaza.
#
def review_draft(draft, evidence: DraftEvidence) -> ReviewedDraft:
    verifier = DraftVerifier(evidence)
    accepted, rejections, lost = [], [], False
    for item in draft.segments:
        segment = _with_compared_documents(item, evidence)
        problems = verifier.check_segment(segment)
        if lost and not problems and ANAPHORIC_START.search(segment.text):
            problems = [Rejection("dangling_reference", segment.text[:60])]
        kept = _salvaged(segment, verifier, lost) if problems else segment
        rejections.extend((segment.role, problem) for problem in problems)
        accepted.extend([kept] if kept is not None else [])
        lost = kept is None

    return ReviewedDraft(accepted, rejections, verifier.missing_facts(accepted))


#
# Conserva de un párrafo rechazado las oraciones que sí se pueden respaldar, cada una verificada sola con lo que
# cita el párrafo: un número mal puesto en una oración no se lleva las demás. Lo que falte de los resultados
# obligatorios se restituye después.
#
def _salvaged(segment, verifier, lost=False):
    kept, dropped = [], lost
    for item in sentences(segment.text):
        continues = dropped and ANAPHORIC_START.search(item) is not None
        valid = len(item.split()) >= _MIN_SENTENCE_WORDS and not continues
        if valid and not verifier.check_segment(segment.model_copy(update={"text": item, "items": []})):
            kept.append(item)
            dropped = False
        else:
            dropped = True

    if not kept:
        return None

    return segment.model_copy(update={"text": _tidied(kept), "items": []})


#
# Une las oraciones que quedaron, sin el conector que remitía a una oración quitada.
#
def _tidied(kept) -> str:
    text = LEADING_CONNECTOR.sub("", " ".join(kept)).rstrip(" ;,")
    closed = text if text.endswith((".", "!", "?")) else f"{text}."

    return closed[:1].upper() + closed[1:]


#
# Agrega al párrafo los documentos de los contrastes que cita.
#
def _with_compared_documents(segment, evidence):
    compared = [
        key for fact in segment.fact_ids for key in evidence.fact_documents.get(fact, [])
    ]
    if not compared:
        return segment

    documents = list(dict.fromkeys([*segment.document_ids, *compared]))

    return segment.model_copy(update={"document_ids": documents})


#
# Devuelve el contexto de un nuevo intento con los problemas del anterior.
#
def retry_record(request: WriterRequest, reviewed: ReviewedDraft) -> ResponseContextRecord:
    problems = [
        f"{role}: «{problem.detail}»: {_CORRECTIONS.get(problem.code, problem.code)}"
        for role, problem in reviewed.rejections
    ]
    problems += [f"result: falta mencionar {key}" for key in reviewed.missing_facts]
    missing = reviewed.missing_values(request.evidence)
    if missing:
        listed = "; ".join(f"{item.shown} ({item.evidence_id})" for item in request.evidence.required_values)
        problems.insert(0, f"{_MISSING_VALUES}: {listed}")
        problems += [f"documented: falta {item.shown} ({item.evidence_id}): «{item.sentence[:160]}»" for item in missing]
    if request.evidence.documents_required and not reviewed.interprets:
        problems.append(_MISSING_INTERPRETATION)
    unexplored = reviewed.unexplored(request.evidence)
    if unexplored:
        problems.insert(0, f"{_UNEXPLORED_DOCUMENTS}: {', '.join(unexplored)}")
    if request.evidence.allows_hypotheses and not reviewed.reads_ambiguity:
        problems.insert(0, _MISSING_HYPOTHESIS)
    body = {**request.record.evidence, "rejected_draft": problems[:12]}

    return _record(body, request.record.budget_bytes, request.record.omitted_rows)


#
# Convierte el borrador verificado en segmentos, reponiendo lo obligatorio que omitió.
#
def published_segments(reviewed: ReviewedDraft, request: WriterRequest, language) -> list[AnswerSegment]:
    catalog = request.catalog
    kept = [item for item in reviewed.accepted if not _covers_missing(item, reviewed.missing_facts)]
    if request.ambiguities and not reviewed.reads_ambiguity:
        kept = [item for item in kept if item.role != "hypothesis"]
    segments = _without_repetitions([_segment(item, catalog, language) for item in kept])
    published = _with_attachments(_with_restored(segments, reviewed, kept, catalog), reviewed.accepted, catalog, language)
    published.extend(_unused_comparisons(reviewed.accepted, catalog))
    if request.ambiguities and not reviewed.reads_ambiguity:
        published.extend(_written_hypotheses(request.ambiguities, language))
    published.extend(_value_notes(reviewed.missing_values(request.evidence), request.evidence, language))

    return published


#
# Avisa qué valores de referencia documentados no quedaron en la redacción verificada, con su fuente.
#
def _value_notes(missing, evidence, language) -> list[AnswerSegment]:
    return [
        AnswerSegment(
            kind="note",
            text=_VALUE_NOTES[language].format(
                value=shown_value(min(item.values), language),
                source=evidence.document_labels.get(item.evidence_id, item.evidence_id),
            ),
            document_ids=[item.evidence_id],
        )
        for item in missing
    ]


#
# Devuelve los contrastes verificados que el borrador no llegó a usar.
#
def _unused_comparisons(accepted, catalog) -> list[AnswerSegment]:
    cited = {key for item in accepted for key in item.fact_ids}
    unused = [
        entry
        for key, entry in catalog.items()
        if entry["kind"] == "comparison" and key not in cited
    ]

    return [
        AnswerSegment(
            role="relation",
            text=sentence(entry["direct"]),
            evidence_ids=list(entry["evidence_ids"]),
            document_ids=list(entry["document_ids"]),
        )
        for entry in unused[:_MAX_RESTORED_COMPARISONS]
    ]


#
# Publica las lecturas ya calculadas cuando el modelo no redactó una hipótesis válida.
#
def _written_hypotheses(ambiguities, language) -> list[AnswerSegment]:
    first = ambiguities[0]
    label = _ROLE_LABELS[language]
    lead = (
        f"Falta confirmar la unidad de {first.column} en la fuente original. {label} según los documentos, "
        f"el valor {first.observed} podría leerse así, sin que se pueda saber cuál es la correcta:"
        if language == "es"
        else f"The unit of {first.column} still has to be confirmed in the original source. {label} "
        f"according to the documents, {first.observed} could be read as follows, and the right reading "
        "cannot be known yet:"
    )

    return [
        AnswerSegment(
            role="hypothesis",
            text=lead,
            items=[_reading(item, language) for item in ambiguities],
            document_ids=list(dict.fromkeys(key for item in ambiguities for key in item.evidence_ids)),
        )
    ]


#
# Redacta una lectura condicional con sus equivalencias y su relación con lo documentado.
#
def _reading(item, language) -> str:
    details = [*item.documented]
    if item.equivalents:
        same = "equivale a" if language == "es" else "equals"
        details.append(f"{same} {', '.join(item.equivalents)}")
    prefix = "si fuera" if language == "es" else "if it were"
    suffix = f": {'; '.join(details)}" if details else ""

    return f"{prefix} {item.observed} {item.unit}{suffix}"


#
# Restituye los hechos obligatorios que el borrador no dijo en el orden del catálogo: los anteriores a lo que dicen
# los párrafos de resultado van primero y los posteriores (un aviso sobre esos resultados), después de ellos.
#
def _with_restored(segments, reviewed, kept, catalog) -> list[AnswerSegment]:
    order = {key: index for index, key in enumerate(catalog)}
    cited = [order[key] for item in kept if item.role == "result" for key in item.fact_ids if key in order]
    last = max(cited, default=-1)
    keys = _facts_to_restore(reviewed, kept, catalog)
    published = [*(_restored(catalog[key]) for key in keys if order[key] < last), *segments]
    index = _after_results(published)
    published[index:index] = [_restored(catalog[key]) for key in keys if order[key] > last]

    return published


#
# Devuelve los hechos obligatorios que ningún párrafo conservado sigue mostrando.
#
def _facts_to_restore(reviewed, kept, catalog) -> list[str]:
    dropped = [item for item in reviewed.accepted if item not in kept]
    still_cited = {key for item in kept if item.role == "result" for key in item.fact_ids}
    pending = [*reviewed.missing_facts, *(key for item in dropped for key in item.fact_ids)]

    return [
        key
        for key in dict.fromkeys(pending)
        if key not in still_cited and catalog.get(key, {}).get("kind") in _REQUIRED_KINDS
    ]


#
# Indica si un párrafo de resultado cita un hecho que no reproduce por completo.
#
def _covers_missing(item, missing) -> bool:
    return item.role == "result" and any(key in missing for key in item.fact_ids)


#
# Convierte un párrafo del borrador en un segmento de respuesta.
#
def _segment(item, catalog, language) -> AnswerSegment:
    evidence_ids = [
        key for fact in item.fact_ids if fact in catalog for key in catalog[fact]["evidence_ids"]
    ]
    text = item.text.strip()
    if item.role == "hypothesis":
        text = f"{_ROLE_LABELS[language]} {text}"

    return AnswerSegment(
        role=item.role,
        text=text,
        items=[entry.strip() for entry in item.items if entry.strip() and not _repeats(entry, text)],
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        document_ids=list(dict.fromkeys(item.document_ids)),
    )


#
# Quita de los párrafos documentales las oraciones que repiten otra ya publicada: casi todas sus palabras dichas y
# ningún número nuevo. El modelo a veces reformula al final lo que dijo al principio.
#
def _without_repetitions(segments) -> list[AnswerSegment]:
    said, published = [], []
    for segment in segments:
        texts = sentences(segment.text)
        kept = [text for text in texts if segment.role not in _INTERPRETING_ROLES or not _said_before(text, said)]
        said.extend(_sentence_content(text) for text in kept)
        if kept == texts:
            published.append(segment)
        elif kept:
            published.append(segment.model_copy(update={"text": _tidied(kept)}))

    return published


#
# Indica si una oración repite alguna de las ya publicadas.
#
def _said_before(text, said) -> bool:
    words, numbers = _sentence_content(text)
    if len(words) < _MIN_REPEATED_WORDS:
        return False

    return any(
        numbers <= earlier_numbers and len(words & earlier_words) >= _REPEATED_SENTENCE * len(words)
        for earlier_words, earlier_numbers in said
    )


#
# Devuelve las raíces de las palabras con contenido de una oración y sus números.
#
def _sentence_content(text) -> tuple[set, frozenset]:
    words = {word[:_STEM] for word in _WORD.findall(folded(text)) if len(word) > _SHORT_WORD}
    numbers = frozenset(value for reading in number_readings(text) for value in reading.values)

    return words, numbers


#
# Indica si una viñeta sólo repite su párrafo: mismos números y casi todas sus palabras ya dichas.
#
def _repeats(entry, text) -> bool:
    values = {value for reading in number_readings(text) for value in reading.values}
    numbers = [reading.values for reading in number_readings(entry)]
    if any(not found & values for found in numbers):
        return False

    said = {word[:_STEM] for word in _WORD.findall(folded(text))}
    words = [word[:_STEM] for word in _WORD.findall(folded(entry)) if len(word) > _SHORT_WORD]

    return bool(words) and sum(word in said for word in words) >= _REPEATED_SHARE * len(words)


#
# Devuelve el enunciado verificado de un hecho que el borrador omitió.
#
def _restored(entry) -> AnswerSegment:
    return AnswerSegment(
        role="result",
        text=entry["direct"],
        evidence_ids=list(entry["evidence_ids"]),
        attachment=entry.get("attachment"),
    )


#
# Ubica cada tabla o gráfico tras el párrafo que lo cita y conserva los enunciados de alcance.
#
def _with_attachments(segments, accepted, catalog, language) -> list[AnswerSegment]:
    placed = {segment.attachment for segment in segments if segment.attachment}
    cited = {key for item in accepted for key in item.fact_ids}
    for key, entry in catalog.items():
        attachment = entry.get("attachment")
        if attachment is None or attachment in placed:
            continue
        target = _citing_segment(segments, entry)
        if target is not None:
            target.attachment = attachment
        elif key in cited:
            segments.insert(_after_results(segments), _attachment_only(entry, language))
        else:
            segments.insert(_after_results(segments), _restored(entry))
        placed.add(attachment)

    segments.extend(
        AnswerSegment(kind="note", text=entry["direct"], evidence_ids=list(entry["evidence_ids"]))
        for key, entry in catalog.items()
        if entry["kind"] in _KEPT_KINDS and key not in cited
    )

    return segments


#
# Devuelve la posición que sigue a los párrafos de resultado con que empieza la respuesta.
#
def _after_results(segments) -> int:
    return next((index for index, item in enumerate(segments) if item.role != "result"), len(segments))


#
# Devuelve el adjunto de un hecho que ya se redactó, sin repetir su enunciado.
#
def _attachment_only(entry, language) -> AnswerSegment:
    return AnswerSegment(
        role="result",
        text=ATTACHMENT_LEADS[language],
        evidence_ids=list(entry["evidence_ids"]),
        attachment=entry["attachment"],
    )


#
# Devuelve el primer segmento sin adjunto que usa la evidencia de un hecho.
#
def _citing_segment(segments, entry):
    return next(
        (
            segment
            for segment in segments
            if segment.attachment is None and set(entry["evidence_ids"]) & set(segment.evidence_ids)
        ),
        None,
    )


#
# Construye las referencias de los fragmentos que la respuesta cita.
#
def cited_references(segments, fragments, extra=()) -> list[DocumentReference]:
    cited = {key for segment in segments for key in segment.document_ids} | set(extra)

    return [_reference(item) for item in fragments if item.evidence_id in cited]


#
# Construye la referencia persistente de un fragmento.
#
def _reference(fragment) -> DocumentReference:
    return DocumentReference(
        evidence_id=fragment.evidence_id,
        chunk_id=fragment.chunk_id,
        document_id=fragment.document_id,
        source_id=fragment.source_id,
        position=fragment.position,
        content_hash=fragment.content_hash,
        file_name=fragment.file_name,
        title=fragment.title,
        kind=fragment.kind,
        page_start=fragment.page_start,
        page_end=fragment.page_end,
        section=fragment.section,
        element_label=fragment.element_label,
        excerpt=fragment.text[:_MAX_EXCERPT],
        doubts=fragment.doubts,
    )


#
# Devuelve las oraciones de los párrafos que se apoyan en documentos, con el párrafo que las contiene.
#
def documented_claims(segments) -> list:
    return [
        (segment, text)
        for segment in segments
        if segment.document_ids and segment.role != "result"
        for text in [*sentences(segment.text), *segment.items]
        if not _states_a_gap(segment, text)
    ]


#
# Devuelve las formas de una oración que se comparan con su evidencia: tal cual y sin la frase que atribuye
# la fuente («Según la OMS, …»), que cambia la redacción pero no lo que se afirma.
#
def claim_variants(text) -> list[str]:
    stripped = ATTRIBUTION_LEAD.sub("", text, count=1).strip()

    return [text, stripped] if stripped and stripped != text and len(stripped.split()) >= _MIN_SENTENCE_WORDS else [text]


#
# Indica si una oración sólo dice qué falta confirmar o qué no figura en los documentos, sin afirmar nada
# documental; un hallazgo negativo («no se encontraron diferencias») sí se audita.
#
def _states_a_gap(segment, text) -> bool:
    pending = segment.role == "hypothesis" and PENDING_CONFIRMATION.search(text) is not None
    absent = EVIDENCE_GAP.search(text) is not None and EVIDENCE_NOUNS.search(text) is not None

    return (pending or absent) and not number_readings(text)


#
# Devuelve las oraciones de la evidencia que respaldan a un párrafo: sus fragmentos, sus hechos y, en una
# hipótesis, las lecturas condicionales que la autorizan.
#
def support_passages(segment, request: WriterRequest) -> list[str]:
    body = request.record.evidence
    fragments = [item["text"] for item in body["documents"] if item["id"] in segment.document_ids]
    facts = [item["text"] for item in body["facts"] if item["id"] in segment.fact_ids]
    readings = request.evidence.hypothesis_values if segment.role == "hypothesis" else []
    lines = [line for text in fragments for line in _lines_and_pairs(text)]

    return [*facts, *readings, *lines]


#
# Devuelve las frases de las oraciones de evidencia, separadas por comas, paréntesis, punto y coma o «y/o»: el ítem
# de una lista traduce una de esas frases («contacto total del cuerpo» es «whole-body contact») y comparado con la
# oración entera, más larga y en otro idioma, parecería no tener respaldo aunque lo tenga.
#
def support_phrases(lines) -> list[str]:
    parts = (part.strip() for line in lines for part in _PHRASE_BREAK.split(line))
    known = set(lines)

    return list(dict.fromkeys(part for part in parts if len(part.split()) >= _MIN_PHRASE_WORDS and part not in known))


#
# Devuelve las oraciones de un fragmento y cada par de oraciones seguidas: una oración puede resumir dos
# consecutivas del mismo fragmento, nunca de fragmentos distintos.
#
def _lines_and_pairs(text) -> list[str]:
    lines = _lines(text)
    clauses = [clause for line in lines for clause in _contrasted(line)]

    return [*lines, *clauses, *(f"{first} {second}" for first, second in zip(lines, lines[1:]))]


#
# Separa una oración que une dos hallazgos con un contraste («…, while the winds and tide had a secondary effect»):
# la paráfrasis de uno solo se compara con ese hallazgo y no con la oración entera, que la diluye.
#
def _contrasted(line) -> list[str]:
    parts = [part.strip() for part in _CONTRAST_SPLIT.split(line)]

    return [part for part in parts if len(part) >= _MIN_PASSAGE_CHARS] if len(parts) > 1 else []


#
# Devuelve cada oración de los fragmentos que cita un párrafo, con el documento, la sección de donde sale
# y los pares que forma con sus oraciones vecinas.
#
def cited_lines(segment, request: WriterRequest) -> list[tuple[str, str, str, tuple[str, ...]]]:
    documents = request.record.evidence["documents"]
    cited = []
    for item in documents:
        if item["id"] not in segment.document_ids:
            continue

        lines = _lines(item["text"])
        heading = (item.get("section") or "").split(" > ")[-1]
        for index, line in enumerate(lines):
            before = [f"{lines[index - 1]} {line}"] if index else []
            after = [f"{line} {lines[index + 1]}"] if index + 1 < len(lines) else []
            cited.append((item["id"], line, heading, (*before, *after)))

    return cited


#
# Separa un fragmento en oraciones y en líneas, para comparar también tablas y fichas.
#
def _lines(text) -> list[str]:
    parts = [part for line in text.splitlines() for part in sentences(line)]

    return [part for part in parts if len(part) >= _MIN_PASSAGE_CHARS]


#
# Indica si una oración se apoya en una ficha o tabla que contiene todos sus números.
#
def listed_values(claim, segment, request: WriterRequest) -> bool:
    documents = request.record.evidence["documents"]
    listed = [
        item["text"]
        for item in documents
        if item["id"] in segment.document_ids and _is_listing(item["text"])
    ]

    return any(_holds_values(claim, text) for text in listed)


#
# Indica si un texto contiene todos los números de una oración.
#
def _holds_values(claim, text) -> bool:
    readings = number_readings(claim)
    found = number_readings(text)

    return bool(readings) and all(
        any(reading.values & item.values for item in found) for reading in readings
    )


#
# Indica si un fragmento es una ficha o una tabla y no prosa comparable oración por oración.
#
def _is_listing(text) -> bool:
    return text.count("|") >= _LISTING_MARKS or text.count("\n-") >= _LISTING_MARKS


#
# Quita del borrador las oraciones sin respaldo, aplica las reescrituras que las acotan (como «según el estudio
# citado») y registra ambas cosas.
#
def apply_support(reviewed: ReviewedDraft, unsupported: list, required=(), rewritten=None) -> ReviewedDraft:
    rewritten = rewritten or {}
    rejected = {id(segment): set() for segment in reviewed.accepted}
    rejections = list(reviewed.rejections)
    for segment, text, *problem in unsupported:
        rejected[id(segment)].add(text)
        rejections.append((segment.role, problem[0] if problem else Rejection("unsupported_claim", text[:120])))

    values = frozenset(value for item in required for value in item.values)
    kept = [
        _without_claims(item, rejected[id(item)], values, _replacements(item, rewritten))
        for item in reviewed.accepted
    ]
    rejections += [
        (item.role, Rejection("lost_presentation", text[:120]))
        for item in reviewed.accepted
        for text in _orphans(item, rejected[id(item)], values)
    ]
    rejections += [
        (item.role, Rejection("scoped_claim", text[:120]))
        for item in reviewed.accepted
        for text in _replacements(item, rewritten)
    ]

    dropped = sum(
        1
        for item, result in zip(reviewed.accepted, kept, strict=True)
        if item.document_ids and item.role != "result" and (result is None or len(result.items) < len(item.items))
    )

    return ReviewedDraft(
        [item for item in kept if item is not None], rejections, reviewed.missing_facts, reviewed.dropped + dropped
    )


#
# Devuelve las reescrituras de las oraciones de un párrafo.
#
def _replacements(segment, rewritten) -> dict:
    return {text: new for (key, text), new in rewritten.items() if key == id(segment)}


#
# Devuelve un párrafo sin las oraciones rechazadas ni las que quedan sueltas, con sus reescrituras, o None si no
# queda nada.
#
def _without_claims(segment, rejected, values=frozenset(), replacements=None):
    replacements = replacements or {}
    if not rejected and not replacements:
        return segment

    dropped = {*rejected, *_orphans(segment, rejected, values)}
    kept, previous = [], None
    for item in sentences(segment.text):
        if item in dropped or (previous in dropped and _continues_clause(item, values)):
            dropped.add(item)
        else:
            kept.append(replacements.get(item, item))
        previous = item
    items = [item for item in segment.items if item not in rejected]
    if not kept:
        return None

    return segment.model_copy(update={"text": _tidied(kept), "items": items})


#
# Indica si un trozo sigue la cláusula anterior, separado sólo por «;» («…; por ejemplo, los vientos del norte…»):
# empieza en minúscula y no trae un valor pedido, así que cae con ella.
#
def _continues_clause(text, values) -> bool:
    return text[:1].islower() and not any(reading.values & values for reading in number_readings(text))


#
# Devuelve lo que queda sin contexto si cae la oración que presentaba la fuente: un valor medido se sostiene
# solo, salvo que remita a lo que se quitó («A este nivel…») y no presente él mismo un valor de referencia
# pedido.
#
def _orphans(segment, rejected, values=frozenset()) -> list[str]:
    lead, *rest = sentences(segment.text) or [""]
    if segment.role != "documented" or lead not in rejected:
        return []

    return [item for item in rest if item not in rejected and not _stands_alone(item, values)]


#
# Indica si una oración se sostiene sin la que la presentaba: informa un valor con su unidad, nombra su propia
# fuente («Según registros fósiles, …») o enuncia algo general en presente («presentan una gran diversidad de
# formas»). No se sostiene si remite a lo quitado («Estos organismos…»), habla de un estudio sin nombrarlo («El
# análisis incluyó…», «Se observó…») o narra en pasado lo que hizo un estudio («Las abundancias se
# transformaron…»): leída sola parecería un análisis propio. Una que remite a lo quitado sólo queda si presenta un
# valor de referencia pedido.
#
def _stands_alone(text, values) -> bool:
    if ANAPHORIC_START.search(text) is not None:
        return _reports_a_measure(text) and any(reading.values & values for reading in number_readings(text))
    if _reports_a_measure(text):
        return True
    if UNNAMED_STUDY_START.search(text) is not None:
        return False
    if ATTRIBUTION_LEAD.match(text) or ATTRIBUTED_SOURCE.search(text):
        return True

    return PAST_NARRATION.search(text) is None


#
# Indica si una oración informa un valor con su unidad.
#
def _reports_a_measure(text) -> bool:
    return any(item.unit for item in number_readings(text))


#
# Quita de los párrafos documentales las citas que no sostienen ninguna de sus oraciones publicadas.
#
def backed_citations(reviewed: ReviewedDraft, request: WriterRequest, support, threshold=0.0) -> ReviewedDraft:
    rejections = list(reviewed.rejections)
    accepted = []
    for segment in reviewed.accepted:
        kept = _backing_documents(segment, request, support, threshold)
        rejections += [
            (segment.role, Rejection("unbacked_citation", key))
            for key in segment.document_ids
            if key not in kept
        ]
        accepted.append(segment.model_copy(update={"document_ids": kept}))

    return ReviewedDraft(accepted, rejections, reviewed.missing_facts, reviewed.dropped)


#
# Devuelve las citas de un párrafo que sostienen alguna de sus oraciones, o todas si ninguna lo hace
# o el párrafo no es documental.
#
def _backing_documents(segment, request: WriterRequest, support, threshold) -> list[str]:
    if segment.role != "documented":
        return segment.document_ids

    texts = {item["id"]: item["text"] for item in request.record.evidence["documents"]}
    backing = set()
    for _, claim in documented_claims([segment]):
        backing |= _claim_documents(claim, segment, request, texts, support, threshold)

    return [key for key in segment.document_ids if key in backing] or segment.document_ids


#
# Devuelve los documentos que sostienen una oración: los que contienen todas sus cifras o, si no tiene
# cifras, los que más se le parecen; un fragmento que sólo trata el mismo tema no queda citado.
#
def _claim_documents(claim, segment, request: WriterRequest, texts, support, threshold) -> set[str]:
    holders = {key for key in segment.document_ids if _holds_values(claim, texts.get(key, ""))}
    if holders:
        return holders

    scores = {
        key: float(support(claim, support_passages(segment.model_copy(update={"document_ids": [key]}), request)))
        for key in segment.document_ids
    }
    best = max(scores.values(), default=0.0)
    if best <= 0 or best < threshold:
        return set()

    return {key for key, score in scores.items() if score >= best - _CITATION_MARGIN}
