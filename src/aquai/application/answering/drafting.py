import json
from dataclasses import dataclass, field
from hashlib import sha256

from aquai.application.answering.measures import measured_results
from aquai.application.answering.readings import number_readings, sentences, without_numbers
from aquai.application.answering.verification import DraftEvidence, DraftVerifier, Rejection
from aquai.application.answering.wording import file_stem, sentence
from aquai.application.contracts.answering import (
    AnswerSegment,
    DocumentReference,
    ResponseContextRecord,
)
from aquai.application.execution import decode_rows, execution_records
from aquai.application.Rules.evidence import (
    ANAPHORIC_START,
    COMPARISON_DIRECTIONS,
    EVIDENCE_GAP,
    LEADING_CONNECTOR,
    PENDING_CONFIRMATION,
)

_REQUIRED_KINDS = {"result"}
_KEPT_KINDS = {"scope", "detail"}
_MAX_EXCERPT = 600
_MISSING_INTERPRETATION = (
    "falta un párrafo documented: parafraseá de cerca lo que dicen los fragmentos sobre esta variable"
)
_MISSING_HYPOTHESIS = (
    "falta un párrafo hypothesis: con las lecturas de ambiguities, en condicional, citando sus documentos"
)
_MIN_SENTENCE_WORDS = 6
_MIN_PASSAGE_CHARS = 10
_LISTING_MARKS = 3
_MAX_RESTORED_COMPARISONS = 2
_MAX_FACT_VALUES = 400
_MAX_PREVIOUS_ANSWER = 1500
_CORRECTIONS = {
    "unsupported_place": "el fragmento no nombra ese lugar: atribuí el dato al estudio o sistema que sí nombra",
    "unsupported_unit": "copiá la unidad tal como figura junto a ese número en la evidencia",
    "unsupported_number": "ese número no está en lo que cita el párrafo",
    "unsupported_result_word": "un párrafo result sólo repite su fact; lo demás va en otro párrafo",
    "observed_in_documented": "un párrafo documented habla de los documentos, no de los datos analizados",
    "unsupported_claim": "reescribí esa oración más cerca del texto del fragmento",
    "unsupported_comparison": "no compares lo observado sin un fact de kind comparison",
    "unsupported_verdict": "no emitas un juicio sobre lo observado",
    "unsupported_classification": "no clasifiques lo observado",
    "unsupported_causality": "no afirmes causas de lo observado",
    "unsupported_attribution": "atribuí cada dato sólo a la fuente del fragmento que lo contiene",
    "unsupported_frequency": "no agregues frecuencia ni generalidad: decí de qué estudio y sitio es el dato",
    "internal_term": "no uses términos internos de la evidencia",
    "dangling_reference": "ese párrafo remitía a otro que se quitó: escribilo de forma autónoma",
}
_ROLE_LABELS = {"es": "Hipótesis, no confirmada:", "en": "Unconfirmed hypothesis:"}
_ATTACHMENT_LEADS = {"es": "Detalle de ese resultado:", "en": "Detail of that result:"}


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

    #
    # Indica si conviene publicar el borrador: resultados completos y evidencia documental en uso.
    #
    def sufficient(self, evidence: DraftEvidence) -> bool:
        return (
            bool(self.accepted)
            and not self.missing_facts
            and (self.interprets or not evidence.documents)
            and (self.reads_ambiguity or not evidence.allows_hypotheses)
        )

    #
    # Indica si el borrador explica algo con documentos, más allá de contrastar el resultado.
    #
    @property
    def interprets(self) -> bool:
        return any(item.document_ids and item.role == "documented" for item in self.accepted)

    #
    # Indica si el borrador da una lectura condicional con sus valores, no sólo un aviso.
    #
    @property
    def reads_ambiguity(self) -> bool:
        return any(item.role == "hypothesis" and number_readings(item.text) for item in self.accepted)

    #
    # Devuelve qué tan útil es el borrador para elegir entre intentos.
    #
    @property
    def score(self) -> tuple:
        documented = sum(item.role == "documented" for item in self.accepted)

        return not self.missing_facts, documented, -len(self.rejections)


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
    body = {
        "question": state.routing.original_question if state.routing else state.question,
        "language": language,
        "status": context.evidence["status"],
        "facts": _facts(catalog),
        "ambiguities": [item.model_dump(mode="json") for item in ambiguities],
        "notes": list(notes),
        "documents": [],
    }
    kept = _fitted_documents(body, _usable_fragments(state), budget)
    available = {item.evidence_id for item in kept}
    contrasted = [item for item in comparisons if item.document_id in available]
    legible = [item for item in ambiguities if set(item.evidence_ids) & available]
    catalog.update(_comparison_entries(contrasted))
    body["facts"] = _facts(catalog, without_definitions=bool(kept))
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
        hypothesis_values=_hypothesis_texts(legible),
        question=body["question"],
        result_names=[name for item in measured_results(state, language) for name in item.names],
        allows_hypotheses=bool(legible),
        language=language,
        places=_asked_places(state),
        document_contexts={item.evidence_id: _fragment_context(item) for item in kept},
    )

    return WriterRequest(_record(body, budget, context.omitted_rows), evidence, catalog, legible)


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
def documental_request(state, language, budget, previous, analyses=()):
    body = {
        "question": state.question,
        "language": language,
        "facts": [],
        "comparisons": [],
        "ambiguities": [],
        "previous_answers": [item[:_MAX_PREVIOUS_ANSWER] for item in previous],
        "notes": [],
        "documents": [],
    }
    kept = _fitted_documents(body, _usable_fragments(state), budget)
    catalog = _retaken_facts(analyses, kept)
    body["facts"] = _facts(catalog)
    evidence = DraftEvidence(
        facts={item["id"]: item["text"] for item in body["facts"]},
        fact_documents=_fact_documents(catalog),
        fact_directions=_fact_directions(catalog),
        documents={item.evidence_id: item.text for item in kept},
        document_languages={item.evidence_id: item.language for item in kept},
        doubtful_documents={item.evidence_id for item in kept if item.doubts},
        document_labels={item.evidence_id: fragment_label(item) for item in kept},
        question=state.question,
        previous_answers=body["previous_answers"],
        language=language,
        places=[item.text for message in analyses for item in message.scope if item.kind == "value"],
        document_contexts={item.evidence_id: _fragment_context(item) for item in kept},
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
# Devuelve el hecho de un párrafo retomado: un resultado, o un contraste con su documento a la vista.
#
def _retaken_entry(segment, documents):
    text = segment.get("text", "")
    if segment.get("role") == "result":
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
# Devuelve los lugares y valores por los que se acotó la consulta.
#
def _asked_places(state) -> list[str]:
    mentions = state.routing.data_mentions if state.routing else []

    return [item.text for item in mentions if not isinstance(item, str) and item.kind == "value"]


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
# Conserva de un párrafo documental rechazado las oraciones que sí se pueden respaldar.
#
def _salvaged(segment, verifier, lost=False):
    if segment.role == "result":
        return None

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
    if not reviewed.interprets:
        problems.append(_MISSING_INTERPRETATION)
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
    segments = [_segment(item, catalog, language) for item in kept]
    restored = [_restored(catalog[key]) for key in _facts_to_restore(reviewed, kept, catalog)]
    published = _with_attachments([*restored, *segments], reviewed.accepted, catalog, language)
    published.extend(_unused_comparisons(reviewed.accepted, catalog))
    if request.ambiguities and not reviewed.reads_ambiguity:
        published.extend(_written_hypotheses(request.ambiguities, language))

    return published


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
        items=[entry.strip() for entry in item.items if entry.strip()],
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        document_ids=list(dict.fromkeys(item.document_ids)),
    )


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
        text=_ATTACHMENT_LEADS[language],
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
# Indica si una oración sólo dice qué falta confirmar o qué no figura, sin afirmar nada documental.
#
def _states_a_gap(segment, text) -> bool:
    pending = segment.role == "hypothesis" and PENDING_CONFIRMATION.search(text) is not None

    return (pending or EVIDENCE_GAP.search(text) is not None) and not number_readings(text)


#
# Devuelve las oraciones de la evidencia que respaldan a un párrafo: sus fragmentos y sus hechos.
#
def support_passages(segment, request: WriterRequest) -> list[str]:
    body = request.record.evidence
    fragments = [item["text"] for item in body["documents"] if item["id"] in segment.document_ids]
    facts = [item["text"] for item in body["facts"] if item["id"] in segment.fact_ids]
    lines = [line for text in fragments for line in _lines(text)]

    return [*facts, *lines]


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
# Quita del borrador las oraciones sin respaldo y registra cuáles fueron.
#
def apply_support(reviewed: ReviewedDraft, unsupported: list) -> ReviewedDraft:
    rejected = {id(segment): set() for segment in reviewed.accepted}
    rejections = list(reviewed.rejections)
    for segment, text in unsupported:
        rejected[id(segment)].add(text)
        rejections.append((segment.role, Rejection("unsupported_claim", text[:120])))

    kept = [_without_claims(item, rejected[id(item)]) for item in reviewed.accepted]
    rejections += [
        (item.role, Rejection("lost_presentation", text[:120]))
        for item in reviewed.accepted
        for text in _orphans(item, rejected[id(item)])
    ]

    return ReviewedDraft(
        [item for item in kept if item is not None], rejections, reviewed.missing_facts
    )


#
# Devuelve un párrafo sin las oraciones rechazadas ni las que quedan sueltas, o None si no queda nada.
#
def _without_claims(segment, rejected):
    if not rejected:
        return segment

    dropped = {*rejected, *_orphans(segment, rejected)}
    kept = [item for item in sentences(segment.text) if item not in dropped]
    items = [item for item in segment.items if item not in rejected]
    if not kept:
        return None

    return segment.model_copy(update={"text": _tidied(kept), "items": items})


#
# Devuelve lo que queda sin contexto si cae la oración que presentaba la fuente: un valor medido se sostiene solo.
#
def _orphans(segment, rejected) -> list[str]:
    lead, *rest = sentences(segment.text) or [""]
    if segment.role != "documented" or lead not in rejected:
        return []

    return [item for item in rest if item not in rejected and not _reports_a_measure(item)]


#
# Indica si una oración informa un valor con su unidad.
#
def _reports_a_measure(text) -> bool:
    return any(item.unit for item in number_readings(text))


#
# Quita de los párrafos documentales las citas que no sostienen ninguna de sus oraciones publicadas.
#
def backed_citations(reviewed: ReviewedDraft, request: WriterRequest, resembles) -> ReviewedDraft:
    rejections = list(reviewed.rejections)
    accepted = []
    for segment in reviewed.accepted:
        kept = _backing_documents(segment, request, resembles)
        rejections += [
            (segment.role, Rejection("unbacked_citation", key))
            for key in segment.document_ids
            if key not in kept
        ]
        accepted.append(segment.model_copy(update={"document_ids": kept}))

    return ReviewedDraft(accepted, rejections, reviewed.missing_facts)


#
# Devuelve las citas de un párrafo que lo sostienen, o todas si ninguna lo hace o no es documental.
#
def _backing_documents(segment, request: WriterRequest, resembles) -> list[str]:
    if segment.role != "documented":
        return segment.document_ids

    claims = [text for _, text in documented_claims([segment])]
    texts = {item["id"]: item["text"] for item in request.record.evidence["documents"]}
    backing = [
        key
        for key in segment.document_ids
        if any(_backs(claim, key, texts.get(key, ""), segment, request, resembles) for claim in claims)
    ]

    return backing or segment.document_ids


#
# Indica si un documento sostiene una oración: contiene sus cifras o se le parece como pide la auditoría.
#
def _backs(claim, key, text, segment, request: WriterRequest, resembles) -> bool:
    cited = segment.model_copy(update={"document_ids": [key]})

    return _holds_values(claim, text) or resembles(claim, support_passages(cited, request))
