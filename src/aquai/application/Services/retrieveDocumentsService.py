import asyncio
import logging
import re
from dataclasses import dataclass, field
from itertools import zip_longest
from time import perf_counter
from uuid import UUID

from aquai.application.answering.context import response_language
from aquai.application.answering.measures import measured_results
from aquai.application.contracts.retrieval import DocumentFragment, DocumentRetrievalResult
from aquai.application.errors import LLMProviderError, SemanticIndexError
from aquai.application.interfaces.documents import (
    DocumentEvidenceProvider,
    DocumentIndex,
    DocumentLexicalIndex,
    DocumentQueries,
)
from aquai.application.matching import normalize_text, variable_name, variable_variants
from aquai.application.Rules.documents import normalized_text
from aquai.application.Rules.evidence import (
    PRINTED_FIGURE_TEXT,
    REFERENCE_REQUEST_CLAUSE,
    SEARCH_STOPWORDS,
)
from aquai.application.units import counts_organisms

_logger = logging.getLogger(__name__)
_RRF_CONSTANT = 60
_MIN_CHUNK_TOKENS = 13
_MAX_VARIABLES = 2
_MAX_CONTEXT_CHARS = 500
_MAX_FOCUS_CHARS = 700
_SHINGLE_WORDS = 5
_SHORT_WORD = 4
_MIN_ROOT = 4
_FLEXION_CHARS = 2
_DUPLICATE_OVERLAP = 0.8
_CANDIDATE_SPREAD = 2
_ASK_EDGES = re.compile(r"^[\s,;]*(?:y|e|con|and|with)?\s*")
_RETRIEVAL_ERRORS = (SemanticIndexError, LLMProviderError, ValueError, OSError)


@dataclass
class _Request:
    purpose: str
    requests: list[str]
    focus: str
    lexical: list[int]
    context: list[str] = field(default_factory=list)
    document_ids: list[UUID] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    strict_subject: bool = False


class RetrieveDocumentsService:
    #
    # Guarda el índice, las consultas, el asistente de evidencia y los límites de la selección.
    #
    def __init__(
        self,
        index: DocumentIndex,
        queries: DocumentQueries,
        evidence: DocumentEvidenceProvider,
        *,
        lexical: DocumentLexicalIndex | None = None,
        search_limit: int = 40,
        candidates: int = 12,
        fragments: int = 6,
        per_document: int = 3,
        candidate_chars: int = 500,
    ):
        self._index = index
        self._queries = queries
        self._evidence = evidence
        self._lexical = lexical
        self._search_limit = search_limit
        self._candidates = candidates
        self._fragments = fragments
        self._per_document = per_document
        self._candidate_chars = candidate_chars

    #
    # Busca la evidencia documental del turno; un fallo nunca invalida el análisis tabular.
    #
    async def retrieve(self, state) -> DocumentRetrievalResult:
        started = perf_counter()
        if state.routing.asks_citations:
            return DocumentRetrievalResult(purpose="question", status="skipped")

        request = _question_request(state) if state.routing.route == "document" else _result_request(state)
        if request is None:
            return DocumentRetrievalResult(purpose="enrichment", status="skipped")

        result = DocumentRetrievalResult(purpose=request.purpose, status="no_relevant_evidence")
        result.document_ids = request.document_ids
        try:
            await self._search(request, result)
        except _RETRIEVAL_ERRORS as error:
            _logger.warning("Recuperación documental no disponible: %s", type(error).__name__)
            result.status = "unavailable"
            result.warnings.append("No se pudo consultar el índice documental.")

        result.duration_ms = round((perf_counter() - started) * 1000, 3)
        _logger.info(
            "Documentos purpose=%s status=%s candidatos=%s fragmentos=%s ms=%s",
            result.purpose,
            result.status,
            result.candidates,
            len(result.fragments),
            result.duration_ms,
        )

        return result

    #
    # Reúne candidatos por varias vías, selecciona los útiles y los deja listos para citar.
    #
    async def _search(self, request, result):
        searches, subject = await self._terms(request, result)
        result.queries = [text for item in searches for text in (item.spanish, item.english)]
        ranked = await self._ranked(request, searches)
        usable = [item for item in ranked if _usable(item) and _mentions(item, subject)]
        if request.purpose == "enrichment":
            usable = _distinct(usable, _CANDIDATE_SPREAD * self._per_document)
        candidates = usable[: self._candidates]
        result.candidates = len(candidates)
        if not candidates:
            return

        selected = await self._selected(request, candidates, result)
        chosen = _distinct(selected, self._per_document)[: self._fragments]
        result.discarded = len(candidates) - len(chosen)
        result.fragments = [_fragment(item, index) for index, item in enumerate(chosen, 1)]
        result.status = "found" if chosen else "no_relevant_evidence"

    #
    # Pide las búsquedas autónomas en español y en inglés de cada pedido.
    #
    async def _terms(self, request, result):
        context = {"requests": request.requests, "context": request.context}
        draft, usage = await self._evidence.search_terms(context)
        _add_usage(result, usage)

        subject = request.subject if request.strict_subject else [*request.subject, *draft.subject]

        return draft.searches, subject

    #
    # Ordena los fragmentos por fusión de rangos entre las búsquedas densas y léxicas.
    #
    async def _ranked(self, request, searches):
        scoped = request.document_ids or None
        chunks, groups = {}, [[] for _ in searches]
        for group, search in zip(groups, searches):
            for query in dict.fromkeys((search.spanish, search.english)):
                found = await self._index.search(query, self._search_limit, document_ids=scoped)
                chunks.update({item["chunk_id"]: item for item in found})
                group.append([item["chunk_id"] for item in found])

        if self._lexical is not None:
            lexical = await self._lexical_rankings(request, searches, chunks, scoped)
            for position, ranking in zip(request.lexical, lexical):
                groups[position].append(ranking)

        order = (
            _interleaved([_fused(group) for group in groups])
            if request.purpose == "enrichment"
            else _fused([ranking for group in groups for ranking in group])
        )

        return [chunks[key] for key in order if key in chunks]

    #
    # Agrega las búsquedas por términos y verifica los fragmentos que sólo ellas encontraron.
    #
    async def _lexical_rankings(self, request, searches, chunks, scoped):
        rankings = []
        for position in request.lexical:
            item = searches[position]
            query = f"{item.spanish} {item.english}"
            rankings.append(
                await self._lexical.search(query, self._search_limit, document_ids=scoped)
            )

        missing = list({key for ranking in rankings for key in ranking if key not in chunks})
        chunks.update(await asyncio.to_thread(self._queries.verified_chunks, missing))

        return rankings

    #
    # Deja sólo los candidatos que ayudan a responder, en orden de utilidad.
    #
    async def _selected(self, request, candidates, result):
        context = {
            "query": request.focus,
            "fragments": [
                _candidate(item, index, self._candidate_chars)
                for index, item in enumerate(candidates, 1)
            ],
        }
        try:
            order, usage = await self._evidence.select(context)
        except LLMProviderError:
            result.warnings.append("No se pudo evaluar la relevancia; se usó el orden de búsqueda.")
            return candidates

        _add_usage(result, usage)

        return [candidates[item - 1] for item in order]


#
# Indica si un fragmento, su sección, su leyenda o su documento nombran el objeto preguntado.
#
def _mentions(chunk, subject) -> bool:
    place = " ".join(
        part
        for part in (
            chunk["title"],
            *(chunk["section_path"] or []),
            chunk["element_caption"],
            chunk["text"],
        )
        if part
    )
    words = set(normalize_text(place).split())

    return any(_covers(term, words) for term in subject)


#
# Indica si todas las palabras de un término aparecen en un texto, sin importar su flexión.
#
def _covers(term, words) -> bool:
    terms = [word for word in normalize_text(term).split() if word not in SEARCH_STOPWORDS]

    return bool(terms) and all(_present(term, words) for term in terms)


#
# Indica si una palabra figura en un texto: las siglas enteras, el resto por su raíz.
#
def _present(term, words) -> bool:
    if len(term) <= _SHORT_WORD:
        return term in words

    root = _root(term)

    return any(word.startswith(root) for word in words)


#
# Devuelve la parte estable de una palabra: «cyanobacterial» y «cyanobacteria» comparten raíz.
#
def _root(word) -> str:
    return word[: max(_MIN_ROOT, len(word) - _FLEXION_CHARS)]


#
# Devuelve la búsqueda de una pregunta documental con el contexto que la conversación aporta.
#
def _question_request(state) -> _Request:
    routing = state.routing

    context = _conversation_context(state)
    focus = f"{state.question} (contexto: {' | '.join(context)})" if context else state.question

    return _Request(
        purpose="question",
        requests=[state.question],
        focus=focus[:_MAX_FOCUS_CHARS],
        lexical=[0],
        context=context,
        document_ids=routing.document_ids,
        subject=_cited_variables(state),
    )


#
# Devuelve lo que la conversación citada aporta para resolver un seguimiento.
#
def _conversation_context(state) -> list[str]:
    cited = set(state.routing.history_message_ids)
    context = [item for item in state.routing.request_history if item != state.question]
    context += [
        item.text[:_MAX_CONTEXT_CHARS]
        for item in state.conversation_history
        if item.message_id in cited and item.role == "assistant"
    ]

    return context[-4:]


#
# Devuelve las variables del análisis que un seguimiento retoma, con sus nombres equivalentes.
#
def _cited_variables(state) -> list[str]:
    cited = set(state.routing.history_message_ids)
    mentions = [
        scope.text
        for item in state.conversation_history
        if item.message_id in cited
        for scope in item.scope
        if scope.kind == "variable"
    ]

    return [name for mention in mentions for name in variable_variants(variable_name(mention))]


#
# Devuelve la búsqueda que interpreta un resultado validado, o None si no hay nada que interpretar.
#
def _result_request(state) -> _Request | None:
    validation = state.validation
    interpretable = (
        validation is not None
        and validation.status in {"validated", "partial"}
        and validation.usable_evidence
        and state.routing.requested_period is None
    )
    if not interpretable:
        return None

    language = response_language(state.question, state.response_language)
    results = measured_results(state, language)
    charted = state.python_execution is not None and state.python_execution.visualizations
    if not results or not (charted or any(item.values for item in results)):
        return None

    variables = list(dict.fromkeys(_searched_name(item) for item in results))[:_MAX_VARIABLES]
    places = _places(state)
    requests, lexical = [], []
    asked = REFERENCE_REQUEST_CLAUSE.search(state.routing.original_question)
    if asked is not None:
        lexical.append(0)
        requests.append(f"{variables[0]}: {_ASK_EDGES.sub('', asked.group())}")
    for variable in variables:
        lexical.append(len(requests))
        requests.append(f"{variable} {places}".strip())
        requests.append(f"{variable}: significado e implicancias para la calidad del agua")
        lexical.append(len(requests))
        requests.append(f"{variable}: valores de referencia, guías o límites")
    lacks_unit = any(item.lacks_unit for item in results)
    if lacks_unit:
        requests.append(f"{variables[0]}: unidades de medida y rangos habituales")

    return _Request(
        purpose="enrichment",
        requests=requests[:6],
        focus=_focus(variables, places, lacks_unit),
        lexical=[item for item in lexical if item < 6],
        subject=list(dict.fromkeys(name for item in results for name in item.names)),
        strict_subject=True,
    )


#
# Devuelve el nombre con que se busca una variable: los documentos nombran los recuentos por su unidad.
#
def _searched_name(result) -> str:
    return f"{result.variable} ({result.unit})" if counts_organisms(result.unit) else result.variable


#
# Devuelve los lugares y valores que acotan el análisis.
#
def _places(state) -> str:
    mentions = [
        item.text
        for item in state.routing.data_mentions
        if not isinstance(item, str) and item.kind == "value"
    ]

    return " ".join(dict.fromkeys(mentions))


#
# Describe qué evidencia se busca para interpretar los resultados.
#
def _focus(variables, places, lacks_unit=False) -> str:
    units = ", en qué unidades se expresa y qué valores son habituales" if lacks_unit else ""
    where = f" y qué antecedentes hay de {places}" if places else ""

    return (
        f"{', '.join(variables)}: qué mide o representa esta variable en el agua, qué implicancias tiene, "
        f"con qué factores se relaciona, qué valores de referencia existen{units}{where}."
    )


#
# Suma al resultado el modelo y los tokens de una tarea auxiliar.
#
def _add_usage(result, usage):
    result.model = usage.model
    result.input_tokens = (result.input_tokens or 0) + (usage.input_tokens or 0)
    result.output_tokens = (result.output_tokens or 0) + (usage.output_tokens or 0)


#
# Fusiona rankings por su posición recíproca, sin depender de puntajes incomparables.
#
def _fused(rankings) -> list:
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0) + 1 / (_RRF_CONSTANT + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)


#
# Indica si un fragmento tiene contenido citable.
#
def _usable(chunk) -> bool:
    if chunk.get("token_count", _MIN_CHUNK_TOKENS) < _MIN_CHUNK_TOKENS:
        return False

    return not (chunk["kind"] == "figure" and PRINTED_FIGURE_TEXT.match(chunk["text"]))


#
# Construye la entrada de un candidato para evaluar su relevancia.
#
def _candidate(chunk, index, chars) -> dict:
    return {
        "id": index,
        "document": (chunk["title"] or chunk["original_filename"])[:90],
        "section": " > ".join((chunk["section_path"] or [])[-2:])[:90],
        "text": chunk["text"][:chars],
    }


#
# Alterna los rankings de cada pedido, para que ninguno desplace a los demás.
#
def _interleaved(rankings) -> list:
    return list(
        dict.fromkeys(item for items in zip_longest(*rankings) for item in items if item is not None)
    )


#
# Quita los fragmentos casi idénticos y limita los de un mismo documento o tabla.
#
def _distinct(chunks, per_document) -> list:
    kept, seen, counts, elements = [], [], {}, set()
    for chunk in chunks:
        shingles = _shingles(chunk["text"])
        repeated = any(_overlap(shingles, other) >= _DUPLICATE_OVERLAP for other in seen)
        count = counts.get(chunk["document_id"], 0)
        element = (chunk["document_id"], chunk["element_position"])
        if repeated or count >= per_document or element in elements:
            continue

        kept.append(chunk)
        seen.append(shingles)
        counts[chunk["document_id"]] = count + 1
        if chunk["element_position"] is not None:
            elements.add(element)

    return kept


#
# Devuelve las secuencias de palabras de un texto.
#
def _shingles(text) -> set:
    words = normalized_text(text).split()

    return {tuple(words[index : index + _SHINGLE_WORDS]) for index in range(len(words))}


#
# Devuelve la proporción del texto más corto que el otro también contiene.
#
def _overlap(left, right) -> float:
    return len(left & right) / max(1, min(len(left), len(right)))


#
# Construye la evidencia citable de un fragmento verificado.
#
def _fragment(chunk, index) -> DocumentFragment:
    return DocumentFragment(
        evidence_id=f"d{index}",
        chunk_id=chunk["chunk_id"],
        document_id=chunk["document_id"],
        source_id=chunk["source_id"],
        position=chunk["position"],
        kind=str(chunk["kind"]),
        text=chunk["text"],
        title=chunk["title"],
        file_name=chunk["original_filename"],
        content_hash=chunk["content_hash"],
        language=chunk["language"],
        page_start=chunk["page_start"],
        page_end=chunk["page_end"],
        section=chunk["section_path"] or [],
        element_label=chunk["element_label"],
        element_caption=chunk["element_caption"],
        doubts=chunk["doubts"],
        score=chunk.get("score", 0.0),
    )
