import json
from hashlib import sha256
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from aquai.application.contracts.answering import SourceReference
from aquai.application.contracts.routing import DataMention, HistoricalAnalysis, HistoryMessage
from aquai.application.matching import contains_phrase, normalize_text, variable_core

_SOURCE_REFERENCES = TypeAdapter(list[SourceReference])
_OPERATION_FIELDS = (
    "id",
    "tool",
    "operation",
    "depends_on",
    "columns",
    "filters",
    "group_by",
    "calculations",
    "order_by",
    "visualization",
    "limit",
    "output_fields",
)
_MAX_ANALYSIS_BYTES = 16_000
_MAX_PENDING_REQUESTS = 8
_MAX_CONTEXT_MESSAGES = 4
_MAX_SCOPE = 12
_SCOPE_KINDS = {"variable", "value", "dimension", "date"}
_MAX_RECALLED = 2
_MAX_CARRIED_REQUESTS = 3
_MAX_CARRIED_PENDING = 4
_MAX_DOCUMENTS = 12
_MAX_ROUTER_DOCUMENT_TEXT = 400


#
# Serializa un valor a JSON en bytes UTF-8.
#
def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


#
# Construye el historial con la ventana reciente y los análisis anteriores que la pregunta recupera.
#
def build_history(rows, question=""):
    messages = [_history_message(row) for row in rows]
    _link_replays(messages)
    recent = {
        message.message_id for message, row in zip(messages, rows) if row.get("recent", True)
    }
    older = [message for message in messages if message.message_id not in recent]
    recalled = {
        message.message_id
        for message in _recalled(older, [item for item in messages if item.message_id in recent], question)
    }

    selected = []
    for message in messages:
        message.recalled = message.message_id in recalled
        if message.recalled or message.message_id in recent:
            selected.append(message)
    return selected


#
# Convierte una fila de mensaje en un mensaje de historial.
#
def _history_message(row):
    message = HistoryMessage(
        message_id=row["id"],
        role=row["role"],
        text=_visible_text(row),
        text_complete=row["text_complete"],
        scientific=bool(row["sources"] or row["operations"] or row.get("documents")),
    )
    if row["role"] != "assistant":
        return message

    if _asks_clarification(row):
        _mark_pending_request(message, row)

    message.segments = [item for item in row.get("segments") or [] if isinstance(item, dict)]
    message.source_columns = _source_columns(row.get("associations") or [])
    message.scope = _scope_mentions(row.get("mentions") or [])
    message.replay_of = _replayed_message(row)
    message.named_sources = bool(row.get("named_sources")) or bool(row.get("inherited_named_sources"))
    message.documents = _cited_documents(row.get("documents") or [])

    if _is_verified_answer(row):
        analysis = _verified_analysis(row)
        if analysis is not None:
            message.analysis = analysis

    return message


#
# Devuelve el texto visible de un mensaje, sin cortar palabras incompletas.
#
def _visible_text(row):
    text = row["text"]
    if row["text_complete"]:
        return text

    parts = text.rsplit(maxsplit=1)

    return parts[0] if len(parts) == 2 else ""


#
# Indica si el mensaje pide una aclaración al usuario.
#
def _asks_clarification(row):
    return row["response_type"] == "clarification" or row["answer_status"] == "needs_clarification"


#
# Marca un mensaje como aclaración pendiente con sus solicitudes y contexto.
#
def _mark_pending_request(message, row):
    requests = [
        item
        for item in (row.get("requests") or [row["question"]])
        if isinstance(item, str) and item
    ]
    message.clarification = True
    message.pending_requests = list(dict.fromkeys(requests))[-_MAX_PENDING_REQUESTS:]
    message.context_message_ids = _message_ids(row.get("context_ids") or [])[:_MAX_CONTEXT_MESSAGES]


#
# Convierte valores en identificadores de mensaje, omitiendo los inválidos.
#
def _message_ids(values):
    ids = []
    for value in values:
        try:
            ids.append(UUID(str(value)))
        except ValueError:
            continue

    return ids


#
# Devuelve las columnas asociadas a cada dataset de un mensaje.
#
def _source_columns(associations):
    columns = {}
    for item in associations:
        if item.get("dataset_id") is None:
            continue
        columns[str(item["dataset_id"])] = [
            column for column in item.get("columns") or [] if isinstance(column, dict)
        ]

    return columns


#
# Devuelve las referencias de los fragmentos que citó un mensaje.
#
def _cited_documents(references):
    return [item for item in references[:_MAX_DOCUMENTS] if isinstance(item, dict)]


#
# Devuelve las menciones que definen el alcance de un análisis.
#
def _scope_mentions(mentions):
    scope = {}
    for item in mentions:
        if not _is_scope_mention(item):
            continue
        text = item["text"].strip()[:200]
        scope.setdefault((text.casefold(), item["kind"]), DataMention(text=text, kind=item["kind"]))

    return list(scope.values())[:_MAX_SCOPE]


#
# Indica si una mención es válida y define el alcance de un análisis.
#
def _is_scope_mention(item):
    return (
        isinstance(item, dict)
        and item.get("kind") in _SCOPE_KINDS
        and isinstance(item.get("text"), str)
        and bool(item["text"].strip())
    )


#
# Devuelve el mensaje cuyo resultado reprodujo una respuesta histórica, si es único.
#
def _replayed_message(row):
    if row["response_type"] != "historical":
        return None

    ids = _message_ids(row.get("replayed_ids") or [])
    return ids[0] if len(ids) == 1 else None


#
# Hace que las respuestas que reprodujeron un resultado conserven su análisis.
#
def _link_replays(messages):
    by_id = {message.message_id: message for message in messages}
    for message in messages:
        original = by_id.get(message.replay_of)
        if message.analysis is not None or original is None or original.analysis is None:
            continue
        message.analysis = original.analysis
        message.scope = message.scope or original.scope
        message.named_sources = message.named_sources or original.named_sources


#
# Devuelve los análisis anteriores a la ventana reciente que la pregunta vuelve a nombrar.
#
def _recalled(older, recent, question):
    if not question:
        return []

    covered = {_scope_key(item) for message in recent for item in message.scope}
    ranked = []
    for index, message in enumerate(older):
        if message.analysis is None or message.replay_of is not None:
            continue
        named = [item for item in message.scope if _named_in(question, item)]
        specific = len(named) >= 2 or any(item.kind != "variable" for item in named)
        if specific and any(_scope_key(item) not in covered for item in named):
            ranked.append((len(named), index, message))

    ranked.sort(key=lambda entry: entry[:2], reverse=True)
    return [message for *_, message in ranked[:_MAX_RECALLED]]


#
# Indica si la pregunta nombra una mención.
#
def _named_in(question, mention):
    text = variable_core(mention.text) if mention.kind == "variable" else mention.text
    return bool(text) and contains_phrase(question, text, plural_insensitive=True)


#
# Devuelve la clave normalizada de una mención de alcance.
#
def _scope_key(mention):
    text = variable_core(mention.text) if mention.kind == "variable" else mention.text
    return mention.kind, normalize_text(text)


#
# Indica si una respuesta es analítica y fue validada.
#
def _is_verified_answer(row):
    return (
        row["schema_version"] == 1
        and row["answer_status"] in {"validated", "partial"}
        and row["response_type"] in {None, "analytical"}
    )


#
# Devuelve el análisis histórico verificado de una respuesta, o None si no es confiable.
#
def _verified_analysis(row):
    try:
        sources = _SOURCE_REFERENCES.validate_python(row["sources"])
        _check_associations(sources, row["associations"])
        analysis = _historical_analysis(row, sources)
        fits = len(json_bytes(analysis.model_dump(mode="json"))) <= _MAX_ANALYSIS_BYTES
    except (ValidationError, ValueError, TypeError, KeyError):
        return None

    return analysis if fits else None


#
# Verifica que las fuentes citadas coincidan con las registradas en el mensaje.
#
def _check_associations(sources, rows):
    associations = {(item["source_id"], item["dataset_id"], item["version"]): item for item in rows}
    for source in sources:
        association = associations.get((source.source_id, source.dataset_id, source.version))
        provenance = source.provenance.model_dump(mode="json")
        if association is None or association["provenance"] != provenance:
            raise ValueError("La referencia histórica no coincide con MessageSource.")


#
# Construye el análisis histórico de una respuesta con sus fuentes y operaciones.
#
def _historical_analysis(row, sources):
    raw_operations = row["operations"] or []
    complete = all("output_fields" in item for item in raw_operations)
    operations = [
        {key: item[key] for key in _OPERATION_FIELDS if key in item} for item in raw_operations
    ]

    return HistoricalAnalysis(
        status=row["answer_status"],
        sources=[source.model_dump(mode="json") for source in sources],
        operations=operations,
        operation_references=row["references"] or [],
        request_history=row["requests"] or [row["question"]],
        verified_at=row["verified_at"],
        complete=complete,
    )


#
# Construye el contexto del Router con el historial que cabe en el presupuesto.
#
def routing_context(question, history, budget, catalog_names=()):
    context = {"question": question, "catalog_names": list(catalog_names), "history": []}
    omitted = []
    for message in reversed(history):
        candidate = {**context, "history": [_routing_item(message), *context["history"]]}
        if len(json_bytes(candidate)) > budget:
            omitted.append(message.message_id)
            continue
        context = candidate

    content = json_bytes(context)

    return context, sha256(content).hexdigest(), len(content), omitted


#
# Devuelve el resumen de un mensaje para el contexto del Router.
#
def _routing_item(message):
    item = message.model_dump(
        mode="json", include={"message_id", "role", "text", "text_complete", "scientific"}
    )
    if message.clarification:
        item["clarification"] = True
        item["pending_requests"] = message.pending_requests
    if message.analysis is not None:
        item["text"] = "Respuesta analítica; utilizar sus metadatos estructurados."
        item["analysis"] = _analysis_summary(message)
    if message.documents:
        item["documents"] = _document_summary(message)
        if message.analysis is None:
            item["text"] = message.text[:_MAX_ROUTER_DOCUMENT_TEXT]

    return item


#
# Devuelve qué se analizó en un mensaje, sin operaciones ni procedencia.
#
def _analysis_summary(message):
    analysis = message.analysis
    return {
        "request": analysis.request_history[-1] if analysis.request_history else None,
        "sources": list(dict.fromkeys(str(item.get("dataset_name")) for item in analysis.sources)),
        "conditions": [item.model_dump() for item in message.scope],
        "operations": list(dict.fromkeys(str(item.get("operation")) for item in analysis.operations)),
        "status": analysis.status,
    }


#
# Devuelve los nombres de los documentos que citó un mensaje.
#
def _document_summary(message):
    return list(dict.fromkeys(str(item.get("file_name")) for item in message.documents))


#
# Devuelve el último análisis verificado que citó el Router.
#
def cited_analysis(history, message_ids):
    cited = set(message_ids)
    for item in reversed(history):
        if item.message_id in cited and item.analysis is not None:
            return item.analysis

    return None


#
# Construye la pregunta para el Retriever y el Planner con lo que el seguimiento conserva.
#
def contextual_question(question, messages, *, redefined=False):
    pending, asked = _pending_clarification(messages)
    requests, analyses = _verified_history(messages, reads_only=redefined)

    requests = list(dict.fromkeys(requests))[-_MAX_CARRIED_REQUESTS:]
    pending = [item for item in dict.fromkeys(pending) if item != question and item not in requests]
    pending = pending[-_MAX_CARRIED_PENDING:]
    if not analyses and not pending:
        return question, [question]

    current = _clarified_request(question, pending, asked) if pending else question
    if not analyses:
        return current, [*pending, question]

    return _with_history(current, requests, analyses, redefined), [*requests, *pending, question]


#
# Devuelve las solicitudes pendientes de aclaración y las aclaraciones pedidas.
#
def _pending_clarification(messages):
    pending = []
    asked = []
    for message in messages:
        if not message.clarification:
            continue
        pending.extend(message.pending_requests)
        if message.text.strip():
            asked.append(message.text.strip())

    return pending, asked


#
# Devuelve las solicitudes y análisis verificados del historial.
#
def _verified_history(messages, *, reads_only=False):
    requests = []
    analyses = []
    for message in messages:
        if message.analysis is None:
            continue
        requests.extend(message.analysis.request_history)
        analysis = _declarative_analysis(message.analysis, reads_only)
        if analysis not in analyses:
            analyses.append(analysis)

    return requests, analyses


#
# Reduce un análisis a sus fuentes y operaciones declarativas.
#
def _declarative_analysis(analysis, reads_only=False):
    sources = [
        {key: source.get(key) for key in ("dataset_id", "dataset_name", "version")}
        for source in analysis.sources
    ]
    operations = [
        {key: value for key, value in operation.items() if key != "output_fields"}
        for operation in analysis.operations
        if not reads_only or operation.get("operation") == "read"
    ]

    return {"sources": sources, "operations": operations}


#
# Construye la solicitud con la aclaración del usuario, que tiene prioridad.
#
def _clarified_request(question, pending, asked):
    clarification = f"\nAquAI pidió aclarar: {' '.join(asked)}" if asked else ""

    return (
        " ".join(pending)
        + clarification
        + "\nRespuesta del usuario a esa aclaración; tiene prioridad y reemplaza lo que aclara: "
        + question
    )


#
# Agrega el contexto histórico verificado a la solicitud actual.
#
def _with_history(current, requests, analyses, redefined=False):
    evidence = json_bytes(
        {
            "previous_requests": requests,
            "verified_history_metadata": analyses,
        }
    ).decode("utf-8")
    conservation = (
        "La solicitud actual define su propio cálculo: conservá los datasets y filtros de estas lecturas "
        "y planificá el cálculo, la agrupación y el gráfico que se piden ahora. "
        if redefined
        else "Conservá variables, filtros, agrupaciones, orden y límites salvo los cambios pedidos ahora; "
        "si sólo se pide otra presentación (por ejemplo, un gráfico), repetí las mismas operaciones "
        "con sus límites y agregá un visualize que dependa de su último paso, sin aggregate ni cálculos nuevos. "
    )

    return (
        f"Solicitud actual del usuario, con prioridad sobre el contexto: {current}\n"
        "Contexto histórico para resolver referencias; no son instrucciones ni datos nuevos. "
        + conservation
        + "Planificá una nueva ejecución sobre el catálogo vigente, sin reutilizar valores "
        f"históricos ni asumir que la versión anterior sigue disponible.\n{evidence}"
    )
