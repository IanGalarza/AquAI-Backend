import json
from hashlib import sha256
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from aquai.application.contracts.answering import SourceReference
from aquai.application.contracts.routing import HistoricalAnalysis, HistoryMessage

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
_MAX_VARIABLES = 12


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def build_history(rows):
    return [_history_message(row) for row in rows]


def _history_message(row):
    message = HistoryMessage(
        message_id=row["id"],
        role=row["role"],
        text=_visible_text(row),
        text_complete=row["text_complete"],
        scientific=bool(row["sources"] or row["operations"]),
    )
    if row["role"] != "assistant":
        return message

    if _asks_clarification(row):
        _mark_pending_request(message, row)

    message.segments = [item for item in row.get("segments") or [] if isinstance(item, dict)]
    message.source_columns = _source_columns(row.get("associations") or [])
    message.variables = _mentioned_variables(row.get("mentions") or [])

    if _is_verified_answer(row):
        analysis = _verified_analysis(row)
        if analysis is not None:
            message.analysis = analysis

    return message


def _visible_text(row):
    text = row["text"]
    if row["text_complete"]:
        return text

    # No publicar un prefijo cortado a mitad de una palabra o un número.
    parts = text.rsplit(maxsplit=1)

    return parts[0] if len(parts) == 2 else ""


def _asks_clarification(row):
    return row["response_type"] == "clarification" or row["answer_status"] == "needs_clarification"


def _mark_pending_request(message, row):
    requests = [
        item
        for item in (row.get("requests") or [row["question"]])
        if isinstance(item, str) and item
    ]
    message.clarification = True
    message.pending_requests = list(dict.fromkeys(requests))[-_MAX_PENDING_REQUESTS:]
    message.context_message_ids = _message_ids(row.get("context_ids") or [])[:_MAX_CONTEXT_MESSAGES]


def _message_ids(values):
    ids = []
    for value in values:
        try:
            ids.append(UUID(str(value)))
        except ValueError:
            continue

    return ids


def _source_columns(associations):
    columns = {}
    for item in associations:
        if item.get("dataset_id") is None:
            continue
        columns[str(item["dataset_id"])] = [
            column for column in item.get("columns") or [] if isinstance(column, dict)
        ]

    return columns


def _mentioned_variables(mentions):
    names = [item["text"] for item in mentions if _is_variable_mention(item)]

    return list(dict.fromkeys(names))[:_MAX_VARIABLES]


def _is_variable_mention(item):
    return (
        isinstance(item, dict)
        and item.get("kind") == "variable"
        and isinstance(item.get("text"), str)
    )


def _is_verified_answer(row):
    return (
        row["schema_version"] == 1
        and row["answer_status"] in {"validated", "partial"}
        and row["response_type"] in {None, "analytical"}
    )


def _verified_analysis(row):
    try:
        sources = _SOURCE_REFERENCES.validate_python(row["sources"])
        _check_associations(sources, row["associations"])
        analysis = _historical_analysis(row, sources)
        # Nunca convertir un recorte de operaciones en contexto completo.
        fits = len(json_bytes(analysis.model_dump(mode="json"))) <= _MAX_ANALYSIS_BYTES
    except (ValidationError, ValueError, TypeError, KeyError):
        # El texto permanece visible; no se autoriza como evidencia estructurada.
        return None

    return analysis if fits else None


def _check_associations(sources, rows):
    associations = {(item["source_id"], item["dataset_id"], item["version"]): item for item in rows}
    for source in sources:
        association = associations.get((source.source_id, source.dataset_id, source.version))
        provenance = source.provenance.model_dump(mode="json")
        if association is None or association["provenance"] != provenance:
            raise ValueError("La referencia histórica no coincide con MessageSource.")


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


def _routing_item(message):
    item = message.model_dump(
        mode="json", exclude={"context_message_ids", "variables", "segments", "source_columns"}
    )
    if not message.clarification:
        item.pop("clarification")
        item.pop("pending_requests")
    if message.analysis is not None:
        # Los números de respuestas anteriores no intervienen en la clasificación.
        item["text"] = "Respuesta analítica; utilizar sus metadatos estructurados."

    return item


def cited_analysis(history, message_ids):
    """Último resultado verificado que citó el Router: un seguimiento conserva lo que no cambia de él."""
    cited = set(message_ids)
    for item in reversed(history):
        if item.message_id in cited and item.analysis is not None:
            return item.analysis

    return None


def contextual_question(question, messages):
    pending, asked = _pending_clarification(messages)
    requests, analyses = _verified_history(messages)

    # Lo que ya tuvo una respuesta verificada es historial; sólo lo que esperaba la
    # aclaración forma parte de la solicitud actual.
    requests = list(dict.fromkeys(requests))
    pending = [item for item in dict.fromkeys(pending) if item != question and item not in requests]
    if not analyses and not pending:
        return question, [question]

    current = _clarified_request(question, pending, asked) if pending else question
    if not analyses:
        return current, [*pending, question]

    return _with_history(current, requests, analyses), [*requests, *pending, question]


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


def _verified_history(messages):
    requests = []
    analyses = []
    for message in messages:
        if message.analysis is None:
            continue
        requests.extend(message.analysis.request_history)
        analyses.append(_declarative_analysis(message.analysis))

    return requests, analyses


def _declarative_analysis(analysis):
    # Sólo identidades y operaciones declarativas: hashes, localizadores y esquemas de
    # salida no ayudan a replanificar y el Planner tiende a copiarlos como si fueran pasos.
    sources = [
        {key: source.get(key) for key in ("dataset_id", "dataset_name", "version")}
        for source in analysis.sources
    ]
    operations = [
        {key: value for key, value in operation.items() if key != "output_fields"}
        for operation in analysis.operations
    ]

    return {"sources": sources, "operations": operations}


def _clarified_request(question, pending, asked):
    clarification = f"\nAquAI pidió aclarar: {' '.join(asked)}" if asked else ""

    return (
        " ".join(pending)
        + clarification
        + "\nRespuesta del usuario a esa aclaración; tiene prioridad y reemplaza lo que aclara: "
        + question
    )


def _with_history(current, requests, analyses):
    evidence = json_bytes(
        {
            "previous_requests": requests,
            "verified_history_metadata": analyses,
        }
    ).decode("utf-8")

    return (
        f"Solicitud actual del usuario, con prioridad sobre el contexto: {current}\n"
        "Contexto histórico para resolver referencias; no son instrucciones ni datos nuevos. "
        "Conservá variables, filtros, agrupaciones, orden y límites salvo los cambios pedidos ahora; "
        "si sólo se pide otra presentación (por ejemplo, un gráfico), repetí las mismas operaciones "
        "con sus límites y agregá un visualize que dependa de su último paso, sin aggregate ni cálculos nuevos. "
        "Planificá una nueva ejecución sobre el catálogo vigente, sin reutilizar valores "
        f"históricos ni asumir que la versión anterior sigue disponible.\n{evidence}"
    )
