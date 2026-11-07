from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import func, select

from aquai.application.contracts.conversations import MessageView
from aquai.application.conversations.presentation import present_message
from aquai.application.errors import ConversationsUnavailableError
from aquai.domain.models import MessageStatus

_REQUIRED_STATUSES = {"pending", "processing", "completed", "failed", "cancelled"}
_CONVERSATION_FIELDS = ("id", "title", "last_message_at", "created_at", "updated_at")
_MESSAGE_FIELDS = (
    "id",
    "conversation_id",
    "sequence_number",
    "role",
    "content",
    "content_data",
    "created_at",
    "updated_at",
)
_MESSAGE_SOURCE_FIELDS = (
    "id",
    "source_id",
    "dataset_id",
    "reference_order",
    "details",
)


#
# Abre una sesión de conversaciones con timeouts de bloqueo y consulta, opcionalmente de solo lectura.
#
@contextmanager
def conversation_session(context, settings, *, read_only=False):
    with context.session() as session:
        if read_only:
            _begin_read_only(session)
        timeouts = (
            ("lock_timeout", settings.lock_timeout_ms),
            ("statement_timeout", settings.statement_timeout_ms),
        )
        for name, value in timeouts:
            session.execute(select(func.pg_catalog.set_config(name, str(value), True)))
        yield session


#
# Pone la transacción en modo solo lectura con aislamiento de lectura repetible.
#
def _begin_read_only(session):
    connection = session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    connection.exec_driver_sql("SET TRANSACTION READ ONLY")


#
# Devuelve el mapa nombre-id de estados de mensaje y exige que el catálogo esté completo.
#
def message_statuses(session):
    statuses = list(session.scalars(select(MessageStatus)))
    names = {item.name: item.id for item in statuses}
    if len(names) != len(statuses) or not _REQUIRED_STATUSES.issubset(names):
        raise ConversationsUnavailableError("El catálogo de estados no es compatible.")
    return names


#
# Convierte una conversación persistida en un diccionario con los campos públicos.
#
def conversation_view(item):
    return {key: getattr(item, key) for key in _CONVERSATION_FIELDS}


#
# Construye la vista de un mensaje con su estado, sus fuentes, su presentación y si su reserva venció.
#
def message_view(item, statuses, sources=()):
    view = {
        **{key: getattr(item, key) for key in _MESSAGE_FIELDS},
        "status": {"id": item.status_id, "name": statuses[item.status_id]},
        "sources": [_source_view(source) for source in sources],
    }
    return MessageView.model_validate(
        {**view, "presentation": present_message(view), "lease_expired": _lease_expired(view)}
    )


#
# Indica si una respuesta todavía en curso tiene la reserva vencida según el reloj del servidor.
#
def _lease_expired(view):
    deadline = view["content_data"].get("lease_expires_at")
    running = view["role"] == "assistant" and view["status"]["name"] in {"pending", "processing"}

    return running and deadline is not None and datetime.fromisoformat(deadline) <= datetime.now(timezone.utc)


#
# Convierte una fuente asociada a un mensaje en un diccionario con los campos públicos.
#
def _source_view(source):
    return {key: getattr(source, key) for key in _MESSAGE_SOURCE_FIELDS}
