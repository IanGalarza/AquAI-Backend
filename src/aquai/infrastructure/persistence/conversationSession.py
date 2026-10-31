from contextlib import contextmanager

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


def _begin_read_only(session):
    connection = session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
    connection.exec_driver_sql("SET TRANSACTION READ ONLY")


def message_statuses(session):
    statuses = list(session.scalars(select(MessageStatus)))
    names = {item.name: item.id for item in statuses}
    if len(names) != len(statuses) or not _REQUIRED_STATUSES.issubset(names):
        raise ConversationsUnavailableError("El catálogo de estados no es compatible.")
    return names


def conversation_view(item):
    return {key: getattr(item, key) for key in _CONVERSATION_FIELDS}


def message_view(item, statuses, sources=()):
    view = {
        **{key: getattr(item, key) for key in _MESSAGE_FIELDS},
        "status": {"id": item.status_id, "name": statuses[item.status_id]},
        "sources": [_source_view(source) for source in sources],
    }
    return MessageView.model_validate({**view, "presentation": present_message(view)})


def _source_view(source):
    return {key: getattr(source, key) for key in _MESSAGE_SOURCE_FIELDS}
