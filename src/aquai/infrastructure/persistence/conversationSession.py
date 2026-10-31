"""Transacciones breves y proyecciones compartidas por queries y commands."""

from contextlib import contextmanager

from sqlalchemy import func, select

from aquai.application.contracts.conversations import MessageView
from aquai.application.conversations.presentation import present_message
from aquai.application.errors import ConversationsUnavailableError
from aquai.domain.models import MessageStatus


@contextmanager
def conversation_session(context, settings, *, read_only=False):
    with context.session() as session:
        if read_only:
            connection = session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        timeouts = (
            ("lock_timeout", settings.lock_timeout_ms),
            ("statement_timeout", settings.statement_timeout_ms),
        )
        for name, value in timeouts:
            session.execute(select(func.pg_catalog.set_config(name, str(value), True)))
        yield session


def message_statuses(session):
    values = list(session.scalars(select(MessageStatus)))
    names = {item.name: item.id for item in values}
    required = {"pending", "processing", "completed", "failed", "cancelled"}
    if len(names) != len(values) or not required.issubset(names):
        raise ConversationsUnavailableError("El catálogo de estados no es compatible.")
    return names


def conversation_view(item):
    fields = ("id", "title", "last_message_at", "created_at", "updated_at")
    return {key: getattr(item, key) for key in fields}


def message_view(item, statuses, sources=()):
    fields = (
        "id", "conversation_id", "sequence_number", "role", "content",
        "content_data", "created_at", "updated_at",
    )
    source_fields = (
        "id", "source_id", "dataset_id", "reference_order", "details",
    )
    view = {
        **{key: getattr(item, key) for key in fields},
        "status": {"id": item.status_id, "name": statuses[item.status_id]},
        "sources": [
            {key: getattr(source, key) for key in source_fields}
            for source in sources
        ],
    }
    return MessageView.model_validate({**view, "presentation": present_message(view)})
