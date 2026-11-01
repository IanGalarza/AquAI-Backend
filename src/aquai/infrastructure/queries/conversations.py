from sqlalchemy import exists, func, or_, select
from sqlalchemy.orm import aliased, selectinload

from aquai.application.errors import ConversationNotFoundError
from aquai.domain.models import Conversation, Message, MessageSource
from aquai.infrastructure.persistence.conversationSession import (
    conversation_session,
    conversation_view,
    message_statuses,
    message_view,
)


class SqlAlchemyConversationQueries:
    def __init__(self, context, settings):
        self._context = context
        self._settings = settings

    def list(self, *, page, page_size):
        with conversation_session(self._context, self._settings, read_only=True) as session:
            total = session.scalar(select(func.count()).select_from(Conversation))
            activity = func.coalesce(Conversation.last_message_at, Conversation.created_at)
            statement = (
                select(Conversation)
                .order_by(activity.desc(), Conversation.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            items = [conversation_view(item) for item in session.scalars(statement)]

            return {"items": items, "total": total, "page": page, "page_size": page_size}

    def get(self, conversation_id):
        with conversation_session(self._context, self._settings, read_only=True) as session:
            item = session.get(Conversation, conversation_id)
            if item is None:
                raise ConversationNotFoundError("La conversación no existe.")
            return conversation_view(item)

    def messages(self, conversation_id, *, page, page_size):
        with conversation_session(self._context, self._settings, read_only=True) as session:
            if session.get(Conversation, conversation_id) is None:
                raise ConversationNotFoundError("La conversación no existe.")

            statuses = {value: key for key, value in message_statuses(session).items()}
            scope = Message.conversation_id == conversation_id
            total = session.scalar(select(func.count()).select_from(Message).where(scope))
            statement = (
                select(Message)
                .where(scope)
                .options(selectinload(Message.message_sources))
                .order_by(Message.sequence_number)
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            items = [
                message_view(item, statuses, item.message_sources)
                for item in session.scalars(statement)
            ]

            return {
                "conversation_id": conversation_id,
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
            }

    def context(self, conversation_id, *, before_sequence):
        if not self._settings.history_messages:
            return []

        limit = max(self._settings.history_messages, self._settings.history_recall_messages)
        with conversation_session(self._context, self._settings, read_only=True) as session:
            completed = message_statuses(session)["completed"]
            statement = _context_statement(conversation_id, before_sequence, completed, limit)
            rows = [dict(row) for row in session.execute(statement).mappings()]

            associations = _source_associations(session, rows)
            for index, row in enumerate(rows):
                row["associations"] = associations.get(row["id"], [])
                # La ventana reciente llega siempre; los anteriores sólo si la pregunta los recupera.
                row["recent"] = index < self._settings.history_messages
            return list(reversed(rows))


def _context_statement(conversation_id, before_sequence, completed, limit):
    """Turnos completos recientes: cada pregunta sólo cuenta si ya tiene respuesta completa."""
    user = aliased(Message)
    reply = aliased(Message)
    data = Message.content_data
    previous_user_message = (
        (user.conversation_id == Message.conversation_id)
        & (user.sequence_number == Message.sequence_number - 1)
        & (user.role == "user")
    )
    has_completed_reply = exists(
        select(reply.id).where(
            reply.conversation_id == Message.conversation_id,
            reply.sequence_number == Message.sequence_number + 1,
            reply.role == "assistant",
            reply.status_id == completed,
        )
    )
    return (
        select(
            Message.id,
            Message.role,
            func.left(Message.content, 2000).label("text"),
            (func.length(Message.content) <= 2000).label("text_complete"),
            user.content.label("question"),
            data["schema_version"].label("schema_version"),
            data["response"]["status"].label("answer_status"),
            data["response"]["response_type"].label("response_type"),
            data["response"]["sources"].label("sources"),
            data["response"]["operation_references"].label("references"),
            data["response"]["routing"]["request_history"].label("requests"),
            data["response"]["routing"]["history_message_ids"].label("context_ids"),
            data["response"]["routing"]["data_mentions"].label("mentions"),
            data["response"]["routing"]["source_mentions"].label("named_sources"),
            data["response"]["routing"]["inherits_named_sources"].label("inherited_named_sources"),
            data["response"]["segments"].label("segments"),
            data["response"]["historical_message_ids"].label("replayed_ids"),
            data["operations"].label("operations"),
            data["validated_at"].label("verified_at"),
        )
        .outerjoin(user, previous_user_message)
        .where(
            Message.conversation_id == conversation_id,
            Message.sequence_number < before_sequence,
            Message.status_id == completed,
            Message.role.in_(["user", "assistant"]),
            or_(Message.role == "assistant", has_completed_reply),
        )
        .order_by(Message.sequence_number.desc())
        .limit(limit)
    )


def _source_associations(session, rows):
    message_ids = [row["id"] for row in rows if row["sources"]]
    associations = {}
    if not message_ids:
        return associations

    projection = select(
        MessageSource.message_id,
        MessageSource.source_id,
        MessageSource.dataset_id,
        MessageSource.details["dataset_version"].label("version"),
        MessageSource.details["provenance"].label("provenance"),
        MessageSource.details["columns"].label("columns"),
    ).where(MessageSource.message_id.in_(message_ids))
    for item in session.execute(projection).mappings():
        associations.setdefault(item["message_id"], []).append(dict(item))
    return associations
