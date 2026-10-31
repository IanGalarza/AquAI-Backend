"""Escritura de conversaciones y turnos; reserva y cierre atómicos."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from aquai.application.contracts.conversations import TurnReservation
from aquai.application.errors import ConversationConflictError, ConversationNotFoundError
from aquai.domain.models import Conversation, Message, MessageSource
from aquai.infrastructure.persistence.conversationSession import (
    conversation_session, conversation_view, message_statuses, message_view,
)


def _locked_conversation(session, conversation_id):
    statement = (
        select(Conversation).where(Conversation.id == conversation_id).with_for_update()
    )
    conversation = session.scalar(statement)
    if conversation is None:
        raise ConversationNotFoundError("La conversación no existe.")
    return conversation


def _generations(session, conversation_id, names):
    return session.scalars(select(Message).where(
        Message.conversation_id == conversation_id,
        Message.role == "assistant",
        Message.status_id.in_([names["pending"], names["processing"]]),
    ))


def _leased(message, now):
    """Una generación sin plazo o con plazo vigente sigue en curso."""
    deadline = message.content_data.get("lease_expires_at")
    return deadline is None or datetime.fromisoformat(deadline) > now


class SqlAlchemyConversationCommands:
    def __init__(self, context, settings):
        self._context = context
        self._settings = settings

    def create(self, title):
        with conversation_session(self._context, self._settings) as session:
            item = Conversation(title=title or "Nueva conversación")
            session.add(item)
            session.commit()
            return conversation_view(item)

    def rename(self, conversation_id, title):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            conversation.title = title
            conversation.updated_at = datetime.now(timezone.utc)
            session.commit()
            return conversation_view(conversation)

    def delete(self, conversation_id):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            now = datetime.now(timezone.utc)
            generations = _generations(session, conversation_id, message_statuses(session))
            if any(_leased(message, now) for message in generations):
                raise ConversationConflictError(
                    "La conversación tiene una respuesta en generación; esperá a que finalice."
                )
            # Las claves foráneas restringen el borrado: primero las referencias y los mensajes.
            messages = select(Message.id).where(Message.conversation_id == conversation_id)
            session.execute(delete(MessageSource).where(MessageSource.message_id.in_(messages)))
            session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            session.delete(conversation)
            session.commit()

    def begin_turn(self, conversation_id, content, language):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            names = message_statuses(session)
            now = datetime.now(timezone.utc)
            for message in _generations(session, conversation_id, names):
                if _leased(message, now):
                    raise ConversationConflictError(
                        "La conversación ya tiene una respuesta en generación."
                    )
                message.status_id = names["failed"]
                message.updated_at = now
                message.content = "La generación anterior se interrumpió y no pudo completarse."
                message.content_data = {
                    "schema_version": 1, "kind": "generation_failure",
                    "error": {"code": "interrupted_generation", "message": message.content},
                }
            last = session.scalar(
                select(func.max(Message.sequence_number))
                .where(Message.conversation_id == conversation_id)
            ) or 0
            user = Message(
                conversation_id=conversation_id, sequence_number=last + 1,
                role="user", content=content, status_id=names["completed"], created_at=now,
                content_data={
                    "schema_version": 1, "kind": "user_message",
                    "response_language": language,
                },
            )
            deadline = now + timedelta(
                seconds=self._settings.generation_timeout_seconds + 60
            )
            assistant = Message(
                conversation_id=conversation_id, sequence_number=last + 2,
                role="assistant", content="Generando respuesta…",
                status_id=names["processing"], created_at=now,
                content_data={
                    "schema_version": 1, "kind": "generation_in_progress",
                    "user_message_id": str(user.id), "lease_expires_at": deadline.isoformat(),
                },
            )
            session.add_all([user, assistant])
            if last == 0 and conversation.title == "Nueva conversación":
                title = " ".join(content.split())
                conversation.title = title if len(title) <= 80 else title[:77] + "…"
            conversation.last_message_at = conversation.updated_at = now
            session.commit()
            return TurnReservation(
                conversation.id, user.id, assistant.id, [], [], content,
                user_sequence_number=user.sequence_number,
            )

    def finish_turn(self, turn, record):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, turn.conversation_id)
            names = message_statuses(session)
            message = session.get(Message, turn.assistant_message_id)
            if (
                message is None or message.conversation_id != turn.conversation_id
                or message.role != "assistant"
                or message.status_id not in {names["processing"], names["pending"]}
                or message.content_data.get("user_message_id") != str(turn.user_message_id)
            ):
                raise ConversationConflictError("El turno ya finalizó o dejó de admitir resultados.")
            now = datetime.now(timezone.utc)
            message.content = record.content
            message.content_data = record.content_data
            message.status_id = names[record.status_name]
            message.updated_at = now
            sources = [
                MessageSource(
                    message_id=message.id, source_id=item.source_id,
                    dataset_id=item.dataset_id, reference_order=item.reference_order,
                    details=item.details,
                )
                for item in record.sources
            ]
            session.add_all(sources)
            conversation.last_message_at = conversation.updated_at = now
            session.commit()
            statuses = {value: key for key, value in names.items()}
            return message_view(message, statuses, sources)
