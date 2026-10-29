"""Escritura de conversaciones y turnos; reserva y cierre atómicos."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

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

    def begin_turn(self, conversation_id, content, language):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            names = message_statuses(session)
            now = datetime.now(timezone.utc)
            statement = select(Message).where(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
                Message.status_id.in_([names["pending"], names["processing"]]),
            )
            for message in session.scalars(statement):
                deadline = message.content_data.get("lease_expires_at")
                if deadline is None or datetime.fromisoformat(deadline) > now:
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
