"""Repositorio de las entidades de chat existentes, con transacciones breves."""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from aquai.application.contracts.answering import ConversationMessage
from aquai.application.contracts.conversations import MessageView, TurnReservation
from aquai.application.errors import ConversationConflictError, ConversationNotFoundError, ConversationsUnavailableError
from aquai.domain.models import Conversation, Message, MessageSource, MessageStatus


_DEFAULT_TITLE = "Nueva conversación"


def _conversation_view(item):
    return {key: getattr(item, key) for key in ("id", "title", "last_message_at", "created_at", "updated_at")}


def _message_view(item, statuses, sources=()):
    return MessageView.model_validate({
        **{key: getattr(item, key) for key in ("id", "conversation_id", "sequence_number", "role", "content", "content_data", "created_at", "updated_at")},
        "status": {"id": item.status_id, "name": statuses[item.status_id]},
        "sources": [{key: getattr(source, key) for key in ("id", "source_id", "dataset_id", "reference_order", "details")} for source in sources],
    })


def _statuses(session):
    values = list(session.scalars(select(MessageStatus)))
    names = {item.name: item.id for item in values}
    if len(names) != len(values) or not {"pending", "processing", "completed", "failed", "cancelled"}.issubset(names):
        raise ConversationsUnavailableError("El catálogo de estados de mensajes no es compatible con la generación.")
    return names


def _locked_conversation(session, conversation_id):
    conversation = session.scalar(select(Conversation).where(Conversation.id == conversation_id).with_for_update())
    if conversation is None:
        raise ConversationNotFoundError("La conversación no existe.")
    return conversation


class SqlAlchemyConversationRepository:
    def __init__(self, context, settings):
        self._context, self._settings = context, settings

    @contextmanager
    def _session(self, *, read_only=False):
        with self._context.session() as session:
            if read_only:
                session.connection(execution_options={"isolation_level": "REPEATABLE READ"}).exec_driver_sql("SET TRANSACTION READ ONLY")
            for key, value in (("lock_timeout", self._settings.lock_timeout_ms),
                               ("statement_timeout", self._settings.statement_timeout_ms)):
                session.execute(select(func.pg_catalog.set_config(key, str(value), True)))
            yield session

    def create(self, title):
        with self._session() as session:
            item = Conversation(title=title or _DEFAULT_TITLE)
            session.add(item)
            session.commit()
            return _conversation_view(item)

    def list(self, *, page, page_size):
        with self._session(read_only=True) as session:
            total = session.scalar(select(func.count()).select_from(Conversation))
            activity = func.coalesce(Conversation.last_message_at, Conversation.created_at)
            items = list(session.scalars(select(Conversation).order_by(activity.desc(), Conversation.id.desc())
                .offset((page - 1) * page_size).limit(page_size)))
            return {"items": [_conversation_view(item) for item in items], "total": total, "page": page, "page_size": page_size}

    def get(self, conversation_id):
        with self._session(read_only=True) as session:
            item = session.get(Conversation, conversation_id)
            if item is None:
                raise ConversationNotFoundError("La conversación no existe.")
            return _conversation_view(item)

    def messages(self, conversation_id, *, page, page_size):
        with self._session(read_only=True) as session:
            if session.get(Conversation, conversation_id) is None:
                raise ConversationNotFoundError("La conversación no existe.")
            statuses = {value: key for key, value in _statuses(session).items()}
            total = session.scalar(select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id))
            items = list(session.scalars(select(Message).where(Message.conversation_id == conversation_id)
                .options(selectinload(Message.message_sources)).order_by(Message.sequence_number)
                .offset((page - 1) * page_size).limit(page_size)))
            return {"conversation_id": conversation_id, "items": [_message_view(item, statuses, item.message_sources) for item in items],
                    "total": total, "page": page, "page_size": page_size}

    def begin_turn(self, conversation_id, content, language):
        with self._session() as session:
            conversation = _locked_conversation(session, conversation_id)
            names = _statuses(session)
            now = datetime.now(timezone.utc)
            active = list(session.scalars(select(Message).where(Message.conversation_id == conversation_id,
                Message.role == "assistant", Message.status_id.in_([names["pending"], names["processing"]]))))
            for message in active:
                deadline = message.content_data.get("lease_expires_at") if isinstance(message.content_data, dict) else None
                if deadline is None or datetime.fromisoformat(deadline) > now:
                    raise ConversationConflictError("La conversación ya tiene una respuesta en generación.")
                message.status_id, message.updated_at = names["failed"], now
                message.content = "La generación anterior se interrumpió y no pudo completarse."
                message.content_data = {"schema_version": 1, "kind": "generation_failure", "error": {
                    "code": "interrupted_generation", "message": message.content}}
            last = session.scalar(select(func.max(Message.sequence_number)).where(Message.conversation_id == conversation_id)) or 0
            history_rows = []
            if self._settings.history_messages:
                history_rows = list(session.execute(select(Message.id, Message.role, Message.content).where(
                    Message.conversation_id == conversation_id, Message.status_id == names["completed"],
                    Message.role.in_(["user", "assistant"])).order_by(Message.sequence_number.desc())
                    .limit(self._settings.history_messages)).all())
                history_rows.reverse()
            user = Message(conversation_id=conversation_id, sequence_number=last + 1, role="user", content=content,
                content_data={"schema_version": 1, "kind": "user_message", "response_language": language}, status_id=names["completed"], created_at=now)
            assistant = Message(conversation_id=conversation_id, sequence_number=last + 2, role="assistant", content="Generando respuesta…",
                content_data={"schema_version": 1, "kind": "generation_in_progress", "user_message_id": str(user.id),
                    "lease_expires_at": (now + timedelta(seconds=self._settings.generation_timeout_seconds + 60)).isoformat()},
                status_id=names["processing"], created_at=now)
            session.add_all([user, assistant])
            if last == 0 and conversation.title == _DEFAULT_TITLE:
                title = " ".join(content.split())
                conversation.title = title if len(title) <= 80 else title[:77] + "…"
            conversation.last_message_at = conversation.updated_at = now
            session.commit()
            return TurnReservation(conversation.id, user.id, assistant.id,
                [ConversationMessage(role=row.role, text=row.content[:2000]) for row in history_rows if row.content.strip()],
                [row.id for row in history_rows if row.content.strip()], content)

    def finish_turn(self, turn, record):
        with self._session() as session:
            conversation = _locked_conversation(session, turn.conversation_id)
            names = _statuses(session)
            message = session.get(Message, turn.assistant_message_id)
            if (message is None or message.conversation_id != turn.conversation_id or message.role != "assistant"
                    or message.status_id not in {names["processing"], names["pending"]}
                    or message.content_data.get("user_message_id") != str(turn.user_message_id)):
                raise ConversationConflictError("El turno ya finalizó o dejó de admitir resultados.")
            now = datetime.now(timezone.utc)
            message.content, message.content_data = record.content, record.content_data
            message.status_id, message.updated_at = names[record.status_name], now
            sources = []
            for association in record.sources:
                source = MessageSource(message_id=message.id, source_id=association.source_id,
                    dataset_id=association.dataset_id, reference_order=association.reference_order, details=association.details)
                sources.append(source)
            session.add_all(sources)
            conversation.last_message_at = conversation.updated_at = now
            session.commit()
            return _message_view(message, {value: key for key, value in names.items()}, sources)
