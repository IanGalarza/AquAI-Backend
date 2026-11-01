from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select

from aquai.application.contracts.conversations import TurnReservation
from aquai.application.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
)
from aquai.domain.models import Conversation, Message, MessageSource
from aquai.infrastructure.persistence.conversationSession import (
    conversation_session,
    conversation_view,
    message_statuses,
    message_view,
)

_DEFAULT_TITLE = "Nueva conversación"


#
# Devuelve la conversación bloqueada para escritura o falla si no existe.
#
def _locked_conversation(session, conversation_id):
    statement = select(Conversation).where(Conversation.id == conversation_id).with_for_update()
    conversation = session.scalar(statement)
    if conversation is None:
        raise ConversationNotFoundError("La conversación no existe.")
    return conversation


#
# Devuelve las respuestas del asistente aún pendientes o en proceso.
#
def _generations(session, conversation_id, names):
    return session.scalars(
        select(Message).where(
            Message.conversation_id == conversation_id,
            Message.role == "assistant",
            Message.status_id.in_([names["pending"], names["processing"]]),
        )
    )


#
# Indica si una generación sigue en curso por no tener plazo o tenerlo vigente.
#
def _leased(message, now):
    deadline = message.content_data.get("lease_expires_at")
    return deadline is None or datetime.fromisoformat(deadline) > now


#
# Marca como fallidas las generaciones vencidas y rechaza si hay una vigente.
#
def _fail_expired_generations(session, conversation_id, names, now):
    for message in _generations(session, conversation_id, names):
        if _leased(message, now):
            raise ConversationConflictError("La conversación ya tiene una respuesta en generación.")
        message.status_id = names["failed"]
        message.updated_at = now
        message.content = "La generación anterior se interrumpió y no pudo completarse."
        message.content_data = {
            "schema_version": 1,
            "kind": "generation_failure",
            "error": {"code": "interrupted_generation", "message": message.content},
        }


#
# Devuelve el último número de secuencia de los mensajes de la conversación.
#
def _last_sequence_number(session, conversation_id):
    last = session.scalar(
        select(func.max(Message.sequence_number)).where(Message.conversation_id == conversation_id)
    )
    return last or 0


#
# Crea el mensaje del usuario para un turno.
#
def _user_message(conversation_id, sequence_number, content, language, names, now):
    return Message(
        conversation_id=conversation_id,
        sequence_number=sequence_number,
        role="user",
        content=content,
        status_id=names["completed"],
        created_at=now,
        content_data={
            "schema_version": 1,
            "kind": "user_message",
            "response_language": language,
        },
    )


#
# Crea el mensaje provisional del asistente mientras se genera la respuesta.
#
def _assistant_placeholder(conversation_id, sequence_number, user, names, now, deadline):
    return Message(
        conversation_id=conversation_id,
        sequence_number=sequence_number,
        role="assistant",
        content="Generando respuesta…",
        status_id=names["processing"],
        created_at=now,
        content_data={
            "schema_version": 1,
            "kind": "generation_in_progress",
            "user_message_id": str(user.id),
            "lease_expires_at": deadline.isoformat(),
        },
    )


#
# Deriva un título corto a partir del contenido del primer mensaje.
#
def _title_from(content):
    title = " ".join(content.split())
    return title if len(title) <= 80 else title[:77] + "…"


#
# Indica si un mensaje del asistente admite todavía el resultado del turno.
#
def _accepts_result(message, turn, names):
    if message is None:
        return False
    return (
        message.conversation_id == turn.conversation_id
        and message.role == "assistant"
        and message.status_id in {names["processing"], names["pending"]}
        and message.content_data.get("user_message_id") == str(turn.user_message_id)
    )


#
# Crea la referencia a una fuente citada por un mensaje.
#
def _message_source(message_id, item):
    return MessageSource(
        message_id=message_id,
        source_id=item.source_id,
        dataset_id=item.dataset_id,
        reference_order=item.reference_order,
        details=item.details,
    )


class SqlAlchemyConversationCommands:
    #
    # Guarda el contexto de base de datos y la configuración de conversaciones.
    #
    def __init__(self, context, settings):
        self._context = context
        self._settings = settings

    #
    # Crea una conversación con el título indicado o uno por defecto.
    #
    def create(self, title):
        with conversation_session(self._context, self._settings) as session:
            item = Conversation(title=title or _DEFAULT_TITLE)
            session.add(item)
            session.commit()
            return conversation_view(item)

    #
    # Cambia el título de una conversación y devuelve su vista.
    #
    def rename(self, conversation_id, title):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            conversation.title = title
            conversation.updated_at = datetime.now(timezone.utc)
            session.commit()
            return conversation_view(conversation)

    #
    # Elimina una conversación y sus mensajes, salvo que tenga una respuesta en generación.
    #
    def delete(self, conversation_id):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            now = datetime.now(timezone.utc)
            generations = _generations(session, conversation_id, message_statuses(session))
            if any(_leased(message, now) for message in generations):
                raise ConversationConflictError(
                    "La conversación tiene una respuesta en generación; esperá a que finalice."
                )

            messages = select(Message.id).where(Message.conversation_id == conversation_id)
            session.execute(delete(MessageSource).where(MessageSource.message_id.in_(messages)))
            session.execute(delete(Message).where(Message.conversation_id == conversation_id))
            session.delete(conversation)
            session.commit()

    #
    # Reserva un turno guardando el mensaje del usuario y un mensaje provisional del asistente.
    #
    def begin_turn(self, conversation_id, content, language):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, conversation_id)
            names = message_statuses(session)
            now = datetime.now(timezone.utc)
            _fail_expired_generations(session, conversation_id, names, now)

            last = _last_sequence_number(session, conversation_id)
            user = _user_message(conversation_id, last + 1, content, language, names, now)
            deadline = now + timedelta(seconds=self._settings.generation_timeout_seconds + 60)
            assistant = _assistant_placeholder(
                conversation_id, last + 2, user, names, now, deadline
            )
            session.add_all([user, assistant])

            if last == 0 and conversation.title == _DEFAULT_TITLE:
                conversation.title = _title_from(content)
            conversation.last_message_at = now
            conversation.updated_at = now
            session.commit()

            return TurnReservation(
                conversation.id,
                user.id,
                assistant.id,
                [],
                [],
                content,
                user_sequence_number=user.sequence_number,
            )

    #
    # Cierra un turno guardando la respuesta del asistente y sus fuentes.
    #
    def finish_turn(self, turn, record):
        with conversation_session(self._context, self._settings) as session:
            conversation = _locked_conversation(session, turn.conversation_id)
            names = message_statuses(session)
            message = session.get(Message, turn.assistant_message_id)
            if not _accepts_result(message, turn, names):
                raise ConversationConflictError(
                    "El turno ya finalizó o dejó de admitir resultados."
                )

            now = datetime.now(timezone.utc)
            message.content = record.content
            message.content_data = record.content_data
            message.status_id = names[record.status_name]
            message.updated_at = now
            sources = [_message_source(message.id, item) for item in record.sources]
            session.add_all(sources)
            conversation.last_message_at = now
            conversation.updated_at = now
            session.commit()

            statuses = {value: key for key, value in names.items()}
            return message_view(message, statuses, sources)
