from dataclasses import dataclass
from functools import lru_cache

from aquai.application.conversations.snapshot import ConversationSnapshotBuilder
from aquai.application.useCases.ConversationOperation.Commands.createConversationHandler import (
    CreateConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.deleteConversationHandler import (
    DeleteConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.renameConversationHandler import (
    RenameConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.sendMessageHandler import (
    SendMessageHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.getConversationHandler import (
    GetConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.getMessagesHandler import (
    GetMessagesHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.listConversationsHandler import (
    ListConversationsHandler,
)
from aquai.bootstrap import initialize_application
from aquai.infrastructure.commands.conversations import SqlAlchemyConversationCommands
from aquai.infrastructure.config.conversations import ConversationSettings
from aquai.infrastructure.graphs.conversationRunner import LangGraphConversationRunner
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.queries.conversations import SqlAlchemyConversationQueries


@dataclass(frozen=True)
class ConversationRuntime:
    create: CreateConversationHandler
    rename: RenameConversationHandler
    delete: DeleteConversationHandler
    send: SendMessageHandler
    list: ListConversationsHandler
    get: GetConversationHandler
    messages: GetMessagesHandler


#
# Devuelve la configuración de conversaciones, cacheada.
#
@lru_cache
def get_conversation_settings():
    return ConversationSettings()


#
# Construye los manejadores de conversaciones con sus comandos, consultas y el ejecutor del grafo.
#
@lru_cache
def create_conversation_runtime() -> ConversationRuntime:
    initialize_application()
    settings = get_conversation_settings()
    context = get_db_context()
    commands = SqlAlchemyConversationCommands(context, settings)
    queries = SqlAlchemyConversationQueries(context, settings)
    snapshots = ConversationSnapshotBuilder(
        max_message_bytes=settings.max_message_bytes,
        max_table_rows=settings.max_table_rows,
        max_table_bytes=settings.max_table_bytes,
        max_source_details_bytes=settings.max_source_details_bytes,
    )

    return ConversationRuntime(
        create=CreateConversationHandler(commands),
        rename=RenameConversationHandler(commands),
        delete=DeleteConversationHandler(commands),
        send=SendMessageHandler(
            commands,
            queries,
            LangGraphConversationRunner(),
            snapshots,
            timeout_seconds=settings.generation_timeout_seconds,
        ),
        list=ListConversationsHandler(queries),
        get=GetConversationHandler(queries),
        messages=GetMessagesHandler(queries),
    )
