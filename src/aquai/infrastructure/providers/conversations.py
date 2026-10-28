"""Composición de la API conversacional, reutilizando PostgreSQL y LangGraph."""

from dataclasses import dataclass
from functools import lru_cache

from aquai.application.conversations.snapshot import ConversationSnapshotBuilder
from aquai.application.useCases.conversations.handlers import ConversationHistoryHandler, SendMessageHandler
from aquai.bootstrap import initialize_application
from aquai.infrastructure.config.conversations import ConversationSettings
from aquai.infrastructure.graphs.conversationRunner import LangGraphConversationRunner
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.repositories.conversations import SqlAlchemyConversationRepository


@dataclass(frozen=True)
class ConversationRuntime:
    history: ConversationHistoryHandler
    send: SendMessageHandler


@lru_cache
def get_conversation_settings():
    return ConversationSettings()


@lru_cache
def create_conversation_runtime() -> ConversationRuntime:
    initialize_application()
    settings = get_conversation_settings()
    repository = SqlAlchemyConversationRepository(get_db_context(), settings)
    snapshots = ConversationSnapshotBuilder(max_message_bytes=settings.max_message_bytes,
        max_table_rows=settings.max_table_rows, max_table_bytes=settings.max_table_bytes,
        max_source_details_bytes=settings.max_source_details_bytes)
    return ConversationRuntime(ConversationHistoryHandler(repository), SendMessageHandler(repository,
        LangGraphConversationRunner(), snapshots, timeout_seconds=settings.generation_timeout_seconds))
