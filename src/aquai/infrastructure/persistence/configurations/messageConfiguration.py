from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import Message
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


def configure(mapper_registry: registry) -> None:
    table = Table(
        "Message",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "conversationId",
            UUID(as_uuid=True),
            ForeignKey("Conversation.id", ondelete="RESTRICT"),
            key="conversation_id",
            nullable=False,
        ),
        Column("sequenceNumber", Integer, key="sequence_number", nullable=False),
        Column("role", String, nullable=False),
        Column("content", Text, nullable=False),
        Column("contentData", JSONB(none_as_null=True), key="content_data", nullable=False),
        Column(
            "statusId",
            Integer,
            ForeignKey("MessageStatus.id", ondelete="RESTRICT"),
            key="status_id",
            nullable=False,
            index=True,
        ),
        *timestamp_columns(),
        UniqueConstraint(
            "conversation_id", "sequence_number", name="uq_Message_conversationId_sequenceNumber"
        ),
    )
    mapper_registry.map_imperatively(
        Message,
        table,
        properties={
            "conversation": relationship("Conversation", back_populates="messages"),
            "status": relationship("MessageStatus", back_populates="messages"),
            "message_sources": relationship(
                "MessageSource",
                back_populates="message",
                passive_deletes="all",
                order_by="MessageSource.reference_order",
            ),
        },
    )
