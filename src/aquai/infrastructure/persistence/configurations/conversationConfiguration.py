from sqlalchemy import Column, String, Table
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import Conversation
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


#
# Define la tabla Conversation y la mapea al modelo de dominio con sus mensajes.
#
def configure(mapper_registry: registry) -> None:
    table = Table(
        "Conversation",
        mapper_registry.metadata,
        identity_column(),
        Column("title", String, nullable=False),
        Column("lastMessageAt", TIMESTAMP(timezone=True), key="last_message_at", nullable=True),
        *timestamp_columns(),
    )
    mapper_registry.map_imperatively(
        Conversation,
        table,
        properties={
            "messages": relationship(
                "Message",
                back_populates="conversation",
                passive_deletes="all",
                order_by="Message.sequence_number",
            ),
        },
    )
