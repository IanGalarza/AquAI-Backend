"""Mapeo del catálogo de mensajes; sus datos iniciales están en statusSeeds."""

from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import MessageStatus


def configure(mapper_registry: registry) -> None:
    table = Table(
        "MessageStatus",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True, nullable=False, autoincrement=False),
        Column("name", String, nullable=False),
    )
    mapper_registry.map_imperatively(
        MessageStatus,
        table,
        properties={
            "messages": relationship(
                "Message", back_populates="status", passive_deletes="all"
            ),
        },
    )
