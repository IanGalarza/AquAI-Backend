"""Mapeo del catálogo de procesamiento; sus datos iniciales están en statusSeeds."""

from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import ProcessingStatus


def configure(mapper_registry: registry) -> None:
    table = Table(
        "ProcessingStatus",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True, nullable=False, autoincrement=False),
        Column("name", String, nullable=False),
    )
    mapper_registry.map_imperatively(
        ProcessingStatus,
        table,
        properties={
            "sources": relationship(
                "Source", back_populates="processing_status", passive_deletes="all"
            ),
        },
    )
