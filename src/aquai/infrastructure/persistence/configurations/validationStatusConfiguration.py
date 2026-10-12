"""Catálogo fijo de estados de validación, con nombres únicos y sin autoincremento."""

from sqlalchemy import Column, Integer, String, Table
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import ValidationStatus


def configure(mapper_registry: registry) -> None:
    table = Table(
        "ValidationStatus",
        mapper_registry.metadata,
        Column("id", Integer, primary_key=True, nullable=False, autoincrement=False),
        Column("name", String, nullable=False, unique=True),
    )
    mapper_registry.map_imperatively(
        ValidationStatus,
        table,
        properties={
            "source_recipes": relationship(
                "SourceRecipe", back_populates="validation_status", passive_deletes="all"
            ),
        },
    )
