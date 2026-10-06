"""Catálogo de conjuntos de datos y clave candidata que asegura su correspondencia con la fuente."""

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Table, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import Dataset
from aquai.infrastructure.persistence.configurations.common import identity_column, timestamp_columns


def configure(mapper_registry: registry) -> None:
    table = Table(
        "Dataset",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "sourceId", UUID(as_uuid=True), ForeignKey("Source.id", ondelete="RESTRICT"),
            key="source_id", nullable=False, index=True,
        ),
        Column("name", String, nullable=False),
        Column("description", Text, nullable=True),
        Column("sourceLocator", JSONB(none_as_null=True), key="source_locator", nullable=False),
        Column("physicalTableName", String, key="physical_table_name", nullable=True),
        Column("version", Integer, nullable=False),
        Column("isActive", Boolean, key="is_active", nullable=False),
        *timestamp_columns(),
        UniqueConstraint("id", "source_id", name="uq_Dataset_id_sourceId"),
    )
    mapper_registry.map_imperatively(
        Dataset,
        table,
        properties={
            "source": relationship("Source", back_populates="datasets"),
            "message_sources": relationship(
                "MessageSource", back_populates="dataset", passive_deletes="all",
                primaryjoin=(
                    "and_(Dataset.id == foreign(MessageSource.dataset_id), "
                    "Dataset.source_id == MessageSource.source_id)"
                ),
            ),
        },
    )
