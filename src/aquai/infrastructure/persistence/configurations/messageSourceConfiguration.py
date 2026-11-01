from sqlalchemy import Column, ForeignKey, ForeignKeyConstraint, Integer, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import MessageSource
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


#
# Define la tabla MessageSource que une mensajes con fuentes y datasets y la mapea.
#
def configure(mapper_registry: registry) -> None:
    table = Table(
        "MessageSource",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "messageId",
            UUID(as_uuid=True),
            ForeignKey("Message.id", ondelete="RESTRICT"),
            key="message_id",
            nullable=False,
            index=True,
        ),
        Column(
            "sourceId",
            UUID(as_uuid=True),
            ForeignKey("Source.id", ondelete="RESTRICT"),
            key="source_id",
            nullable=False,
            index=True,
        ),
        Column("datasetId", UUID(as_uuid=True), key="dataset_id", nullable=True, index=True),
        Column("ReferenceOrder", Integer, key="reference_order", nullable=False),
        Column("details", JSONB(none_as_null=True), nullable=False),
        *timestamp_columns(),
        ForeignKeyConstraint(
            ["dataset_id", "source_id"],
            ["Dataset.id", "Dataset.source_id"],
            name="fk_MessageSource_datasetId_sourceId_Dataset",
            ondelete="RESTRICT",
        ),
    )
    mapper_registry.map_imperatively(
        MessageSource,
        table,
        properties={
            "message": relationship("Message", back_populates="message_sources"),
            "source": relationship(
                "Source",
                back_populates="message_sources",
                foreign_keys=[table.c.source_id],
            ),
            "dataset": relationship(
                "Dataset",
                back_populates="message_sources",
                primaryjoin=(
                    "and_(Dataset.id == foreign(MessageSource.dataset_id), "
                    "Dataset.source_id == MessageSource.source_id)"
                ),
            ),
        },
    )
