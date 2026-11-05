from sqlalchemy import JSON, CheckConstraint, Column, ForeignKey, Integer, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import Document
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


#
# Define la tabla Document, una por fuente documental procesada, y la mapea.
#
def configure(mapper_registry: registry) -> None:
    table = Table(
        "Document",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "sourceId",
            UUID(as_uuid=True),
            ForeignKey("Source.id", ondelete="RESTRICT"),
            key="source_id",
            nullable=False,
            unique=True,
        ),
        Column("title", Text, nullable=True),
        Column("language", String(8), nullable=True),
        Column("pageCount", Integer, key="page_count", nullable=True),
        Column("textHash", String(64), key="text_hash", nullable=False, index=True),
        Column(
            "duplicateOfSourceId",
            UUID(as_uuid=True),
            ForeignKey("Source.id", ondelete="SET NULL"),
            key="duplicate_of_source_id",
            nullable=True,
        ),
        Column("details", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
        *timestamp_columns(),
        CheckConstraint('length("textHash") = 64', name="text_hash_length"),
        CheckConstraint('"pageCount" IS NULL OR "pageCount" >= 1', name="page_count_positive"),
    )
    mapper_registry.map_imperatively(
        Document,
        table,
        properties={
            "source": relationship("Source", foreign_keys=[table.c.source_id]),
            "sections": relationship(
                "DocumentSection", back_populates="document", passive_deletes="all"
            ),
            "elements": relationship(
                "DocumentElement", back_populates="document", passive_deletes="all"
            ),
            "chunks": relationship(
                "DocumentChunk", back_populates="document", passive_deletes="all"
            ),
        },
    )
