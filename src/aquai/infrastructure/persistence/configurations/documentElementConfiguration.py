from sqlalchemy import (
    JSON,
    CheckConstraint,
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

from aquai.domain.models import DocumentElement
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


#
# Define la tabla DocumentElement con las tablas y figuras de un documento y la mapea.
#
def configure(mapper_registry: registry) -> None:
    table = Table(
        "DocumentElement",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "documentId",
            UUID(as_uuid=True),
            ForeignKey("Document.id", ondelete="CASCADE"),
            key="document_id",
            nullable=False,
        ),
        Column(
            "sectionId",
            UUID(as_uuid=True),
            ForeignKey("DocumentSection.id", ondelete="SET NULL"),
            key="section_id",
            nullable=True,
            index=True,
        ),
        Column("position", Integer, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("label", String(64), nullable=True),
        Column("caption", Text, nullable=True),
        Column("content", Text, nullable=True),
        Column("cells", JSONB(none_as_null=True).with_variant(JSON(), "sqlite"), nullable=True),
        Column("pageStart", Integer, key="page_start", nullable=True),
        Column("pageEnd", Integer, key="page_end", nullable=True),
        Column("locations", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
        *timestamp_columns(),
        UniqueConstraint("document_id", "position", name="uq_DocumentElement_documentId_position"),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        CheckConstraint("kind IN ('table', 'figure')", name="kind_values"),
        CheckConstraint('"pageStart" IS NULL OR "pageStart" >= 1', name="page_start_positive"),
        CheckConstraint('"pageEnd" IS NULL OR "pageEnd" >= "pageStart"', name="page_end_ordered"),
    )
    mapper_registry.map_imperatively(
        DocumentElement,
        table,
        properties={"document": relationship("Document", back_populates="elements")},
    )
