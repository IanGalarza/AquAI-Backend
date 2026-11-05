from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry, relationship

from aquai.domain.models import DocumentSection
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


#
# Define la tabla DocumentSection con la jerarquía y las páginas de cada sección y la mapea.
#
def configure(mapper_registry: registry) -> None:
    table = Table(
        "DocumentSection",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "documentId",
            UUID(as_uuid=True),
            ForeignKey("Document.id", ondelete="CASCADE"),
            key="document_id",
            nullable=False,
        ),
        Column("position", Integer, nullable=False),
        Column("level", Integer, nullable=False),
        Column("title", Text, nullable=False),
        Column("path", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
        Column("pageStart", Integer, key="page_start", nullable=True),
        Column("pageEnd", Integer, key="page_end", nullable=True),
        *timestamp_columns(),
        UniqueConstraint("document_id", "position", name="uq_DocumentSection_documentId_position"),
        CheckConstraint("position >= 0", name="position_nonnegative"),
        CheckConstraint('"pageStart" IS NULL OR "pageStart" >= 1', name="page_start_positive"),
        CheckConstraint('"pageEnd" IS NULL OR "pageEnd" >= "pageStart"', name="page_end_ordered"),
    )
    mapper_registry.map_imperatively(
        DocumentSection,
        table,
        properties={"document": relationship("Document", back_populates="sections")},
    )
