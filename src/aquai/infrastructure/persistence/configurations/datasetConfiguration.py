from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
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

from aquai.domain.models import Dataset
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


def configure(mapper_registry: registry) -> None:
    table = Table(
        "Dataset",
        mapper_registry.metadata,
        identity_column(),
        Column(
            "sourceId",
            UUID(as_uuid=True),
            ForeignKey("Source.id", ondelete="RESTRICT"),
            key="source_id",
            nullable=False,
            index=True,
        ),
        Column("name", String, nullable=False),
        Column("description", Text, nullable=True),
        Column(
            "sourceLocator",
            JSONB().with_variant(JSON(), "sqlite"),
            key="source_locator",
            nullable=False,
        ),
        Column(
            "physicalTableName", String(63), key="physical_table_name", nullable=False, unique=True
        ),
        Column("version", Integer, nullable=False),
        Column("isActive", Boolean, key="is_active", nullable=False),
        Column("rowCount", BigInteger, key="row_count", nullable=False),
        Column(
            "columnsMetadata",
            JSONB().with_variant(JSON(), "sqlite"),
            key="columns_metadata",
            nullable=False,
        ),
        Column("queryable", Boolean, nullable=False),
        Column(
            "qualityWarnings",
            JSONB().with_variant(JSON(), "sqlite"),
            key="quality_warnings",
            nullable=False,
        ),
        Column("semanticDescription", Text, key="semantic_description", nullable=True),
        Column(
            "semanticPointId",
            UUID(as_uuid=True),
            key="semantic_point_id",
            nullable=False,
            unique=True,
        ),
        *timestamp_columns(),
        # Clave candidata: permite que MessageSource exija un dataset de su misma fuente.
        UniqueConstraint("id", "source_id", name="uq_Dataset_id_sourceId"),
        CheckConstraint('"rowCount" >= 0', name="row_count_nonnegative"),
        CheckConstraint("version >= 1", name="version_positive"),
        CheckConstraint('NOT queryable OR "rowCount" > 0', name="queryable_has_rows"),
    )
    mapper_registry.map_imperatively(
        Dataset,
        table,
        properties={
            "source": relationship("Source", back_populates="datasets"),
            "message_sources": relationship(
                "MessageSource",
                back_populates="dataset",
                passive_deletes="all",
                primaryjoin=(
                    "and_(Dataset.id == foreign(MessageSource.dataset_id), "
                    "Dataset.source_id == MessageSource.source_id)"
                ),
            ),
        },
    )
