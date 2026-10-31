from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    LargeBinary,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import deferred, registry, relationship

from aquai.domain.models import Source
from aquai.infrastructure.persistence.configurations.common import (
    identity_column,
    timestamp_columns,
)


def configure(mapper_registry: registry) -> None:
    table = Table(
        "Source",
        mapper_registry.metadata,
        identity_column(),
        Column("name", String, nullable=False),
        Column("originalFilename", String, key="original_filename", nullable=False),
        Column("fileType", String, key="file_type", nullable=False),
        Column("mimeType", String, key="mime_type", nullable=False),
        Column("fileSizeBytes", BigInteger, key="file_size_bytes", nullable=False),
        Column("isActive", Boolean, key="is_active", nullable=False),
        Column("description", Text, nullable=True),
        Column("contentHash", String(64), key="content_hash", nullable=False, unique=True),
        Column("processingStatus", String(32), key="processing_status", nullable=False, index=True),
        Column(
            "processingReport",
            JSONB().with_variant(JSON(), "sqlite"),
            key="processing_report",
            nullable=False,
        ),
        Column(
            "semanticIndexStatus",
            String(16),
            key="semantic_index_status",
            nullable=False,
            index=True,
        ),
        Column("fileContent", LargeBinary, key="file_content", nullable=False),
        *timestamp_columns(),
        CheckConstraint('"fileSizeBytes" >= 0', name="file_size_nonnegative"),
        CheckConstraint('length("contentHash") = 64', name="content_hash_length"),
        CheckConstraint(
            "\"processingStatus\" IN ('uploaded', 'processing', 'ready', 'ready_with_warnings', 'failed')",
            name="processing_status_values",
        ),
        CheckConstraint(
            "\"semanticIndexStatus\" IN ('pending', 'processing', 'ready', 'failed')",
            name="semantic_index_status_values",
        ),
    )
    mapper_registry.map_imperatively(
        Source,
        table,
        properties={
            "file_content": deferred(table.c.file_content),
            "datasets": relationship("Dataset", back_populates="source", passive_deletes="all"),
            "message_sources": relationship(
                "MessageSource",
                back_populates="source",
                passive_deletes="all",
                foreign_keys="MessageSource.source_id",
            ),
        },
    )
