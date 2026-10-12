"""Tabla de fuentes, columnas escalares y mapeos de las propiedades de navegación."""

from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, Integer, String, Table, Text
from sqlalchemy.dialects.postgresql import BYTEA
from sqlalchemy.orm import deferred, registry, relationship

from aquai.domain.models import Source
from aquai.infrastructure.persistence.configurations.common import identity_column, timestamp_columns


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
        Column(
            "processingStatusId", Integer,
            ForeignKey("ProcessingStatus.id", ondelete="RESTRICT"),
            key="processing_status_id", nullable=False, index=True,
        ),
        Column("fileContent", BYTEA, key="file_content", nullable=False),
        *timestamp_columns(),
    )
    mapper_registry.map_imperatively(
        Source,
        table,
        properties={
            # El contenido binario se carga recién cuando se accede a esta propiedad.
            "file_content": deferred(table.c.file_content),
            "processing_status": relationship("ProcessingStatus", back_populates="sources"),
            "source_recipe": relationship(
                "SourceRecipe", back_populates="source", uselist=False, passive_deletes="all",
            ),
            "datasets": relationship("Dataset", back_populates="source", passive_deletes="all"),
            "message_sources": relationship(
                "MessageSource", back_populates="source", passive_deletes="all",
                foreign_keys="MessageSource.source_id",
            ),
        },
    )
