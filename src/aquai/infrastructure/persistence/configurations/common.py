"""Definiciones de columnas compartidas por las cinco entidades con fechas de creación y actualización."""

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID


def identity_column() -> Column:
    return Column("id", UUID(as_uuid=True), primary_key=True, nullable=False)


def timestamp_columns() -> tuple[Column, Column]:
    return (
        Column("CreatedAt", TIMESTAMP(timezone=True), key="created_at", nullable=False),
        Column("UpdatedAt", TIMESTAMP(timezone=True), key="updated_at", nullable=True),
    )
