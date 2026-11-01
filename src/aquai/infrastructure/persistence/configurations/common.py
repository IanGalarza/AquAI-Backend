from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID


#
# Crea la columna de clave primaria UUID «id» común a las tablas.
#
def identity_column() -> Column:
    return Column("id", UUID(as_uuid=True), primary_key=True, nullable=False)


#
# Crea las columnas de auditoría de fecha de creación y de actualización.
#
def timestamp_columns() -> tuple[Column, Column]:
    return (
        Column("CreatedAt", TIMESTAMP(timezone=True), key="created_at", nullable=False),
        Column("UpdatedAt", TIMESTAMP(timezone=True), key="updated_at", nullable=True),
    )
