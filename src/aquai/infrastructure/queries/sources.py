"""Lecturas de fuentes: proyecciones sin entidades ni sesiones fuera de Infrastructure."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aquai.application.errors import SourceNotFoundError
from aquai.domain.models import ProcessingStatus, Source, SourceRecipe, ValidationStatus
from aquai.infrastructure.persistence.dbContext import DbContext


def _metadata_columns() -> tuple:
    return (
        Source.id, Source.name, Source.original_filename, Source.file_type,
        Source.mime_type, Source.file_size_bytes, Source.is_active,
        Source.created_at, Source.updated_at,
        ProcessingStatus.name.label("processing_status"),
        func.coalesce(ValidationStatus.name, "pending").label("validation_status"),
    )


def _source_join(statement):
    return (
        statement.select_from(Source)
        .join(ProcessingStatus, ProcessingStatus.id == Source.processing_status_id)
        .outerjoin(SourceRecipe, SourceRecipe.source_id == Source.id)
        .outerjoin(ValidationStatus, ValidationStatus.id == SourceRecipe.validation_status_id)
    )


def source_snapshot(
    session: Session, source_id: UUID, *, include_content: bool = False,
) -> dict[str, Any] | None:
    """Copia los documentos para que cambios de Application no modifiquen una sesión."""
    columns = (
        *_metadata_columns(), SourceRecipe.id.label("recipe_id"),
        SourceRecipe.recipe_data, SourceRecipe.rules_snapshot, SourceRecipe.validation_report,
    )
    if include_content:
        columns += (Source.file_content,)
    row = session.execute(_source_join(select(*columns)).where(Source.id == source_id)).mappings().first()
    if row is None:
        return None
    snapshot = {
        "id": row["id"],
        "name": row["name"],
        "original_filename": row["original_filename"],
        "file_type": row["file_type"],
        "mime_type": row["mime_type"],
        "file_size_bytes": row["file_size_bytes"],
        "is_active": row["is_active"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "processing_status": row["processing_status"],
        "validation_status": row["validation_status"],
        "recipe_id": row["recipe_id"],
        "recipe_data": row["recipe_data"],
        "rules_snapshot": row["rules_snapshot"],
        "validation_report": row["validation_report"],
    }
    if include_content:
        snapshot["file_content"] = row["file_content"]
    return deepcopy(snapshot)


class SqlAlchemySourceQueries:
    """Consulta el esquema existente sin traer originales ni informes en listados."""

    def __init__(self, context: DbContext) -> None:
        self._context = context

    def get(self, source_id: UUID, *, include_content: bool = False) -> dict[str, Any]:
        with self._context.session() as session:
            snapshot = source_snapshot(session, source_id, include_content=include_content)
            if snapshot is None:
                raise SourceNotFoundError("La fuente no existe.")
            return snapshot

    def list(
        self, *, page: int = 1, page_size: int = 20, search: str | None = None,
        processing_status: str | None = None, validation_status: str | None = None,
    ) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("La página debe ser positiva y el tamaño debe estar entre 1 y 100.")
        predicates = []
        if search:
            # Una búsqueda literal no interpreta % y _ del nombre como comodines.
            predicates.append(Source.name.icontains(search.strip(), autoescape=True))
        if processing_status:
            predicates.append(ProcessingStatus.name == processing_status)
        if validation_status:
            predicates.append(func.coalesce(ValidationStatus.name, "pending") == validation_status)
        statement = _source_join(select(*_metadata_columns())).where(*predicates)
        with self._context.session() as session:
            total = session.scalar(_source_join(select(func.count(Source.id))).where(*predicates))
            rows = session.execute(
                statement.order_by(Source.created_at.desc(), Source.id.desc())
                .offset((page - 1) * page_size).limit(page_size)
            ).mappings()
            items = [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "original_filename": row["original_filename"],
                    "file_type": row["file_type"],
                    "mime_type": row["mime_type"],
                    "file_size_bytes": row["file_size_bytes"],
                    "is_active": row["is_active"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "processing_status": row["processing_status"],
                    "validation_status": row["validation_status"],
                }
                for row in rows
            ]
        return {"items": items, "total": total or 0, "page": page, "page_size": page_size}
