"""Proyecciones de fuentes y datasets sin mantener sesiones fuera de Infrastructure."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from aquai.application.errors import SourceNotFoundError
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.domain.models import Dataset, Source
from aquai.infrastructure.persistence.dbContext import DbContext


def _source_columns() -> tuple:
    return (
        Source.id, Source.name, Source.original_filename, Source.file_type,
        Source.mime_type, Source.file_size_bytes, Source.content_hash,
        Source.is_active, Source.description, Source.processing_status,
        Source.processing_report, Source.semantic_index_status,
        Source.created_at, Source.updated_at,
    )


def source_snapshot(
    session: Session, source_id: UUID, *, include_content: bool = False,
) -> dict[str, Any] | None:
    columns = _source_columns()
    if include_content:
        columns += (Source.file_content,)
    row = session.execute(select(*columns).where(Source.id == source_id)).mappings().first()
    if row is None:
        return None
    datasets = session.execute(
        select(
            Dataset.id, Dataset.source_id, Dataset.name, Dataset.description,
            Dataset.source_locator, Dataset.physical_table_name, Dataset.version,
            Dataset.is_active, Dataset.row_count, Dataset.columns_metadata,
            Dataset.queryable, Dataset.quality_warnings, Dataset.semantic_description,
            Dataset.semantic_point_id, Dataset.created_at, Dataset.updated_at,
        ).where(Dataset.source_id == source_id).order_by(Dataset.created_at, Dataset.id)
    ).mappings()
    return deepcopy({**dict(row), "datasets": [dict(dataset) for dataset in datasets]})


class SqlAlchemySourceQueries:
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
        processing_status: str | None = None,
    ) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("La página debe ser positiva y el tamaño debe estar entre 1 y 100.")
        predicates = []
        if search and search.strip():
            predicates.append(Source.name.icontains(search.strip(), autoescape=True))
        if processing_status:
            predicates.append(Source.processing_status == ProcessingStatus(processing_status))
        dataset_count = (
            select(func.count(Dataset.id)).where(Dataset.source_id == Source.id)
            .correlate(Source).scalar_subquery().label("dataset_count")
        )
        with self._context.session() as session:
            total = session.scalar(select(func.count(Source.id)).where(*predicates)) or 0
            rows = session.execute(
                select(*_source_columns(), dataset_count).where(*predicates)
                .order_by(Source.created_at.desc(), Source.id.desc())
                .offset((page - 1) * page_size).limit(page_size)
            ).mappings()
            items = deepcopy([dict(row) for row in rows])
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def usable_dataset_ids(self, dataset_ids: list[UUID]) -> set[UUID]:
        """PostgreSQL confirma disponibilidad aun cuando el índice tenga un payload antiguo."""
        if not dataset_ids:
            return set()
        with self._context.session() as session:
            return set(session.scalars(
                select(Dataset.id).join(Source, Source.id == Dataset.source_id).where(
                    Dataset.id.in_(dataset_ids), Dataset.is_active.is_(True),
                    Dataset.queryable.is_(True), Source.is_active.is_(True),
                    Source.processing_status.in_(
                        (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS)
                    ),
                )
            ))
