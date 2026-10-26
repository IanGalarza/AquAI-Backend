"""Proyecciones de fuentes y datasets sin mantener sesiones fuera de Infrastructure."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
from uuid import UUID

from sqlalchemy import Column, MetaData, Table, func, inspect, select, tuple_
from sqlalchemy.exc import DBAPIError, NoSuchTableError
from sqlalchemy.orm import Session

from aquai.application.contracts.retrieval import DatasetProvenance, VerifiedDataset
from aquai.application.errors import SourceNotFoundError
from aquai.application.dtos.requests.listSources import SourceSort
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
        processing_status: str | None = None, sort: SourceSort = SourceSort.NEWEST,
    ) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("La página debe ser positiva y el tamaño debe estar entre 1 y 100.")
        predicates = []
        if search and search.strip():
            predicates.append(Source.name.icontains(search.strip(), autoescape=True))
        if processing_status:
            predicates.append(Source.processing_status == ProcessingStatus(processing_status))
        ordering = {
            SourceSort.NEWEST: (Source.created_at.desc(), Source.id.desc()),
            SourceSort.OLDEST: (Source.created_at.asc(), Source.id.asc()),
            SourceSort.NAME_ASC: (func.lower(Source.name).asc(), Source.created_at.desc(), Source.id.desc()),
            SourceSort.NAME_DESC: (func.lower(Source.name).desc(), Source.created_at.desc(), Source.id.desc()),
        }[SourceSort(sort)]
        dataset_count = (
            select(func.count(Dataset.id)).where(Dataset.source_id == Source.id)
            .correlate(Source).scalar_subquery().label("dataset_count")
        )
        with self._context.session() as session:
            total = session.scalar(select(func.count(Source.id)).where(*predicates)) or 0
            rows = session.execute(
                select(*_source_columns(), dataset_count).where(*predicates)
                .order_by(*ordering)
                .offset((page - 1) * page_size).limit(page_size)
            ).mappings()
            items = deepcopy([dict(row) for row in rows])
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def usable_dataset_ids(self, dataset_ids: list[UUID], *,
                           versions: dict[UUID, int] | None = None) -> set[UUID]:
        """Comparte con la recuperación la validación del catálogo y tabla física."""
        return {dataset.dataset_id for dataset in self.verified_datasets(dataset_ids, versions=versions)}

    def verified_datasets(self, dataset_ids: list[UUID], *,
                          versions: dict[UUID, int] | None = None) -> list[VerifiedDataset]:
        """Sólo devuelve datasets vigentes, legibles y con esquema físico coherente."""
        if not dataset_ids:
            return []
        version_filter = [] if versions is None else [tuple_(Dataset.id, Dataset.version).in_(list(versions.items()))]
        with self._context.session() as session:
            rows = session.execute(
                select(
                    Dataset, Source.name, Source.original_filename, Source.file_type,
                    Source.content_hash, Source.processing_status,
                    Source.semantic_index_status, Source.processing_report,
                ).join(Source, Source.id == Dataset.source_id).where(
                    Dataset.id.in_(dataset_ids), Dataset.is_active.is_(True),
                    Dataset.queryable.is_(True), Dataset.row_count > 0, Source.is_active.is_(True),
                    Source.processing_status.in_(
                        (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS)
                    ),
                    *version_filter,
                )
            ).all()
            result = []
            for dataset, name, filename, file_type, digest, processing, index_status, report in rows:
                if not self._physical_dataset_available(session, dataset):
                    continue
                result.append(VerifiedDataset(
                    dataset_id=dataset.id, source_id=dataset.source_id,
                    semantic_point_id=dataset.semantic_point_id,
                    name=dataset.name,
                    description=dataset.semantic_description or dataset.description,
                    physical_table_name=dataset.physical_table_name,
                    verified_version=dataset.version, row_count=dataset.row_count,
                    columns_metadata=deepcopy(dataset.columns_metadata),
                    provenance=DatasetProvenance(
                        source_name=name, original_filename=filename, file_type=file_type,
                        content_hash=digest, source_locator=deepcopy(dataset.source_locator),
                        processing_status=processing, semantic_index_status=index_status,
                    ),
                    warnings=list(dict.fromkeys(dataset.quality_warnings + report.get("warnings", []))),
                ))
            return result

    @staticmethod
    def _physical_dataset_available(session: Session, dataset: Dataset) -> bool:
        if not re.fullmatch(r"dataset_[0-9a-f]{32}", dataset.physical_table_name):
            return False
        expected = {column["sql_name"]: column["sql_type"] for column in dataset.columns_metadata}
        expected.update({"_aquai_row": "BIGINT", "_aquai_source_row": "BIGINT"})
        if dataset.source_locator.get("orientation") == "columns":
            expected["_aquai_source_column"] = "BIGINT"
        try:
            with session.begin_nested():
                connection = session.connection()
                inspector = inspect(connection)
                if not inspector.has_table(dataset.physical_table_name):
                    return False
                columns = inspector.get_columns(dataset.physical_table_name)
                actual = {column["name"]: str(column["type"].compile(dialect=connection.dialect)) for column in columns}
                if any(actual.get(name) != sql_type for name, sql_type in expected.items()):
                    return False
                table = Table(dataset.physical_table_name, MetaData(), *(
                    Column(column["name"], column["type"], quote=True) for column in columns
                ), quote=True)
                probe = select(*(table.c[name].is_not(None) for name in expected)).limit(1)
                return session.execute(probe).first() is not None
        except NoSuchTableError:
            return False
        except DBAPIError as error:
            if getattr(error.orig, "sqlstate", None) in {"42P01", "42703", "42501"}:
                return False
            raise
