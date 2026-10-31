"""Proyecciones de fuentes y datasets sin mantener sesiones fuera de Infrastructure."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import re
from typing import Any
from uuid import UUID

from sqlalchemy import Column, MetaData, Table, Text, func, inspect, literal, or_, select, tuple_, union_all
from sqlalchemy.exc import DBAPIError, NoSuchTableError
from sqlalchemy.orm import Session

from aquai.application.contracts.retrieval import DatasetProvenance, VerifiedDataset
from aquai.application.contracts.routing import CatalogDataset
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
    def __init__(self, context: DbContext, *, session: Session | None = None) -> None:
        self._context = context
        self._verification_session = session

    def matching_datasets(self, reference: str) -> list[CatalogDataset]:
        """Resuelve identidades por nombres/UUID; las descripciones no son nombres."""
        reference = reference.strip()
        fields = (
            Source.name, Source.original_filename, Dataset.name,
        )
        predicates = [field.icontains(reference, autoescape=True) for field in fields]
        try:
            identifier = UUID(reference)
        except ValueError:
            identifier = None
        if identifier is not None:
            predicates.extend([Source.id == identifier, Dataset.id == identifier])
        statement = (
            select(
                Dataset.id, Dataset.source_id, Dataset.name, Dataset.version,
                Dataset.is_active, Dataset.queryable, Dataset.row_count,
                Source.name.label("source_name"), Source.original_filename,
                Source.is_active.label("source_active"), Source.processing_status,
            )
            .join(Source, Source.id == Dataset.source_id)
            .where(or_(*predicates)).order_by(Source.id, Dataset.id).limit(101)
        )
        with self._context.session() as session:
            rows = session.execute(statement).mappings()
            result = []
            for row in rows:
                exact_names = (
                    row["source_name"], row["original_filename"], row["name"],
                    str(row["id"]), str(row["source_id"]),
                )
                available = (
                    row["is_active"] and row["queryable"] and row["row_count"] > 0
                    and row["source_active"] and row["processing_status"] in {
                        ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS,
                    }
                )
                result.append(CatalogDataset(
                    dataset_id=row["id"], source_id=row["source_id"],
                    dataset_name=row["name"], source_name=row["source_name"],
                    version=row["version"], available=available,
                    exact=any(name.casefold() == reference.casefold() for name in exact_names),
                    name_match=any(
                        reference.casefold() in name.casefold() for name in exact_names
                    ),
                ))
            return result

    def catalog_overview(self) -> list[dict[str, Any]]:
        """Inventario real de fuentes y datasets; nunca lee contenidos ni registros."""
        with self._context.session() as session:
            sources = session.execute(
                select(
                    Source.id, Source.name, Source.original_filename, Source.file_type,
                    Source.is_active, Source.processing_status, Source.semantic_index_status,
                    Source.created_at,
                ).order_by(Source.created_at, Source.id)
            ).mappings().all()
            datasets = session.execute(
                select(
                    Dataset.id, Dataset.source_id, Dataset.name, Dataset.version,
                    Dataset.row_count, Dataset.is_active, Dataset.queryable,
                    Dataset.columns_metadata,
                ).order_by(Dataset.created_at, Dataset.id)
            ).mappings().all()
        by_source: dict[UUID, list[dict[str, Any]]] = {}
        for row in datasets:
            by_source.setdefault(row["source_id"], []).append(row)
        result = []
        for source in sources:
            ready = source["processing_status"] in {
                ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS,
            }
            result.append({
                "source_id": source["id"], "name": source["name"],
                "original_filename": source["original_filename"], "file_type": source["file_type"],
                "is_active": source["is_active"], "processing_status": str(source["processing_status"]),
                "semantic_index_status": str(source["semantic_index_status"]),
                "created_at": source["created_at"],
                "datasets": [{
                    "dataset_id": row["id"], "name": row["name"], "version": row["version"],
                    "row_count": row["row_count"], "is_active": row["is_active"],
                    "available": bool(row["is_active"] and row["queryable"] and row["row_count"] > 0
                                      and source["is_active"] and ready),
                    "columns": [{key: column.get(key) for key in ("sql_name", "original_name", "sql_type", "unit")}
                                for column in row["columns_metadata"]],
                } for row in by_source.get(source["id"], [])],
            })
        return deepcopy(result)

    @contextmanager
    def _verified_session(self):
        if self._verification_session is not None:
            yield self._verification_session
        else:
            with self._context.session() as session:
                yield session

    def matching_column_values(self, dataset_id: UUID, version: int, column: str, value: str) -> list[str]:
        """Consulta literales reales; nunca interpreta un valor como nombre de fuente."""
        datasets = self.verified_datasets([dataset_id], versions={dataset_id: version})
        if not datasets or not any(item.sql_name == column and item.sql_type == "TEXT"
                                   for item in datasets[0].columns_metadata):
            return []
        table = Table(datasets[0].physical_table_name, MetaData(), Column(column, Text))
        field = table.c[column]
        with self._context.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.execute(select(func.pg_catalog.set_config("statement_timeout", "2000", True)))
            exact = connection.scalar(select(field).where(field == value).limit(1))
            if exact is not None:
                return [exact]
            return list(connection.scalars(select(field).where(
                func.pg_catalog.lower(field) == func.pg_catalog.lower(value)).distinct().limit(2)))

    def find_text_values(self, datasets: list[VerifiedDataset], terms: list[str],
                         *, per_column: int = 50) -> dict[UUID, dict[str, list[dict[str, Any]]]]:
        """Valores TEXT reales que contienen cada término, sin distinguir mayúsculas ni acentos.

        Devuelve candidatos amplios; la aplicación confirma la coincidencia por palabras.
        """
        accents, plain = "áàäâãéèëêíìïîóòöôõúùüûñç", "aaaaaeeeeiiiiooooouuuunc"
        result: dict[UUID, dict[str, list[dict[str, Any]]]] = {}
        patterns = {}
        for term in terms:
            words = re.findall(r"\w+", term.casefold().translate(str.maketrans(accents, plain)))
            if words:
                patterns[term] = "%" + "%".join(words) + "%"
        if not patterns:
            return result
        with self._context.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.execute(select(func.pg_catalog.set_config("statement_timeout", "3000", True)))
            for dataset in datasets:
                columns = [item.sql_name for item in dataset.columns_metadata if item.sql_type == "TEXT"]
                if not columns or not re.fullmatch(r"dataset_[0-9a-f]{32}", dataset.physical_table_name):
                    continue
                table = Table(dataset.physical_table_name, MetaData(), *(Column(name, Text) for name in columns))
                for term, pattern in patterns.items():
                    parts = []
                    for name in columns:
                        field = table.c[name]
                        normalized = func.pg_catalog.translate(func.pg_catalog.lower(field), accents, plain)
                        parts.append(
                            select(literal(name).label("column"), field.label("value"), func.count().label("rows"))
                            .where(normalized.like(pattern)).group_by(field).limit(per_column)
                        )
                    for row in connection.execute(union_all(*parts)).mappings():
                        result.setdefault(dataset.dataset_id, {}).setdefault(term, []).append(dict(row))
        return result

    def distinct_text_values(self, dataset: VerifiedDataset, columns: list[str],
                             *, limit: int = 500) -> dict[str, list[str]]:
        """Valores distintos acotados de columnas TEXT verificadas, para perfilar su formato."""
        names = [item.sql_name for item in dataset.columns_metadata
                 if item.sql_type == "TEXT" and item.sql_name in columns]
        if not names or not re.fullmatch(r"dataset_[0-9a-f]{32}", dataset.physical_table_name):
            return {}
        table = Table(dataset.physical_table_name, MetaData(), *(Column(name, Text) for name in names))
        result = {}
        with self._context.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.execute(select(func.pg_catalog.set_config("statement_timeout", "3000", True)))
            for name in names:
                field = table.c[name]
                result[name] = list(connection.scalars(
                    select(field).where(field.is_not(None)).distinct().limit(limit + 1)))
        return result

    def frequent_text_values(self, dataset: VerifiedDataset, column: str, *, measured: str | None = None,
                             limit: int = 12) -> list[dict[str, Any]]:
        """Valores de una columna TEXT con más registros; con measured, sólo cuentan los que tienen esa variable."""
        names = {item.sql_name: item for item in dataset.columns_metadata}
        if (column not in names or names[column].sql_type != "TEXT" or (measured is not None and measured not in names)
                or not re.fullmatch(r"dataset_[0-9a-f]{32}", dataset.physical_table_name)):
            return []
        table = Table(dataset.physical_table_name, MetaData(), *(Column(name, Text) for name in {column, measured} - {None}))
        field = table.c[column]
        statement = select(field.label("value"), func.count().label("rows")).where(field.is_not(None))
        if measured is not None:
            statement = statement.where(table.c[measured].is_not(None))
        statement = statement.group_by(field).order_by(func.count().desc(), field).limit(limit)
        with self._context.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.execute(select(func.pg_catalog.set_config("statement_timeout", "3000", True)))
            return [dict(row) for row in connection.execute(statement).mappings()]

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
        with self._verified_session() as session:
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
