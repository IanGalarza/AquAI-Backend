from __future__ import annotations

import re
from contextlib import contextmanager
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Column,
    MetaData,
    Table,
    Text,
    func,
    inspect,
    literal,
    or_,
    select,
    tuple_,
    union_all,
)
from sqlalchemy.exc import DBAPIError, NoSuchTableError
from sqlalchemy.orm import Session

from aquai.application.contracts.retrieval import DatasetProvenance, VerifiedDataset
from aquai.application.contracts.routing import CatalogDataset
from aquai.application.dtos.requests.listSources import SourceSort
from aquai.application.errors import SourceNotFoundError
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.domain.models import Dataset, Source
from aquai.infrastructure.persistence.dbContext import DbContext

_READY_STATUSES = (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS)
_ACCENTED = "áàäâãéèëêíìïîóòöôõúùüûñç"
_UNACCENTED = "aaaaaeeeeiiiiooooouuuunc"
_COLUMN_SUMMARY_KEYS = ("sql_name", "original_name", "sql_type", "unit")
_MISSING_OBJECT_SQLSTATES = {"42P01", "42703", "42501"}


#
# Devuelve las columnas de la tabla de fuentes que se exponen en consultas y listados.
#
def _source_columns() -> tuple:
    return (
        Source.id,
        Source.name,
        Source.original_filename,
        Source.file_type,
        Source.mime_type,
        Source.file_size_bytes,
        Source.content_hash,
        Source.is_active,
        Source.description,
        Source.processing_status,
        Source.processing_report,
        Source.semantic_index_status,
        Source.created_at,
        Source.updated_at,
    )


#
# Obtiene una copia de una fuente con sus datasets, opcionalmente con el contenido original.
#
def source_snapshot(
    session: Session,
    source_id: UUID,
    *,
    include_content: bool = False,
) -> dict[str, Any] | None:
    columns = _source_columns()
    if include_content:
        columns += (Source.file_content,)

    row = session.execute(select(*columns).where(Source.id == source_id)).mappings().first()
    if row is None:
        return None

    datasets = session.execute(_source_datasets_statement(source_id)).mappings()
    return deepcopy({**dict(row), "datasets": [dict(dataset) for dataset in datasets]})


class SqlAlchemySourceQueries:
    #
    # Guarda el contexto de base de datos y la sesión opcional usada para verificar datasets.
    #
    def __init__(self, context: DbContext, *, session: Session | None = None) -> None:
        self._context = context
        self._verification_session = session

    #
    # Busca datasets cuyo nombre, archivo o identificador coincide con una referencia dada.
    #
    def matching_datasets(self, reference: str) -> list[CatalogDataset]:
        reference = reference.strip()
        statement = (
            select(
                Dataset.id,
                Dataset.source_id,
                Dataset.name,
                Dataset.version,
                Dataset.is_active,
                Dataset.queryable,
                Dataset.row_count,
                Source.name.label("source_name"),
                Source.original_filename,
                Source.is_active.label("source_active"),
                Source.processing_status,
            )
            .join(Source, Source.id == Dataset.source_id)
            .where(or_(*_reference_predicates(reference)))
            .order_by(Source.id, Dataset.id)
            .limit(101)
        )

        with self._context.session() as session:
            rows = session.execute(statement).mappings()
            return [_catalog_dataset(row, reference) for row in rows]

    #
    # Devuelve el catálogo de fuentes y datasets con sus columnas, sin leer registros.
    #
    def catalog_overview(self) -> list[dict[str, Any]]:
        sources_statement = select(
            Source.id,
            Source.name,
            Source.original_filename,
            Source.file_type,
            Source.is_active,
            Source.processing_status,
            Source.semantic_index_status,
            Source.created_at,
        ).order_by(Source.created_at, Source.id)
        datasets_statement = select(
            Dataset.id,
            Dataset.source_id,
            Dataset.name,
            Dataset.version,
            Dataset.row_count,
            Dataset.is_active,
            Dataset.queryable,
            Dataset.columns_metadata,
        ).order_by(Dataset.created_at, Dataset.id)

        with self._context.session() as session:
            sources = session.execute(sources_statement).mappings().all()
            datasets = session.execute(datasets_statement).mappings().all()

        datasets_by_source: dict[UUID, list[dict[str, Any]]] = {}
        for row in datasets:
            datasets_by_source.setdefault(row["source_id"], []).append(row)

        overview = [
            _overview_source(source, datasets_by_source.get(source["id"], [])) for source in sources
        ]
        return deepcopy(overview)

    #
    # Provee la sesión de verificación inyectada o abre una nueva del contexto.
    #
    @contextmanager
    def _verified_session(self):
        if self._verification_session is not None:
            yield self._verification_session
            return
        with self._context.session() as session:
            yield session

    #
    # Abre una conexión de solo lectura con un límite de tiempo por consulta.
    #
    @contextmanager
    def _read_only_connection(self, statement_timeout: str):
        with self._context.engine.connect() as connection, connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.execute(
                select(func.pg_catalog.set_config("statement_timeout", statement_timeout, True))
            )
            yield connection

    #
    # Devuelve el valor de una columna de texto que coincide exactamente o sin distinguir mayúsculas.
    #
    def matching_column_values(
        self, dataset_id: UUID, version: int, column: str, value: str
    ) -> list[str]:
        datasets = self.verified_datasets([dataset_id], versions={dataset_id: version})
        if not datasets or not _has_text_column(datasets[0], column):
            return []

        table = Table(datasets[0].physical_table_name, MetaData(), Column(column, Text))
        field = table.c[column]
        with self._read_only_connection("2000") as connection:
            exact = connection.scalar(select(field).where(field == value).limit(1))
            if exact is not None:
                return [exact]

            case_insensitive = (
                select(field)
                .where(func.pg_catalog.lower(field) == func.pg_catalog.lower(value))
                .distinct()
                .limit(2)
            )
            return list(connection.scalars(case_insensitive))

    #
    # Busca en las columnas de texto de los datasets los valores que contienen los términos dados.
    #
    def find_text_values(
        self, datasets: list[VerifiedDataset], terms: list[str], *, per_column: int = 50
    ) -> dict[UUID, dict[str, list[dict[str, Any]]]]:
        result: dict[UUID, dict[str, list[dict[str, Any]]]] = {}
        patterns = _contains_patterns(terms)
        if not patterns:
            return result

        with self._read_only_connection("3000") as connection:
            for dataset in datasets:
                columns = _text_columns(dataset)
                if not columns or not _is_managed_table(dataset.physical_table_name):
                    continue

                table = Table(
                    dataset.physical_table_name,
                    MetaData(),
                    *(Column(name, Text) for name in columns),
                )
                for term, pattern in patterns.items():
                    statement = _matching_values_statement(table, columns, pattern, per_column)
                    for row in connection.execute(statement).mappings():
                        matches = result.setdefault(dataset.dataset_id, {})
                        matches.setdefault(term, []).append(dict(row))
        return result

    #
    # Devuelve los valores distintos de columnas de texto de un dataset, con un tope por columna.
    #
    def distinct_text_values(
        self, dataset: VerifiedDataset, columns: list[str], *, limit: int = 500
    ) -> dict[str, list[str]]:
        names = [name for name in _text_columns(dataset) if name in columns]
        if not names or not _is_managed_table(dataset.physical_table_name):
            return {}

        table = Table(
            dataset.physical_table_name, MetaData(), *(Column(name, Text) for name in names)
        )
        result = {}
        with self._read_only_connection("3000") as connection:
            for name in names:
                field = table.c[name]
                statement = select(field).where(field.is_not(None)).distinct().limit(limit + 1)
                result[name] = list(connection.scalars(statement))
        return result

    #
    # Devuelve los valores de una columna de texto con más registros, opcionalmente con una variable medida.
    #
    def frequent_text_values(
        self, dataset: VerifiedDataset, column: str, *, measured: str | None = None, limit: int = 12
    ) -> list[dict[str, Any]]:
        columns_by_name = {item.sql_name: item for item in dataset.columns_metadata}
        if column not in columns_by_name or columns_by_name[column].sql_type != "TEXT":
            return []
        if measured is not None and measured not in columns_by_name:
            return []
        if not _is_managed_table(dataset.physical_table_name):
            return []

        table = Table(
            dataset.physical_table_name,
            MetaData(),
            *(Column(name, Text) for name in {column, measured} - {None}),
        )
        field = table.c[column]
        statement = select(field.label("value"), func.count().label("rows")).where(
            field.is_not(None)
        )
        if measured is not None:
            statement = statement.where(table.c[measured].is_not(None))
        statement = statement.group_by(field).order_by(func.count().desc(), field).limit(limit)

        with self._read_only_connection("3000") as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    #
    # Devuelve una fuente con sus datasets o falla si no existe.
    #
    def get(self, source_id: UUID, *, include_content: bool = False) -> dict[str, Any]:
        with self._context.session() as session:
            snapshot = source_snapshot(session, source_id, include_content=include_content)
            if snapshot is None:
                raise SourceNotFoundError("La fuente no existe.")
            return snapshot

    #
    # Lista las fuentes de forma paginada, con filtro por nombre y estado y con orden elegido.
    #
    def list(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
        processing_status: str | None = None,
        sort: SourceSort = SourceSort.NEWEST,
    ) -> dict[str, Any]:
        if page < 1 or not 1 <= page_size <= 100:
            raise ValueError("La página debe ser positiva y el tamaño debe estar entre 1 y 100.")

        predicates = _list_predicates(search, processing_status)
        ordering = _source_ordering(sort)
        dataset_count = (
            select(func.count(Dataset.id))
            .where(Dataset.source_id == Source.id)
            .correlate(Source)
            .scalar_subquery()
            .label("dataset_count")
        )
        page_statement = (
            select(*_source_columns(), dataset_count)
            .where(*predicates)
            .order_by(*ordering)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        with self._context.session() as session:
            total = session.scalar(select(func.count(Source.id)).where(*predicates)) or 0
            rows = session.execute(page_statement).mappings()
            items = deepcopy([dict(row) for row in rows])

        return {"items": items, "total": total, "page": page, "page_size": page_size}

    #
    # Devuelve los identificadores de datasets que siguen vigentes y consultables.
    #
    def usable_dataset_ids(
        self, dataset_ids: list[UUID], *, versions: dict[UUID, int] | None = None
    ) -> set[UUID]:
        return {
            dataset.dataset_id for dataset in self.verified_datasets(dataset_ids, versions=versions)
        }

    #
    # Devuelve los datasets vigentes, legibles y con esquema físico coherente con el catálogo.
    #
    def verified_datasets(
        self, dataset_ids: list[UUID], *, versions: dict[UUID, int] | None = None
    ) -> list[VerifiedDataset]:
        if not dataset_ids:
            return []

        statement = (
            select(
                Dataset,
                Source.name,
                Source.original_filename,
                Source.file_type,
                Source.content_hash,
                Source.processing_status,
                Source.semantic_index_status,
                Source.processing_report,
            )
            .join(Source, Source.id == Dataset.source_id)
            .where(
                Dataset.id.in_(dataset_ids),
                Dataset.is_active.is_(True),
                Dataset.queryable.is_(True),
                Dataset.row_count > 0,
                Source.is_active.is_(True),
                Source.processing_status.in_(_READY_STATUSES),
                *_version_filter(versions),
            )
        )
        with self._verified_session() as session:
            rows = session.execute(statement).all()
            verified = []
            for row in rows:
                if not self._physical_dataset_available(session, row[0]):
                    continue
                verified.append(_verified_dataset(*row))
            return verified

    #
    # Indica si la tabla física de un dataset existe, coincide con sus columnas y tiene registros.
    #
    @staticmethod
    def _physical_dataset_available(session: Session, dataset: Dataset) -> bool:
        if not _is_managed_table(dataset.physical_table_name):
            return False

        expected = _expected_physical_columns(dataset)
        try:
            with session.begin_nested():
                return _physical_table_matches(session, dataset.physical_table_name, expected)
        except NoSuchTableError:
            return False
        except DBAPIError as error:
            if getattr(error.orig, "sqlstate", None) in _MISSING_OBJECT_SQLSTATES:
                return False
            raise


#
# Construye la consulta que obtiene los datasets de una fuente con sus metadatos.
#
def _source_datasets_statement(source_id: UUID):
    return (
        select(
            Dataset.id,
            Dataset.source_id,
            Dataset.name,
            Dataset.description,
            Dataset.source_locator,
            Dataset.physical_table_name,
            Dataset.version,
            Dataset.is_active,
            Dataset.row_count,
            Dataset.columns_metadata,
            Dataset.queryable,
            Dataset.quality_warnings,
            Dataset.semantic_description,
            Dataset.semantic_point_id,
            Dataset.created_at,
            Dataset.updated_at,
        )
        .where(Dataset.source_id == source_id)
        .order_by(Dataset.created_at, Dataset.id)
    )


#
# Indica si un nombre corresponde a una tabla física gestionada por la aplicación.
#
def _is_managed_table(name: str) -> bool:
    return re.fullmatch(r"dataset_[0-9a-f]{32}", name) is not None


#
# Convierte un texto en UUID o devuelve None si no es válido.
#
def _as_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


#
# Construye las condiciones para buscar una referencia en nombres de fuente, archivo y dataset.
#
def _reference_predicates(reference: str) -> list:
    fields = (
        Source.name,
        Source.original_filename,
        Dataset.name,
    )
    predicates = [field.icontains(reference, autoescape=True) for field in fields]

    identifier = _as_uuid(reference)
    if identifier is not None:
        predicates.extend([Source.id == identifier, Dataset.id == identifier])
    return predicates


#
# Convierte una fila de catálogo en un dataset con su disponibilidad y tipo de coincidencia.
#
def _catalog_dataset(row, reference: str) -> CatalogDataset:
    exact_names = (
        row["source_name"],
        row["original_filename"],
        row["name"],
        str(row["id"]),
        str(row["source_id"]),
    )
    wanted = reference.casefold()
    return CatalogDataset(
        dataset_id=row["id"],
        source_id=row["source_id"],
        dataset_name=row["name"],
        source_name=row["source_name"],
        version=row["version"],
        available=_catalog_row_available(row),
        exact=any(name.casefold() == wanted for name in exact_names),
        name_match=any(wanted in name.casefold() for name in exact_names),
    )


#
# Indica si un dataset del catálogo está activo, es consultable y su fuente está lista.
#
def _catalog_row_available(row):
    return (
        row["is_active"]
        and row["queryable"]
        and row["row_count"] > 0
        and row["source_active"]
        and row["processing_status"] in _READY_STATUSES
    )


#
# Resume una fuente y sus datasets para la vista general del catálogo.
#
def _overview_source(source, datasets: list) -> dict[str, Any]:
    ready = source["processing_status"] in _READY_STATUSES
    return {
        "source_id": source["id"],
        "name": source["name"],
        "original_filename": source["original_filename"],
        "file_type": source["file_type"],
        "is_active": source["is_active"],
        "processing_status": str(source["processing_status"]),
        "semantic_index_status": str(source["semantic_index_status"]),
        "created_at": source["created_at"],
        "datasets": [_overview_dataset(row, source["is_active"], ready) for row in datasets],
    }


#
# Resume un dataset del catálogo con su disponibilidad y el resumen de sus columnas.
#
def _overview_dataset(row, source_active, source_ready: bool) -> dict[str, Any]:
    available = (
        row["is_active"]
        and row["queryable"]
        and row["row_count"] > 0
        and source_active
        and source_ready
    )
    return {
        "dataset_id": row["id"],
        "name": row["name"],
        "version": row["version"],
        "row_count": row["row_count"],
        "is_active": row["is_active"],
        "available": bool(available),
        "columns": [
            {key: column.get(key) for key in _COLUMN_SUMMARY_KEYS}
            for column in row["columns_metadata"]
        ],
    }


#
# Indica si el dataset tiene una columna de texto con ese nombre.
#
def _has_text_column(dataset: VerifiedDataset, column: str) -> bool:
    return any(
        item.sql_name == column and item.sql_type == "TEXT" for item in dataset.columns_metadata
    )


#
# Devuelve los nombres de las columnas de texto de un dataset.
#
def _text_columns(dataset: VerifiedDataset) -> list[str]:
    return [item.sql_name for item in dataset.columns_metadata if item.sql_type == "TEXT"]


#
# Convierte términos de búsqueda en patrones LIKE sin acentos ni distinción de mayúsculas.
#
def _contains_patterns(terms: list[str]) -> dict[str, str]:
    accent_table = str.maketrans(_ACCENTED, _UNACCENTED)
    patterns = {}
    for term in terms:
        words = re.findall(r"\w+", term.casefold().translate(accent_table))
        if words:
            patterns[term] = "%" + "%".join(words) + "%"
    return patterns


#
# Construye la consulta que cuenta los valores de texto que coinciden con un patrón por columna.
#
def _matching_values_statement(table: Table, columns: list[str], pattern: str, per_column: int):
    parts = []
    for name in columns:
        field = table.c[name]
        normalized = func.pg_catalog.translate(func.pg_catalog.lower(field), _ACCENTED, _UNACCENTED)
        parts.append(
            select(
                literal(name).label("column"),
                field.label("value"),
                func.count().label("rows"),
            )
            .where(normalized.like(pattern))
            .group_by(field)
            .limit(per_column)
        )
    return union_all(*parts)


#
# Construye las condiciones del listado de fuentes según búsqueda y estado de procesamiento.
#
def _list_predicates(search: str | None, processing_status: str | None) -> list:
    predicates = []
    if search and search.strip():
        predicates.append(Source.name.icontains(search.strip(), autoescape=True))
    if processing_status:
        predicates.append(Source.processing_status == ProcessingStatus(processing_status))
    return predicates


#
# Devuelve el criterio de ordenamiento SQL que corresponde al orden de listado pedido.
#
def _source_ordering(sort: SourceSort) -> tuple:
    orderings = {
        SourceSort.NEWEST: (Source.created_at.desc(), Source.id.desc()),
        SourceSort.OLDEST: (Source.created_at.asc(), Source.id.asc()),
        SourceSort.NAME_ASC: (
            func.lower(Source.name).asc(),
            Source.created_at.desc(),
            Source.id.desc(),
        ),
        SourceSort.NAME_DESC: (
            func.lower(Source.name).desc(),
            Source.created_at.desc(),
            Source.id.desc(),
        ),
    }
    return orderings[SourceSort(sort)]


#
# Construye la condición que restringe los datasets a las versiones indicadas.
#
def _version_filter(versions: dict[UUID, int] | None) -> list:
    if versions is None:
        return []
    return [tuple_(Dataset.id, Dataset.version).in_(list(versions.items()))]


#
# Construye un dataset verificado con su procedencia y advertencias a partir de una fila.
#
def _verified_dataset(
    dataset: Dataset,
    source_name: str,
    original_filename: str,
    file_type: str,
    content_hash: str,
    processing_status: str,
    semantic_index_status: str,
    processing_report: dict,
) -> VerifiedDataset:
    return VerifiedDataset(
        dataset_id=dataset.id,
        source_id=dataset.source_id,
        semantic_point_id=dataset.semantic_point_id,
        name=dataset.name,
        description=dataset.semantic_description or dataset.description,
        physical_table_name=dataset.physical_table_name,
        verified_version=dataset.version,
        row_count=dataset.row_count,
        columns_metadata=deepcopy(dataset.columns_metadata),
        provenance=DatasetProvenance(
            source_name=source_name,
            original_filename=original_filename,
            file_type=file_type,
            content_hash=content_hash,
            source_locator=deepcopy(dataset.source_locator),
            processing_status=processing_status,
            semantic_index_status=semantic_index_status,
        ),
        warnings=list(
            dict.fromkeys(dataset.quality_warnings + processing_report.get("warnings", []))
        ),
    )


#
# Devuelve las columnas y tipos que debe tener la tabla física de un dataset.
#
def _expected_physical_columns(dataset: Dataset) -> dict[str, str]:
    expected = {column["sql_name"]: column["sql_type"] for column in dataset.columns_metadata}
    expected.update({"_aquai_row": "BIGINT", "_aquai_source_row": "BIGINT"})
    if dataset.source_locator.get("orientation") == "columns":
        expected["_aquai_source_column"] = "BIGINT"
    return expected


#
# Comprueba que la tabla física exista, tenga los tipos esperados y conserve al menos un registro.
#
def _physical_table_matches(session: Session, table_name: str, expected: dict[str, str]) -> bool:
    connection = session.connection()
    inspector = inspect(connection)
    if not inspector.has_table(table_name):
        return False

    columns = inspector.get_columns(table_name)
    actual = {
        column["name"]: str(column["type"].compile(dialect=connection.dialect))
        for column in columns
    }
    if any(actual.get(name) != sql_type for name, sql_type in expected.items()):
        return False

    table = Table(
        table_name,
        MetaData(),
        *(Column(column["name"], column["type"], quote=True) for column in columns),
        quote=True,
    )
    probe = select(*(table.c[name].is_not(None) for name in expected)).limit(1)
    return session.execute(probe).first() is not None
