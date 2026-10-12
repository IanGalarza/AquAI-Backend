"""Transacciones cortas de carga, tablas físicas y estados operativos de fuentes."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from threading import Lock
from typing import Any
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary

from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Engine, MetaData, Numeric,
    Table, Text, func, select, text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aquai.application.contracts.ingestion import PreparedTable
from aquai.application.errors import SourceNotFoundError
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus
from aquai.domain.models import Dataset, Source
from aquai.infrastructure.persistence.dbContext import DbContext


_WORKER_KEY = int.from_bytes(sha256(b"aquai-source-ingestion").digest()[:8], "big", signed=True)
_engine_locks: WeakKeyDictionary[Engine, tuple[Lock, Lock]] = WeakKeyDictionary()
_engine_locks_guard = Lock()
_SQL_TYPES = {
    "TEXT": Text, "BIGINT": BigInteger, "NUMERIC": Numeric,
    "BOOLEAN": Boolean, "DATE": Date, "TIMESTAMP": DateTime,
}
_PROVENANCE_COLUMNS = {"_aquai_row", "_aquai_source_row"}


def _sqlite_locks(engine: Engine) -> tuple[Lock, Lock]:
    with _engine_locks_guard:
        if engine not in _engine_locks:
            _engine_locks[engine] = (Lock(), Lock())
        return _engine_locks[engine]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _warnings(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item)[:1000] for group in groups for item in group))[:50]


def _validate_table(table: PreparedTable) -> None:
    if not isinstance(table.source_locator, dict):
        raise ValueError("La procedencia de la tabla debe ser un objeto JSON.")
    if not isinstance(table.columns, list) or any(not isinstance(column, dict) for column in table.columns):
        raise ValueError("Los metadatos de columnas deben ser objetos JSON.")
    json.dumps(table.source_locator, allow_nan=False)
    json.dumps(table.columns, allow_nan=False)
    names = [column.get("sql_name") for column in table.columns]
    if not names or len(set(names)) != len(names):
        raise ValueError("La tabla debe tener columnas con nombres SQL únicos.")
    for column in table.columns:
        name = column["sql_name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name):
            raise ValueError("El nombre técnico de columna no es seguro.")
        if name in _PROVENANCE_COLUMNS or name.startswith("_aquai_"):
            raise ValueError("El nombre técnico usa una columna interna reservada.")
        if column.get("sql_type") not in _SQL_TYPES:
            raise ValueError("El tipo SQL de columna no está admitido.")
        if not isinstance(column.get("original_name"), str):
            raise ValueError("Falta el nombre original de una columna.")
        if type(column.get("numeric_eligible")) is not bool:
            raise ValueError("Falta la aptitud numérica verificada de una columna.")
        if column["numeric_eligible"] and column["sql_type"] not in {"BIGINT", "NUMERIC"}:
            raise ValueError("Una columna no numérica no puede habilitar cálculos numéricos.")
    if not table.rows:
        raise ValueError("Una tabla consultable debe contener registros.")
    expected = set(names) | _PROVENANCE_COLUMNS
    for position, row in enumerate(table.rows, 1):
        if set(row) != expected:
            raise ValueError("Los registros no corresponden a las columnas declaradas.")
        if type(row["_aquai_row"]) is not int or row["_aquai_row"] != position:
            raise ValueError("El orden interno de registros debe ser consecutivo.")
        if type(row["_aquai_source_row"]) is not int or row["_aquai_source_row"] < 1:
            raise ValueError("La fila original debe tener procedencia válida.")


def _physical_table(name: str, columns: list[dict[str, Any]]) -> Table:
    return Table(
        name, MetaData(),
        Column("_aquai_row", BigInteger, primary_key=True, autoincrement=False),
        Column("_aquai_source_row", BigInteger, nullable=False),
        *(Column(column["sql_name"], _SQL_TYPES[column["sql_type"]](), quote=True)
          for column in columns),
        quote=True,
    )


class SqlAlchemySourceCommands:
    def __init__(self, context: DbContext) -> None:
        self._context = context

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        engine = self._context.engine
        lock = _sqlite_locks(engine)[1] if engine.dialect.name == "sqlite" else None
        if lock is not None:
            lock.acquire()
        try:
            with self._context.session() as session:
                yield session
                session.commit()
        finally:
            if lock is not None:
                lock.release()

    @contextmanager
    def worker_lock(self) -> Iterator[bool]:
        """Un lock global evita inferencias simultáneas y habilita recuperar interrupciones."""
        engine = self._context.engine
        if engine.dialect.name == "sqlite":
            lock = _sqlite_locks(engine)[0]
            acquired = lock.acquire(blocking=False)
            try:
                yield acquired
            finally:
                if acquired:
                    lock.release()
            return
        if engine.dialect.name != "postgresql":
            raise RuntimeError("El worker requiere PostgreSQL.")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            acquired = bool(connection.scalar(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": _WORKER_KEY}
            ))
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _WORKER_KEY})

    @staticmethod
    def _source(session: Session, source_id: UUID) -> Source:
        source = session.scalar(select(Source).where(Source.id == source_id).with_for_update())
        if source is None:
            raise SourceNotFoundError("La fuente no existe.")
        return source

    def create_upload(self, source: Source) -> tuple[UUID, bool]:
        digest = sha256(source.file_content).hexdigest()
        if source.content_hash and source.content_hash != digest:
            raise ValueError("La huella de la fuente no corresponde al original.")
        if source.file_size_bytes != len(source.file_content):
            raise ValueError("El tamaño declarado no corresponde al original.")
        if source.processing_status != ProcessingStatus.UPLOADED:
            raise ValueError("Una fuente nueva debe comenzar en uploaded.")
        source.content_hash = digest
        try:
            with self._write_session() as session:
                existing = session.scalar(select(Source.id).where(Source.content_hash == digest))
                if existing is not None:
                    return existing, False
                session.add(source)
                session.flush()
                return source.id, True
        except IntegrityError:
            # La constraint SHA también coordina cargas simultáneas entre procesos.
            with self._context.session() as session:
                existing = session.scalar(select(Source.id).where(Source.content_hash == digest))
                if existing is None:
                    raise
                return existing, False

    def claim_next(self) -> UUID | None:
        with self._write_session() as session:
            source = session.scalar(
                select(Source).where(Source.processing_status == ProcessingStatus.UPLOADED)
                .order_by(Source.created_at, Source.id).limit(1)
                .with_for_update(skip_locked=True)
            )
            if source is None:
                return None
            source.processing_status = ProcessingStatus.PROCESSING
            source.updated_at = _now()
            return source.id

    def recover_interrupted(self) -> int:
        """Sólo se llama tras obtener worker_lock, cuando ningún otro worker trabaja."""
        with self._write_session() as session:
            sources = session.scalars(select(Source).where(
                (Source.processing_status == ProcessingStatus.PROCESSING)
                | (Source.semantic_index_status == SemanticIndexStatus.PROCESSING)
            ).with_for_update()).all()
            for source in sources:
                if source.processing_status == ProcessingStatus.PROCESSING:
                    has_datasets = session.scalar(select(Dataset.id).where(
                        Dataset.source_id == source.id, Dataset.queryable.is_(True)
                    ).limit(1)) is not None
                    source.processing_status = (
                        ProcessingStatus.READY_WITH_WARNINGS if has_datasets
                        else ProcessingStatus.UPLOADED
                    )
                    report = deepcopy(source.processing_report)
                    report["warnings"] = _warnings(
                        report.get("warnings", []), ["Se recuperó un procesamiento interrumpido."]
                    )
                    source.processing_report = report
                if source.semantic_index_status == SemanticIndexStatus.PROCESSING:
                    source.semantic_index_status = SemanticIndexStatus.PENDING
                source.updated_at = _now()
            return len(sources)

    def save_prepared(
        self, source_id: UUID, tables: list[PreparedTable], report: dict[str, Any],
    ) -> None:
        locators: set[str] = set()
        for table in tables:
            _validate_table(table)
            locator = json.dumps(table.source_locator, sort_keys=True, allow_nan=False)
            if locator in locators:
                raise ValueError("Dos datasets de la misma fuente no pueden tener idéntica procedencia.")
            locators.add(locator)
        if not isinstance(report, dict):
            raise ValueError("El informe de procesamiento debe ser un objeto JSON.")
        json.dumps(report, allow_nan=False)
        created_tables: list[Table] = []
        try:
            with self._write_session() as session:
                source = self._source(session, source_id)
                if session.scalar(select(Dataset.id).where(Dataset.source_id == source_id).limit(1)):
                    return
                final_report = deepcopy(report)
                warnings = _warnings(
                    final_report.get("warnings", []),
                    *(table.warnings for table in tables),
                )
                final_report["warnings"] = warnings
                final_report["dataset_count"] = len(tables)
                final_report["row_count"] = sum(len(table.rows) for table in tables)
                for prepared in tables:
                    dataset_id = uuid4()
                    physical_name = f"dataset_{dataset_id.hex}"
                    table = _physical_table(physical_name, prepared.columns)
                    table.create(session.connection())
                    created_tables.append(table)
                    for offset in range(0, len(prepared.rows), 1000):
                        session.execute(table.insert(), prepared.rows[offset:offset + 1000])
                    count = session.scalar(select(func.count()).select_from(table))
                    if count != len(prepared.rows):
                        raise RuntimeError("No se confirmó la cantidad completa de registros.")
                    session.add(Dataset(
                        id=dataset_id, source_id=source_id, name=prepared.name,
                        description=prepared.description, source_locator=deepcopy(prepared.source_locator),
                        physical_table_name=physical_name, row_count=count,
                        columns_metadata=deepcopy(prepared.columns), queryable=True,
                        quality_warnings=_warnings(prepared.warnings), semantic_point_id=dataset_id,
                    ))
                source.processing_report = final_report
                source.processing_status = (
                    ProcessingStatus.READY_WITH_WARNINGS if warnings else ProcessingStatus.READY
                ) if tables else ProcessingStatus.FAILED
                if not tables:
                    final_report["errors"] = [{
                        "code": "no_reliable_tables",
                        "message": "No se encontraron tablas confiables; se conserva el original.",
                    }]
                source.semantic_index_status = SemanticIndexStatus.PENDING
                source.updated_at = _now()
        except BaseException:
            # PostgreSQL revierte DDL e INSERT juntos. SQLite necesita limpiar DDL explícitamente.
            if self._context.engine.dialect.name == "sqlite" and created_tables:
                with self._context.engine.begin() as connection:
                    for table in reversed(created_tables):
                        table.drop(connection, checkfirst=True)
            raise

    def save_dataset_semantics(
        self, dataset_id: UUID, description: str, columns: list[dict[str, Any]],
        warnings: list[str] | None = None,
    ) -> None:
        with self._write_session() as session:
            dataset = session.scalar(select(Dataset).where(Dataset.id == dataset_id).with_for_update())
            if dataset is None:
                raise SourceNotFoundError("El dataset no existe.")
            metadata = deepcopy(dataset.columns_metadata)
            existing = {column["sql_name"]: column for column in metadata}
            if columns and [column.get("sql_name") for column in columns] != list(existing):
                raise ValueError("La descripción alteró, omitió o reordenó columnas de la tabla.")
            for column in columns:
                name = column.get("sql_name")
                stored = existing[name]
                for field in ("original_name", "sql_name", "sql_type", "numeric_eligible", "unit"):
                    if column.get(field) != stored.get(field):
                        raise ValueError("La descripción no puede alterar metadatos verificados.")
                if "meaning" in column:
                    value = column["meaning"]
                    if value is not None and not isinstance(value, str):
                        raise ValueError("El significado debe ser texto o null.")
                    stored["meaning"] = value
            dataset.columns_metadata = metadata
            dataset.semantic_description = description
            dataset.quality_warnings = _warnings(dataset.quality_warnings, warnings or [])
            dataset.updated_at = _now()
            if warnings:
                source = self._source(session, dataset.source_id)
                report = deepcopy(source.processing_report)
                report["warnings"] = _warnings(report.get("warnings", []), warnings)
                source.processing_report = report
                if source.processing_status == ProcessingStatus.READY:
                    source.processing_status = ProcessingStatus.READY_WITH_WARNINGS
                source.updated_at = _now()

    def set_index_status(
        self, source_id: UUID, status: str, error: str | None = None,
        *, expected_active: bool | None = None,
    ) -> None:
        status = SemanticIndexStatus(status)
        with self._write_session() as session:
            source = self._source(session, source_id)
            if (
                status == SemanticIndexStatus.READY and expected_active is not None
                and source.is_active != expected_active
            ):
                status = SemanticIndexStatus.PENDING
            source.semantic_index_status = status
            report = deepcopy(source.processing_report)
            if error:
                report["semantic_index_error"] = str(error)[:2000]
            elif status == SemanticIndexStatus.READY:
                report.pop("semantic_index_error", None)
            source.processing_report = report
            source.updated_at = _now()

    def claim_index_next(self) -> UUID | None:
        with self._write_session() as session:
            source = session.scalar(select(Source).where(
                Source.semantic_index_status == SemanticIndexStatus.PENDING,
                Source.processing_status.in_(
                    (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS)
                ),
            ).order_by(Source.created_at, Source.id).limit(1).with_for_update(skip_locked=True))
            if source is None:
                return None
            source.semantic_index_status = SemanticIndexStatus.PROCESSING
            source.updated_at = _now()
            return source.id

    def set_active(self, source_id: UUID, is_active: bool) -> None:
        if type(is_active) is not bool:
            raise ValueError("is_active debe ser booleano.")
        with self._write_session() as session:
            source = self._source(session, source_id)
            source.is_active = is_active
            if source.processing_status in (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS):
                source.semantic_index_status = SemanticIndexStatus.PENDING
            source.updated_at = _now()

    def record_failure(self, source_id: UUID, code: str, message: str) -> None:
        with self._write_session() as session:
            source = self._source(session, source_id)
            report = deepcopy(source.processing_report)
            report["errors"] = [*report.get("errors", []), {
                "code": code[:100], "message": str(message)[:2000],
            }][-20:]
            source.processing_report = report
            has_datasets = session.scalar(select(Dataset.id).where(
                Dataset.source_id == source_id, Dataset.queryable.is_(True)
            ).limit(1)) is not None
            source.processing_status = (
                ProcessingStatus.READY_WITH_WARNINGS if has_datasets
                else ProcessingStatus.UPLOADED if code == "interrupted"
                else ProcessingStatus.FAILED
            )
            if code == "interrupted" and source.semantic_index_status == SemanticIndexStatus.PROCESSING:
                source.semantic_index_status = SemanticIndexStatus.PENDING
            source.updated_at = _now()

    def request_reindex(self, source_id: UUID) -> None:
        self.set_index_status(source_id, SemanticIndexStatus.PENDING)
