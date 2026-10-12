"""Base SQLite nueva por test; adapta sólo la compilación de tipos PostgreSQL."""

from __future__ import annotations

from contextlib import contextmanager
import json
from threading import RLock

from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aquai.infrastructure.persistence.configurations.statusSeeds import (
    processing_status_seed_rows, validation_status_seed_rows,
)
from aquai.infrastructure.persistence.mapping import configure_mappings, metadata


@compiles(BYTEA, "sqlite")
def _bytea_for_test(element, compiler, **kwargs):
    return "BLOB"


@compiles(JSONB, "sqlite")
def _jsonb_for_test(element, compiler, **kwargs):
    return "JSON"


class SourceTestDbContext:
    """No lee .env, no modifica metadatos ni contacta bases configuradas."""

    def __init__(self) -> None:
        configure_mappings()
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        self._session_lock = RLock()

        @event.listens_for(self.engine, "connect")
        def configure(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")

            def jsonb_typeof(document):
                if document is None:
                    return None
                value = json.loads(document)
                return "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "scalar"

            connection.create_function("jsonb_typeof", 1, jsonb_typeof)

        # Sólo las entidades utilizadas por gestión de fuentes; no hay tablas nuevas.
        tables = [metadata.tables[name] for name in (
            "ProcessingStatus", "ValidationStatus", "Source", "SourceRecipe", "Dataset",
        )]
        metadata.create_all(self.engine, tables=tables)
        with self.engine.begin() as connection:
            connection.execute(metadata.tables["ProcessingStatus"].insert(), processing_status_seed_rows())
            connection.execute(metadata.tables["ValidationStatus"].insert(), validation_status_seed_rows())

    @contextmanager
    def session(self):
        # StaticPool comparte una conexión; ninguna petición puede intercalar su
        # transacción con el worker. El lock dura sólo la sesión corta del test.
        with self._session_lock, Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
            except BaseException:
                session.rollback()
                raise

    def dispose(self) -> None:
        self.engine.dispose()
