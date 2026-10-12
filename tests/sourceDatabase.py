"""SQLite aislado con exclusión de transacciones y tipos compatibles."""
from contextlib import contextmanager
from threading import RLock

from sqlalchemy import create_engine, event
from sqlalchemy.dialects.postgresql import BYTEA, JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from aquai.infrastructure.persistence.mapping import configure_mappings, metadata


@compiles(BYTEA, "sqlite")
def bytea_for_test(element, compiler, **kwargs):
    return "BLOB"


@compiles(JSONB, "sqlite")
def jsonb_for_test(element, compiler, **kwargs):
    return "JSON"


class SourceTestDbContext:
    def __init__(self):
        configure_mappings()
        self.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
        )
        self._lock = RLock()

        @event.listens_for(self.engine, "connect")
        def configure(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")

        metadata.create_all(
            self.engine, tables=[metadata.tables[name] for name in ("Source", "Dataset")],
        )

    @contextmanager
    def session(self):
        with self._lock, Session(self.engine, expire_on_commit=False) as session:
            try:
                yield session
            except BaseException:
                session.rollback()
                raise

    def dispose(self):
        self.engine.dispose()
