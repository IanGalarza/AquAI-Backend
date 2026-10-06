"""Contexto de SQLAlchemy y manejo explícito de las sesiones de PostgreSQL."""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import Session, sessionmaker

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.mapping import metadata


class DbContext:
    """Administra el motor y las sesiones sin crear tablas ni conectar al inicializarse."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._engine_lock = Lock()

    @property
    def metadata(self) -> MetaData:
        """Expone el esquema completo para revisarlo o preparar futuras migraciones."""
        return metadata

    @property
    def engine(self) -> Engine:
        """Construye el motor al acceder por primera vez, sin abrir una conexión."""
        with self._engine_lock:
            if self._engine is None:
                self._engine = create_engine(self._settings.sqlalchemy_url)
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    expire_on_commit=False,
                )
            return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provee una sesión con commit explícito, revierte ante errores y siempre la cierra."""
        _ = self.engine
        assert self._session_factory is not None
        session = self._session_factory()
        try:
            yield session
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Libera los recursos del motor y permite reconstruirlo si se vuelve a usar."""
        with self._engine_lock:
            if self._engine is not None:
                self._engine.dispose()
                self._engine = None
                self._session_factory = None
