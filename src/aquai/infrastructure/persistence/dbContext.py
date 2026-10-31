from collections.abc import Generator
from contextlib import contextmanager
from threading import Lock

from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import Session, sessionmaker

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.mapping import metadata


class DbContext:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._engine_lock = Lock()

    @property
    def metadata(self) -> MetaData:
        return metadata

    @property
    def engine(self) -> Engine:
        with self._engine_lock:
            if self._engine is None:
                self._engine = create_engine(
                    self._settings.sqlalchemy_url,
                    connect_args={"connect_timeout": 5},
                    pool_pre_ping=True,
                    pool_timeout=10,
                )
                self._session_factory = sessionmaker(
                    bind=self._engine,
                    expire_on_commit=False,
                )
            return self._engine

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
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
        with self._engine_lock:
            if self._engine is None:
                return
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
