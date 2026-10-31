from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.dbContext import DbContext


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache(maxsize=1)
def get_db_context() -> DbContext:
    return DbContext(get_database_settings())


def get_db_session() -> Iterator[Session]:
    with get_db_context().session() as session:
        yield session
