"""Funciones de composición y dependencia de base de datos para futuras rutas de FastAPI."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy.orm import Session

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.dbContext import DbContext


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Lee y valida la configuración cuando la aplicación la solicita."""
    return DatabaseSettings()


@lru_cache(maxsize=1)
def get_db_context() -> DbContext:
    """Comparte un contexto de base de datos que se inicializa al solicitarlo."""
    return DbContext(get_database_settings())


def get_db_session() -> Iterator[Session]:
    """Provee una sesión con commit explícito para usar con Depends de FastAPI."""
    with get_db_context().session() as session:
        yield session
