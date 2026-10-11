"""Registra los modelos de dominio sin agregar dependencias del ORM al dominio."""

from threading import Lock

from sqlalchemy import MetaData
from sqlalchemy.orm import registry

from aquai.infrastructure.persistence.configurations import (
    conversation_configuration,
    dataset_configuration,
    message_configuration,
    message_source_configuration,
    message_status_configuration,
    processing_status_configuration,
    source_configuration,
    source_recipe_configuration,
    validation_status_configuration,
)

metadata = MetaData(naming_convention={
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
})
mapper_registry = registry(metadata=metadata)
_mapping_lock = Lock()
_configured = False


def configure_mappings() -> None:
    """Prepara los mapeos en memoria una sola vez, sin acceder a la base de datos ni ejecutar DDL."""
    global _configured
    with _mapping_lock:
        if _configured:
            return

        for configuration in (
            processing_status_configuration,
            message_status_configuration,
            validation_status_configuration,
            source_configuration,
            source_recipe_configuration,
            dataset_configuration,
            conversation_configuration,
            message_configuration,
            message_source_configuration,
        ):
            configuration.configure(mapper_registry)

        mapper_registry.configure()
        _configured = True
