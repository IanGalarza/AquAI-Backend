from threading import Lock

from sqlalchemy import MetaData
from sqlalchemy.orm import registry

from aquai.infrastructure.persistence.configurations import (
    conversationConfiguration,
    datasetConfiguration,
    messageConfiguration,
    messageSourceConfiguration,
    messageStatusConfiguration,
    sourceConfiguration,
)

metadata = MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
    }
)
mapper_registry = registry(metadata=metadata)
_mapping_lock = Lock()
_configured = False


def configure_mappings() -> None:
    global _configured
    with _mapping_lock:
        if _configured:
            return

        for configuration in (
            messageStatusConfiguration,
            sourceConfiguration,
            datasetConfiguration,
            conversationConfiguration,
            messageConfiguration,
            messageSourceConfiguration,
        ):
            configuration.configure(mapper_registry)

        mapper_registry.configure()
        _configured = True
