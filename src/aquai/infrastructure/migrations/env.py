import re
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, create_engine

from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.persistence.mapping import configure_mappings, metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

configure_mappings()
target_metadata = metadata


#
# Excluye de la comparación las tablas físicas de datasets creadas durante la ingesta.
#
def include_name(name: str | None, type_: str, parent_names: dict) -> bool:
    return not (
        type_ == "table" and name is not None and re.fullmatch(r"dataset_[0-9a-f]{32}", name)
    )


#
# Ejecuta las migraciones generando SQL sin conectarse a la base de datos.
#
def run_migrations_offline() -> None:
    context.configure(
        url=DatabaseSettings().sqlalchemy_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


#
# Configura Alembic sobre una conexión y ejecuta las migraciones en su transacción.
#
def run_migrations_on_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


#
# Ejecuta las migraciones con la conexión de la aplicación o una propia.
#
def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        run_migrations_on_connection(connection)
        return

    engine = create_engine(DatabaseSettings().sqlalchemy_url)
    try:
        with engine.connect() as connection:
            run_migrations_on_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
