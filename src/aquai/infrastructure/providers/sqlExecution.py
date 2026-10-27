"""Composición del SQL Agent con conexión de lectura opcional y límites propios."""

from functools import lru_cache

from aquai.application.useCases.queries.executeSqlPlanHandler import ExecuteSqlPlanHandler
from aquai.infrastructure.config.database import DatabaseSettings
from aquai.infrastructure.config.sqlExecution import SqlExecutionSettings
from aquai.infrastructure.persistence.dbContext import DbContext
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.queries.sqlPlanExecutor import SqlAlchemyPlanExecutor


@lru_cache
def get_sql_execution_settings() -> SqlExecutionSettings:
    return SqlExecutionSettings()


@lru_cache
def get_sql_execution_context() -> DbContext:
    url = get_sql_execution_settings().database_url
    return DbContext(DatabaseSettings(_env_file=None, DATABASE_URL=url)) if url else get_db_context()


def create_sql_agent() -> ExecuteSqlPlanHandler:
    return ExecuteSqlPlanHandler(SqlAlchemyPlanExecutor(get_sql_execution_context(), get_sql_execution_settings()))
