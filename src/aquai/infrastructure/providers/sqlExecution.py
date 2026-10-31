from functools import lru_cache

from aquai.application.Services.executeSqlPlanService import ExecuteSqlPlanService
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
    read_only_url = get_sql_execution_settings().database_url
    if not read_only_url:
        return get_db_context()

    return DbContext(DatabaseSettings(_env_file=None, DATABASE_URL=read_only_url))


def create_sql_agent() -> ExecuteSqlPlanService:
    return ExecuteSqlPlanService(
        SqlAlchemyPlanExecutor(get_sql_execution_context(), get_sql_execution_settings())
    )
