"""Ejecutor PostgreSQL acotado y de sólo lectura; nunca acepta sentencias SQL."""

from datetime import datetime, timezone
from time import perf_counter

from pydantic import ConfigDict, TypeAdapter, ValidationError
from sqlalchemy import MetaData, Table, Text, case, cast, func, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from aquai.application.contracts.execution import (
    ExecutionDataset, ExecutionError, OperationExecution, ResultValue, SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.errors import PlannerValidationError
from aquai.application.execution import plan_fingerprint
from aquai.application.useCases.queries.executeSqlPlanHandler import validate_execution_plan
from aquai.infrastructure.config.sqlExecution import SqlExecutionSettings
from aquai.infrastructure.persistence.dbContext import DbContext
from aquai.infrastructure.queries.planCompiler import UnsupportedSqlOperation, compile_step
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


_row_adapter = TypeAdapter(dict[str, ResultValue], config=ConfigDict(allow_inf_nan=False))


class _ExecutionLimit(ValueError):
    pass


class SqlAlchemyPlanExecutor:
    def __init__(self, context: DbContext, settings: SqlExecutionSettings) -> None:
        self._context = context
        self._settings = settings

    def execute(self, plan: PlanningResult) -> SqlExecutionResult:
        started = perf_counter()
        now = datetime.now(timezone.utc)
        result = SqlExecutionResult(status="not_executed", started_at=now, finished_at=now,
                                    plan_fingerprint=plan_fingerprint(plan),
                                    output_operations=list(plan.output_steps), warnings=list(plan.warnings))
        current = None
        try:
            validate_execution_plan(plan, plan.selected_datasets)
            settings = self._settings
            with self._context.engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
                with connection.begin():
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                    self._configure(connection, started)
                    if connection.scalar(text("SHOW transaction_read_only")) != "on":
                        raise PlannerValidationError("No se pudo establecer una transacción de sólo lectura.")
                    result.read_only = True
                    schema = connection.dialect.identifier_preparer.quote_identifier(settings.dataset_schema)
                    connection.execute(select(func.pg_catalog.set_config("search_path", f"{schema}, pg_catalog", True)))
                    with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                        queries = SqlAlchemySourceQueries(self._context, session=session)
                        versions = {item.dataset_id: item.verified_version for item in plan.selected_datasets}
                        catalog = queries.verified_datasets(list(versions), versions=versions)
                        selected, operations = validate_execution_plan(plan, catalog)
                        session.commit()  # Libera el savepoint, no la transacción externa READ ONLY.
                    result.datasets = [ExecutionDataset(
                        dataset_id=item.dataset_id, source_id=item.source_id, version=item.verified_version,
                        provenance=item.provenance, columns=item.columns_metadata, warnings=item.warnings,
                    ) for item in selected]
                    owner = connection.execute(text(
                        "SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_catalog.pg_roles WHERE rolname = current_user"
                    )).one()
                    if any(owner):
                        result.warnings.append("La conexión usa un rol privilegiado; se aplica READ ONLY. Configurar SQL_EXECUTION_DATABASE_URL con un rol restringido para defensa adicional.")
                    table_names = set(inspect(connection).get_table_names(schema=settings.dataset_schema))
                    tables = {}
                    for dataset in selected:
                        if dataset.physical_table_name not in table_names:
                            raise PlannerValidationError("La tabla autorizada no es una tabla física del schema configurado.")
                        tables[dataset.dataset_id] = Table(dataset.physical_table_name, MetaData(),
                            schema=settings.dataset_schema, autoload_with=connection, resolve_fks=False)
                    compiled = {}
                    records = {}
                    for step in operations:
                        record = OperationExecution(operation_id=step.id, status="not_executed", columns=step.output_fields,
                            warnings=[issue.message for issue in plan.validation_issues if issue.step_id == step.id])
                        record.dataset_ids = sorted({ref.dataset_id for ref in step.columns} |
                            {dataset_id for dependency in step.depends_on for dataset_id in records[dependency].dataset_ids}, key=str)
                        result.operations.append(record)
                        records[step.id] = record
                        if step.tool == "python":
                            record.status = "pending_python"
                            record.warnings.append("La operación se delega al Python Agent con entradas completas y verificadas.")
                        elif any(parent not in compiled for parent in step.depends_on):
                            record.status = "blocked"
                            record.warnings.append("Falta ejecutar una dependencia Python o una operación no soportada.")
                        else:
                            try:
                                compiled[step.id] = compile_step(step, compiled, tables)
                                record.dataset_ids = sorted(compiled[step.id].dataset_ids, key=str)
                                record.warnings.extend(compiled[step.id].warnings)
                            except UnsupportedSqlOperation as error:
                                record.status = "unsupported"
                                record.warnings.append(str(error))
                    consumed = {dependency for step in operations if step.id in compiled for dependency in step.depends_on}
                    pending_inputs = {dependency for step in operations if step.id not in compiled
                                      for dependency in step.depends_on if dependency in compiled}
                    targets = [step for step in operations if step.id in compiled and
                               (step.id in plan.output_steps or step.id not in consumed or step.id in pending_inputs)]
                    targets.sort(key=lambda step: (step.id not in plan.output_steps,
                                                  plan.output_steps.index(step.id) if step.id in plan.output_steps else 0))
                    used_bytes = len(result.model_dump_json().encode("utf-8")) + 4096
                    if used_bytes >= settings.max_result_bytes:
                        result.datasets = []
                        for record in result.operations:
                            record.columns = []
                        result.warnings = ["Los metadatos completos permanecen en PlanningResult; no se duplican porque exceden el límite del resultado."]
                        raise _ExecutionLimit("Los metadatos del resultado superan el límite configurado.")
                    for step in targets:
                        current = records[step.id]
                        query_started = perf_counter()
                        remaining = settings.max_rows - result.rows_returned
                        if remaining <= 0 or used_bytes >= settings.max_result_bytes:
                            current.warnings.append("Se agotó el presupuesto total de devolución; la consulta no se ejecutó.")
                            continue
                        self._configure(connection, started)
                        item = compiled[step.id]
                        ties = step.limit is not None and bool(step.order_by)
                        fetch_limit = min(step.limit, remaining + 1) if step.limit is not None else remaining + 1
                        # Con empates el paso ya acota sus filas (FETCH ... WITH TIES); el presupuesto
                        # total se aplica afuera para no recortar un empate.
                        base = (item.statement if ties else item.statement.limit(fetch_limit)).subquery()
                        columns = [base.c[column.name] for column in step.output_fields]
                        size = sum(func.coalesce(func.pg_catalog.octet_length(cast(column, Text)), 0) for column in columns)
                        guard = size <= min(settings.max_row_bytes, settings.max_result_bytes)
                        guard_index, guard_name = 0, "__aquai_size_guard_0"
                        while guard_name in base.c:
                            guard_index += 1
                            guard_name = f"__aquai_size_guard_{guard_index}"
                        query = select(*(case((guard, column), else_=None).label(column.name) for column in columns),
                                       (~guard).label(guard_name)).select_from(base)
                        for ordering in step.order_by:
                            name = ordering.field.column if hasattr(ordering.field, "column") else ordering.field.field
                            expression = base.c[name]
                            query = query.order_by((expression.asc() if ordering.direction == "asc" else expression.desc()).nulls_last())
                        if ties:
                            query = query.limit(remaining + 1)
                        with connection.execute(query.execution_options(stream_results=True, yield_per=50)) as cursor:
                            for mapping in cursor.mappings():
                                if (perf_counter() - started) * 1000 > settings.max_duration_ms:
                                    raise _ExecutionLimit("La ejecución superó el tiempo total permitido.")
                                if current.rows_returned == remaining:
                                    current.truncated = True
                                    break
                                if mapping[guard_name]:
                                    raise _ExecutionLimit("Una fila supera el tamaño permitido; no se sustituyeron sus valores.")
                                row = _row_adapter.validate_python({column.name: mapping[column.name] for column in step.output_fields})
                                row_bytes = len(_row_adapter.dump_json(row)) + 1
                                if row_bytes > settings.max_row_bytes:
                                    raise _ExecutionLimit("Una fila serializada supera el tamaño permitido; no se truncaron sus valores.")
                                if used_bytes + row_bytes > settings.max_result_bytes:
                                    current.truncated = True
                                    break
                                current.rows.append(row)
                                current.rows_returned += 1
                                result.rows_returned += 1
                                used_bytes += row_bytes
                        current.status, current.materialized = "executed", True
                        current.duration_ms = round((perf_counter() - query_started) * 1000, 3)
                        if current.truncated:
                            current.warnings.append("Se devuelve un prefijo acotado del resultado; no usarlo como conjunto completo para análisis posteriores.")
                        for ancestor in item.ancestors:
                            record = records[ancestor]
                            record.query_operation_ids.append(step.id)
                            if record.status == "not_executed":
                                record.status = "executed_as_dependency"
                    current = None
        except PlannerValidationError:
            result.error = ExecutionError(code="invalid_plan", message="El plan o sus metadatos no superaron la revalidación. Es necesario volver a planificar.")
        except ValidationError:
            code = "invalid_result" if current is not None else "invalid_plan"
            result.error = ExecutionError(code=code, message="El resultado o contrato contiene valores no admitidos; no se sustituyeron por datos ficticios.")
        except _ExecutionLimit as error:
            result.error = ExecutionError(code="execution_limit", message=str(error))
        except SQLAlchemyError as error:
            code = getattr(getattr(error, "orig", None), "sqlstate", None)
            public = {
                "57014": ("query_timeout", "PostgreSQL canceló la consulta por tiempo excedido."),
                "55P03": ("lock_timeout", "No se pudo acceder a los datos dentro del tiempo permitido."),
                "42501": ("read_permission_denied", "La conexión no tiene los permisos de lectura necesarios."),
                "42P01": ("catalog_changed", "La tabla autorizada ya no está disponible."),
                "42703": ("catalog_changed", "El esquema autorizado cambió."),
            }.get(code, ("database_error", "No se pudo completar la ejecución en PostgreSQL."))
            result.error = ExecutionError(code=public[0], message=public[1])
        if current is not None and result.error is not None:
            current.status, current.error = "error", result.error
            current.truncated = bool(current.rows)
            current.duration_ms = round((perf_counter() - query_started) * 1000, 3)
        records = {item.operation_id: item for item in result.operations}
        for step in plan.operations:
            record = records.get(step.id)
            if record is not None and record.status == "pending_python" and any(
                not records[parent].materialized or records[parent].truncated for parent in step.depends_on
            ):
                record.warnings.append("Hay entradas pendientes o incompletas; no se autoriza calcular sobre un prefijo como si fuera el conjunto completo.")
        result.pending_operations = [item.operation_id for item in result.operations
                                     if item.status not in {"executed", "executed_as_dependency"}]
        executed = any(item.status in {"executed", "executed_as_dependency"} for item in result.operations)
        if result.error:
            result.status = "partial" if executed else ("not_executed" if result.error.code == "invalid_plan" else "failed")
        elif result.pending_operations or any(item.truncated for item in result.operations):
            result.status = "partial" if executed else "not_executed"
        else:
            result.status = "succeeded"
        result.finished_at = datetime.now(timezone.utc)
        result.duration_ms = round((perf_counter() - started) * 1000, 3)
        return result

    def _configure(self, connection, started):
        settings = self._settings
        remaining = int(settings.max_duration_ms - (perf_counter() - started) * 1000)
        if remaining <= 0:
            raise _ExecutionLimit("La ejecución superó el tiempo total permitido.")
        for key, value in {
            "statement_timeout": min(settings.statement_timeout_ms, remaining),
            "lock_timeout": min(settings.lock_timeout_ms, remaining),
            "idle_in_transaction_session_timeout": settings.max_duration_ms,
        }.items():
            connection.execute(select(func.pg_catalog.set_config(key, str(value), True))).scalar_one()
