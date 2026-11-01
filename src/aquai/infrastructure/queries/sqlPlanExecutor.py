from datetime import datetime, timezone
from time import perf_counter

from pydantic import ConfigDict, TypeAdapter, ValidationError
from sqlalchemy import MetaData, Table, Text, case, cast, func, inspect, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from aquai.application.contracts.execution import (
    ExecutionDataset,
    ExecutionError,
    OperationExecution,
    ResultValue,
    SqlExecutionResult,
)
from aquai.application.contracts.planning import PlanningResult
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.errors import PlannerValidationError
from aquai.application.execution import plan_fingerprint
from aquai.application.planning.validation import validate_execution_plan
from aquai.infrastructure.config.sqlExecution import SqlExecutionSettings
from aquai.infrastructure.persistence.dbContext import DbContext
from aquai.infrastructure.queries.planCompiler import (
    CompiledStep,
    UnsupportedSqlOperation,
    compile_step,
)
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries

_row_adapter = TypeAdapter(dict[str, ResultValue], config=ConfigDict(allow_inf_nan=False))

_EXECUTED_STATUSES = {"executed", "executed_as_dependency"}

_DATABASE_ERRORS = {
    "57014": ("query_timeout", "PostgreSQL canceló la consulta por tiempo excedido."),
    "55P03": (
        "lock_timeout",
        "No se pudo acceder a los datos dentro del tiempo permitido.",
    ),
    "42501": (
        "read_permission_denied",
        "La conexión no tiene los permisos de lectura necesarios.",
    ),
    "42P01": ("catalog_changed", "La tabla autorizada ya no está disponible."),
    "42703": ("catalog_changed", "El esquema autorizado cambió."),
}
_UNKNOWN_DATABASE_ERROR = ("database_error", "No se pudo completar la ejecución en PostgreSQL.")


class _ExecutionLimit(ValueError):
    pass


class SqlAlchemyPlanExecutor:
    #
    # Guarda el contexto de base de datos y la configuración de ejecución SQL.
    #
    def __init__(self, context: DbContext, settings: SqlExecutionSettings) -> None:
        self._context = context
        self._settings = settings

    #
    # Ejecuta un plan en una transacción de solo lectura y devuelve su resultado.
    #
    def execute(self, plan: PlanningResult) -> SqlExecutionResult:
        return _PlanExecution(self._context, self._settings, plan).run()


class _PlanExecution:
    #
    # Prepara el estado y el resultado inicial de una ejecución de un plan.
    #
    def __init__(
        self, context: DbContext, settings: SqlExecutionSettings, plan: PlanningResult
    ) -> None:
        self._context = context
        self._settings = settings
        self._plan = plan
        self._started = perf_counter()
        now = datetime.now(timezone.utc)
        self._result = SqlExecutionResult(
            status="not_executed",
            started_at=now,
            finished_at=now,
            plan_fingerprint=plan_fingerprint(plan),
            output_operations=list(plan.output_steps),
            warnings=list(plan.warnings),
        )
        self._current: OperationExecution | None = None
        self._query_started = 0.0

    #
    # Ejecuta el plan, traduce los fallos a un error de ejecución y devuelve el resultado final.
    #
    def run(self) -> SqlExecutionResult:
        result = self._result
        try:
            self._execute_in_read_only_transaction()
        except PlannerValidationError:
            result.error = ExecutionError(
                code="invalid_plan",
                message="El plan o sus metadatos no superaron la revalidación. Es necesario volver a planificar.",
            )
        except ValidationError:
            code = "invalid_result" if self._current is not None else "invalid_plan"
            result.error = ExecutionError(
                code=code,
                message="El resultado o contrato contiene valores no admitidos; no se sustituyeron por datos ficticios.",
            )
        except _ExecutionLimit as error:
            result.error = ExecutionError(code="execution_limit", message=str(error))
        except SQLAlchemyError as error:
            result.error = _database_error(error)

        self._mark_interrupted_operation()
        self._warn_incomplete_python_inputs()
        self._finish()

        return result

    #
    # Revalida el plan y ejecuta sus consultas dentro de una transacción de solo lectura.
    #
    def _execute_in_read_only_transaction(self) -> None:
        plan = self._plan
        validate_execution_plan(plan, plan.selected_datasets)

        engine = self._context.engine
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                self._start_read_only(connection)
                self._set_search_path(connection)
                selected, operations = self._revalidated_plan(connection)
                self._result.datasets = [_execution_dataset(item) for item in selected]
                self._warn_privileged_role(connection)
                tables = self._authorized_tables(connection, selected)

                records, compiled = self._compile_operations(operations, tables)
                targets = _query_targets(operations, compiled, plan.output_steps)
                used_bytes = self._metadata_bytes()
                for step in targets:
                    used_bytes = self._run_target(
                        connection, step, compiled[step.id], records, used_bytes
                    )
                self._current = None

    #
    # Pone la transacción en solo lectura y verifica que quedó activa.
    #
    def _start_read_only(self, connection) -> None:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        self._configure(connection)
        if connection.scalar(text("SHOW transaction_read_only")) != "on":
            raise PlannerValidationError("No se pudo establecer una transacción de sólo lectura.")
        self._result.read_only = True

    #
    # Fija el schema de los datasets como único camino de búsqueda de la transacción.
    #
    def _set_search_path(self, connection) -> None:
        preparer = connection.dialect.identifier_preparer
        schema = preparer.quote_identifier(self._settings.dataset_schema)
        connection.execute(
            select(func.pg_catalog.set_config("search_path", f"{schema}, pg_catalog", True))
        )

    #
    # Revalida el plan contra el catálogo vigente y devuelve los datasets y operaciones.
    #
    def _revalidated_plan(self, connection):
        plan = self._plan
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            queries = SqlAlchemySourceQueries(self._context, session=session)
            versions = {item.dataset_id: item.verified_version for item in plan.selected_datasets}
            catalog = queries.verified_datasets(list(versions), versions=versions)
            selected, operations = validate_execution_plan(plan, catalog)
            session.commit()
        return selected, operations

    #
    # Agrega una advertencia si la conexión usa un rol con privilegios elevados.
    #
    def _warn_privileged_role(self, connection) -> None:
        privileges = connection.execute(
            text(
                "SELECT rolsuper, rolcreaterole, rolcreatedb FROM pg_catalog.pg_roles WHERE rolname = current_user"
            )
        ).one()
        if any(privileges):
            self._result.warnings.append(
                "La conexión usa un rol privilegiado; se aplica READ ONLY. Configurar SQL_EXECUTION_DATABASE_URL con un rol restringido para defensa adicional."
            )

    #
    # Carga las tablas físicas autorizadas de los datasets seleccionados.
    #
    def _authorized_tables(self, connection, selected: list[VerifiedDataset]) -> dict:
        schema = self._settings.dataset_schema
        table_names = set(inspect(connection).get_table_names(schema=schema))
        tables = {}
        for dataset in selected:
            if dataset.physical_table_name not in table_names:
                raise PlannerValidationError(
                    "La tabla autorizada no es una tabla física del schema configurado."
                )
            tables[dataset.dataset_id] = Table(
                dataset.physical_table_name,
                MetaData(),
                schema=schema,
                autoload_with=connection,
                resolve_fks=False,
            )
        return tables

    #
    # Compila las operaciones del plan y crea el registro de ejecución de cada una.
    #
    def _compile_operations(self, operations, tables: dict):
        records: dict[str, OperationExecution] = {}
        compiled: dict[str, CompiledStep] = {}
        for step in operations:
            record = OperationExecution(
                operation_id=step.id,
                status="not_executed",
                columns=step.output_fields,
                warnings=self._validation_warnings(step.id),
            )
            record.dataset_ids = _input_dataset_ids(step, records)
            self._result.operations.append(record)
            records[step.id] = record
            _compile_operation(step, record, compiled, tables)
        return records, compiled

    #
    # Devuelve los mensajes de validación asociados a un paso del plan.
    #
    def _validation_warnings(self, step_id: str) -> list[str]:
        return [issue.message for issue in self._plan.validation_issues if issue.step_id == step_id]

    #
    # Calcula los bytes que ocupan los metadatos y falla si superan el límite del resultado.
    #
    def _metadata_bytes(self) -> int:
        result = self._result
        used_bytes = len(result.model_dump_json().encode("utf-8")) + 4096
        if used_bytes >= self._settings.max_result_bytes:
            result.datasets = []
            for record in result.operations:
                record.columns = []
            result.warnings = [
                "Los metadatos completos permanecen en PlanningResult; no se duplican porque exceden el límite del resultado."
            ]
            raise _ExecutionLimit("Los metadatos del resultado superan el límite configurado.")
        return used_bytes

    #
    # Ejecuta una consulta objetivo dentro del presupuesto restante y registra su resultado.
    #
    def _run_target(self, connection, step, item: CompiledStep, records, used_bytes: int) -> int:
        settings = self._settings
        record = records[step.id]
        self._current = record
        self._query_started = perf_counter()

        remaining = settings.max_rows - self._result.rows_returned
        if remaining <= 0 or used_bytes >= settings.max_result_bytes:
            record.warnings.append(
                "Se agotó el presupuesto total de devolución; la consulta no se ejecutó."
            )
            return used_bytes

        self._configure(connection)
        query, guard_name = _bounded_query(step, item, remaining, settings)
        used_bytes = self._stream_rows(connection, query, guard_name, step, remaining, used_bytes)

        record.status = "executed"
        record.materialized = True
        record.duration_ms = round((perf_counter() - self._query_started) * 1000, 3)
        if record.truncated:
            record.warnings.append(
                "Se devuelve un prefijo acotado del resultado; no usarlo como conjunto completo para análisis posteriores."
            )
        _link_ancestors(item, records, step.id)
        return used_bytes

    #
    # Lee las filas de una consulta por lotes respetando los límites de tiempo, filas y bytes.
    #
    def _stream_rows(
        self, connection, query, guard_name: str, step, remaining: int, used_bytes: int
    ) -> int:
        settings = self._settings
        record = self._current
        streamed = query.execution_options(stream_results=True, yield_per=50)
        with connection.execute(streamed) as cursor:
            for mapping in cursor.mappings():
                if (perf_counter() - self._started) * 1000 > settings.max_duration_ms:
                    raise _ExecutionLimit("La ejecución superó el tiempo total permitido.")
                if record.rows_returned == remaining:
                    record.truncated = True
                    break
                if mapping[guard_name]:
                    raise _ExecutionLimit(
                        "Una fila supera el tamaño permitido; no se sustituyeron sus valores."
                    )

                row = _validated_row(mapping, step.output_fields)
                row_bytes = len(_row_adapter.dump_json(row)) + 1
                if row_bytes > settings.max_row_bytes:
                    raise _ExecutionLimit(
                        "Una fila serializada supera el tamaño permitido; no se truncaron sus valores."
                    )
                if used_bytes + row_bytes > settings.max_result_bytes:
                    record.truncated = True
                    break

                record.rows.append(row)
                record.rows_returned += 1
                self._result.rows_returned += 1
                used_bytes += row_bytes
        return used_bytes

    #
    # Marca como fallida la operación que estaba en curso cuando ocurrió el error.
    #
    def _mark_interrupted_operation(self) -> None:
        current = self._current
        error = self._result.error
        if current is None or error is None:
            return
        current.status = "error"
        current.error = error
        current.truncated = bool(current.rows)
        current.duration_ms = round((perf_counter() - self._query_started) * 1000, 3)

    #
    # Advierte de las operaciones Python cuyas entradas están pendientes o incompletas.
    #
    def _warn_incomplete_python_inputs(self) -> None:
        records = {item.operation_id: item for item in self._result.operations}
        for step in self._plan.operations:
            record = records.get(step.id)
            if record is None or record.status != "pending_python":
                continue
            incomplete_inputs = any(
                not records[parent].materialized or records[parent].truncated
                for parent in step.depends_on
            )
            if incomplete_inputs:
                record.warnings.append(
                    "Hay entradas pendientes o incompletas; no se autoriza calcular sobre un prefijo como si fuera el conjunto completo."
                )

    #
    # Cierra el resultado con operaciones pendientes, estado final y duración.
    #
    def _finish(self) -> None:
        result = self._result
        result.pending_operations = [
            item.operation_id for item in result.operations if item.status not in _EXECUTED_STATUSES
        ]
        result.status = _final_status(result)
        result.finished_at = datetime.now(timezone.utc)
        result.duration_ms = round((perf_counter() - self._started) * 1000, 3)

    #
    # Aplica los tiempos límite de consulta y bloqueo según el tiempo total restante.
    #
    def _configure(self, connection) -> None:
        settings = self._settings
        remaining = int(settings.max_duration_ms - (perf_counter() - self._started) * 1000)
        if remaining <= 0:
            raise _ExecutionLimit("La ejecución superó el tiempo total permitido.")
        timeouts = {
            "statement_timeout": min(settings.statement_timeout_ms, remaining),
            "lock_timeout": min(settings.lock_timeout_ms, remaining),
            "idle_in_transaction_session_timeout": settings.max_duration_ms,
        }
        for key, value in timeouts.items():
            connection.execute(
                select(func.pg_catalog.set_config(key, str(value), True))
            ).scalar_one()


#
# Traduce un error de base de datos en un error de ejecución con código y mensaje.
#
def _database_error(error: SQLAlchemyError) -> ExecutionError:
    sqlstate = getattr(getattr(error, "orig", None), "sqlstate", None)
    code, message = _DATABASE_ERRORS.get(sqlstate, _UNKNOWN_DATABASE_ERROR)
    return ExecutionError(code=code, message=message)


#
# Convierte un dataset verificado en el dataset descrito en el resultado de ejecución.
#
def _execution_dataset(item: VerifiedDataset) -> ExecutionDataset:
    return ExecutionDataset(
        dataset_id=item.dataset_id,
        source_id=item.source_id,
        version=item.verified_version,
        provenance=item.provenance,
        columns=item.columns_metadata,
        warnings=item.warnings,
    )


#
# Devuelve los datasets de entrada de un paso, incluidos los de sus dependencias.
#
def _input_dataset_ids(step, records: dict[str, OperationExecution]) -> list:
    dataset_ids = {ref.dataset_id for ref in step.columns}
    for dependency in step.depends_on:
        dataset_ids.update(records[dependency].dataset_ids)
    return sorted(dataset_ids, key=str)


#
# Compila un paso a SQL o lo marca como pendiente, bloqueado o no soportado.
#
def _compile_operation(step, record: OperationExecution, compiled: dict, tables: dict) -> None:
    if step.tool == "python":
        record.status = "pending_python"
        record.warnings.append(
            "La operación se delega al Python Agent con entradas completas y verificadas."
        )
        return

    if any(parent not in compiled for parent in step.depends_on):
        record.status = "blocked"
        record.warnings.append(
            "Falta ejecutar una dependencia Python o una operación no soportada."
        )
        return

    try:
        compiled[step.id] = compile_step(step, compiled, tables)
    except UnsupportedSqlOperation as error:
        record.status = "unsupported"
        record.warnings.append(str(error))
        return

    record.dataset_ids = sorted(compiled[step.id].dataset_ids, key=str)
    record.warnings.extend(compiled[step.id].warnings)


#
# Determina los pasos compilados cuya consulta debe ejecutarse.
#
def _query_targets(operations, compiled: dict, output_steps: list[str]) -> list:
    consumed = set()
    pending_inputs = set()
    for step in operations:
        if step.id in compiled:
            consumed.update(step.depends_on)
        else:
            pending_inputs.update(
                dependency for dependency in step.depends_on if dependency in compiled
            )

    targets = [
        step
        for step in operations
        if step.id in compiled
        and (step.id in output_steps or step.id not in consumed or step.id in pending_inputs)
    ]
    targets.sort(key=lambda step: _output_order(step.id, output_steps))
    return targets


#
# Devuelve la clave de orden de un paso, con los pasos de salida primero y en su orden.
#
def _output_order(step_id: str, output_steps: list[str]) -> tuple[bool, int]:
    if step_id in output_steps:
        return False, output_steps.index(step_id)
    return True, 0


#
# Construye la consulta de un paso acotada por filas restantes y tamaño máximo de fila.
#
def _bounded_query(step, item: CompiledStep, remaining: int, settings: SqlExecutionSettings):
    ties = step.limit is not None and bool(step.order_by)
    fetch_limit = min(step.limit, remaining + 1) if step.limit is not None else remaining + 1
    base = (item.statement if ties else item.statement.limit(fetch_limit)).subquery()

    columns = [base.c[column.name] for column in step.output_fields]
    guard = _row_size_guard(columns, settings)
    guard_name = _unused_guard_name(base)
    query = select(
        *(case((guard, column), else_=None).label(column.name) for column in columns),
        (~guard).label(guard_name),
    ).select_from(base)

    for ordering in step.order_by:
        query = query.order_by(_nulls_last_ordering(base, ordering))
    if ties:
        query = query.limit(remaining + 1)
    return query, guard_name


#
# Construye la condición que detecta filas demasiado grandes para no transferirlas.
#
def _row_size_guard(columns, settings: SqlExecutionSettings):
    size = sum(
        func.coalesce(func.pg_catalog.octet_length(cast(column, Text)), 0) for column in columns
    )
    return size <= min(settings.max_row_bytes, settings.max_result_bytes)


#
# Devuelve un nombre de columna de guarda que no choque con las columnas existentes.
#
def _unused_guard_name(base) -> str:
    guard_index = 0
    guard_name = "__aquai_size_guard_0"
    while guard_name in base.c:
        guard_index += 1
        guard_name = f"__aquai_size_guard_{guard_index}"
    return guard_name


#
# Construye el orden de una columna con los valores nulos al final.
#
def _nulls_last_ordering(base, ordering):
    field = ordering.field
    name = field.column if hasattr(field, "column") else field.field
    expression = base.c[name]
    directed = expression.asc() if ordering.direction == "asc" else expression.desc()
    return directed.nulls_last()


#
# Valida los valores de una fila y devuelve solo las columnas de salida.
#
def _validated_row(mapping, output_fields) -> dict:
    values = {column.name: mapping[column.name] for column in output_fields}
    return _row_adapter.validate_python(values)


#
# Registra la consulta ejecutada en los pasos antecesores y los marca como ejecutados.
#
def _link_ancestors(item: CompiledStep, records: dict, step_id: str) -> None:
    for ancestor in item.ancestors:
        record = records[ancestor]
        record.query_operation_ids.append(step_id)
        if record.status == "not_executed":
            record.status = "executed_as_dependency"


#
# Calcula el estado final de la ejecución según errores, pendientes y truncamientos.
#
def _final_status(result: SqlExecutionResult) -> str:
    executed = any(item.status in _EXECUTED_STATUSES for item in result.operations)
    if result.error:
        if executed:
            return "partial"
        return "not_executed" if result.error.code == "invalid_plan" else "failed"
    if result.pending_operations or any(item.truncated for item in result.operations):
        return "partial" if executed else "not_executed"
    return "succeeded"
