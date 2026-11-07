import json

from aquai.application.answering.segments import answer_text
from aquai.application.contracts.answering import AnswerSegment, SourceReference
from aquai.application.contracts.conversations import (
    AssistantRecord,
    AssistantSnapshot,
    SourceAssociation,
    StoredGeneration,
    StoredQueryResponse,
    StoredTable,
)
from aquai.application.errors import ConversationStorageError
from aquai.application.execution import evidence_fingerprint

_CONTEXT_RECORD_FIELDS = {
    "prompt_version",
    "sha256",
    "size_bytes",
    "budget_bytes",
    "omitted_rows",
    "sent_to_model",
}
_EXECUTION_FIELDS = {
    "status",
    "materialized",
    "truncated",
    "rows_returned",
    "rows_processed",
    "rows_filtered",
    "duration_ms",
    "error",
}
_OPERATION_SUMMARY_FIELDS = ("id", "tool", "operation", "depends_on", "execution")
_OMITTED_CHART_FIELDS = {
    "operation_id",
    "chart_type",
    "title",
    "datasets",
    "input_operation_ids",
    "input_rows",
    "excluded_rows",
}
_FRAGMENT_FIELDS = {
    "evidence_id",
    "chunk_id",
    "position",
    "kind",
    "page_start",
    "page_end",
    "section",
    "element_label",
    "doubts",
}
_SOURCE_COLUMN_FIELDS = {
    "sql_name",
    "original_name",
    "sql_type",
    "unit",
    "unit_evidence",
    "numeric_eligible",
    "derived_from",
    "representation",
    "excluded_count",
    "source_column",
    "source_row",
}
_DIRECT_RESPONSE_TYPES = {
    "general": "general",
    "catalog": "catalog",
    "historical": "historical",
    "needs_clarification": "clarification",
    "failed": "general",
    "document": "document",
}


#
# Indica si cada párrafo de la respuesta figura tal cual en el texto original.
#
def _reproduces(answer, original):
    paragraphs = [item for item in answer.split("\n\n") if item.strip()]
    return bool(paragraphs) and all(item in original for item in paragraphs)


#
# Calcula el tamaño en bytes UTF-8 del valor serializado como JSON compacto.
#
def _bytes(value):
    content = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    return len(content.encode("utf-8"))


INTERRUPTED_GENERATION = (
    "La consulta se interrumpió antes de terminar, por ejemplo porque el servidor se reinició. "
    "La pregunta quedó guardada y podés volver a enviarla."
)


#
# Construye el registro del asistente para un turno fallido o cancelado.
#
def failure_record(turn, code, message, *, cancelled=False):
    return AssistantRecord(
        content=message,
        sources=[],
        status_name="cancelled" if cancelled else "failed",
        content_data={
            "schema_version": 1,
            "kind": "generation_failure",
            "user_message_id": str(turn.user_message_id),
            "error": {"code": code, "message": message},
        },
    )


class ConversationSnapshotBuilder:
    #
    # Guarda los límites de tamaño de mensaje, tablas y detalles de fuentes.
    #
    def __init__(
        self, *, max_message_bytes, max_table_rows, max_table_bytes, max_source_details_bytes
    ):
        self._max_message_bytes = max_message_bytes
        self._max_table_rows = max_table_rows
        self._max_table_bytes = max_table_bytes
        self._max_source_details_bytes = max_source_details_bytes

    #
    # Arma el registro del asistente de un turno, validando la evidencia y ajustándolo a los límites.
    #
    def build(self, state, turn, *, reused=False):
        if state.routing is not None and state.routing.route != "data":
            return self._direct(state, turn)

        response = state.response
        validation = state.validation
        _check_turn(state, turn, reused)
        allowed = {item.evidence_id: item for item in validation.usable_evidence}
        _check_references(response, validation, allowed)
        _check_documents(state, response)

        stored = _stored_response(response)
        warnings = self._store_tables(state, response.results, allowed, stored)
        _check_visualizations(state, validation, stored)
        operations = _stored_operations(state)
        snapshot = AssistantSnapshot(
            response=stored,
            operations=operations,
            context_policy=_analysis_context_policy(state, turn, reused),
            validated_at=validation.verified_at,
        )
        self._fit(snapshot, stored, operations, warnings, response.language)

        associations = self._associations(state, response)
        associations += self._document_associations(response, len(associations))

        return AssistantRecord(
            content=stored.answer,
            content_data=snapshot.model_dump(
                mode="json", exclude={"response": {"generation": {"context"}}}
            ),
            sources=associations,
            status_name="failed" if response.generation.status == "failed" else "completed",
        )

    #
    # Guarda las tablas de resultados dentro de los presupuestos de filas y bytes, con avisos si se recortan.
    #
    def _store_tables(self, state, references, allowed, stored):
        warnings = []
        budget = _TableBudget(self._max_table_rows, self._max_table_bytes)
        for reference in references:
            evidence = _table_evidence(reference, allowed)
            record = _complete_record(state, evidence)
            rows = budget.take(record.model_dump(mode="json")["rows"])
            table = _stored_table(evidence, record, rows)
            budget.fit_exclusions(table)
            if len(rows) != record.rows_returned:
                warnings.append(
                    f"La tabla {evidence.evidence_id} conserva {len(rows)} "
                    f"de {record.rows_returned} filas; las restantes no están "
                    "almacenadas ni disponibles mediante un recurso HTTP."
                )
            stored.results.append(table)

        return warnings

    #
    # Reduce el snapshot por etapas hasta que cabe en el límite del mensaje o falla si no es posible.
    #
    def _fit(self, snapshot, stored, operations, warnings, language):
        if self._too_large(snapshot):
            _drop_table_details(stored)
            warnings.append(
                "Las tablas y exclusiones detalladas se omitieron por el límite "
                "del historial; se conservan sus metadatos."
            )
        if self._too_large(snapshot):
            snapshot.operations = [
                {key: value[key] for key in _OPERATION_SUMMARY_FIELDS} for value in operations
            ]
            warnings.append(
                "Se conserva un resumen de las operaciones por el límite del historial."
            )
        if self._too_large(snapshot):
            _omit_visualizations(snapshot, stored, language)
            warnings.append(
                "Las figuras exceden el límite del historial y no se conservaron; "
                "no existe un archivo o enlace para recuperarlas."
            )

        if warnings:
            snapshot.storage_warnings = list(dict.fromkeys(warnings))
            stored.limitations = list(
                dict.fromkeys([*stored.limitations, *snapshot.storage_warnings])
            )

        if self._too_large(snapshot):
            raise ConversationStorageError(
                "Los metadatos mínimos exceden el límite de almacenamiento del mensaje."
            )

    #
    # Indica si el snapshot serializado supera el tamaño máximo permitido para un mensaje.
    #
    def _too_large(self, snapshot):
        return len(snapshot.model_dump_json().encode("utf-8")) > self._max_message_bytes

    #
    # Arma el registro de una respuesta directa (general, catálogo, histórica o aclaración) sin datos nuevos.
    #
    def _direct(self, state, turn):
        response = state.response
        routing = state.routing
        expected_type = _DIRECT_RESPONSE_TYPES[routing.route]
        if _differs_from_turn(state, turn, expected_type):
            raise ConversationStorageError("La respuesta directa no corresponde al turno.")

        history = {item.message_id: item for item in turn.structured_history}
        if not set(routing.history_message_ids).issubset(history):
            raise ConversationStorageError("La respuesta cita mensajes ajenos al contexto.")

        historical = routing.route == "historical"
        if historical:
            associations = self._historical_associations(response, routing, history)
            associations += self._document_associations(response, len(associations))
        else:
            _check_without_references(response)
            _check_documents(state, response)
            associations = self._document_associations(response, 0)

        stored = _stored_response(response)
        generation = stored.generation
        operations = (
            history[routing.history_message_ids[0]].analysis.operations if historical else []
        )
        snapshot = AssistantSnapshot(
            response=stored,
            validated_at=None,
            operations=operations,
            context_policy={
                "history_message_ids": [str(key) for key in turn.history_message_ids],
                "history_used_by": "router,direct_response",
                "followup_resolution": "structured_history",
            },
        )
        if self._too_large(snapshot):
            raise ConversationStorageError("La respuesta directa supera el límite del historial.")

        return AssistantRecord(
            content=stored.answer,
            sources=associations,
            content_data=snapshot.model_dump(
                mode="json", exclude={"response": {"generation": {"context"}}}
            ),
            status_name="failed" if generation.status == "failed" else "completed",
        )

    #
    # Devuelve una asociación por documento citado, con los fragmentos que la respaldan.
    #
    def _document_associations(self, response, offset):
        grouped = {}
        for item in response.document_references:
            grouped.setdefault(item.source_id, []).append(item)

        return [
            SourceAssociation(source_id, None, offset + index, self._document_details(items))
            for index, (source_id, items) in enumerate(grouped.items(), 1)
        ]

    #
    # Construye los detalles de un documento citado y de sus fragmentos.
    #
    def _document_details(self, items):
        first = items[0]
        details = {
            "schema_version": 1,
            "kind": "document",
            "document_id": str(first.document_id),
            "file_name": first.file_name,
            "title": first.title,
            "content_hash": first.content_hash,
            "fragments": [item.model_dump(mode="json", include=_FRAGMENT_FIELDS) for item in items],
        }
        if _bytes(details) > self._max_source_details_bytes:
            raise ConversationStorageError("La referencia documental excede su límite.")

        return details

    #
    # Valida que una respuesta histórica reproduzca el mensaje citado y devuelve sus fuentes asociadas.
    #
    def _historical_associations(self, response, routing, history):
        if len(routing.history_message_ids) != 1:
            raise ConversationStorageError("La referencia histórica es ambigua.")

        message_id = routing.history_message_ids[0]
        original = history[message_id]
        if original.analysis is None or not _reproduces(response.answer, original.text):
            raise ConversationStorageError("La respuesta alteró el resultado histórico.")

        sources = [SourceReference.model_validate(item) for item in original.analysis.sources]
        references = [item.model_dump(mode="json") for item in response.operation_references]
        if (
            response.sources != sources
            or references != original.analysis.operation_references
            or response.historical_message_ids != [message_id]
        ):
            raise ConversationStorageError("Las referencias históricas no coinciden.")

        associations = []
        for index, source in enumerate(sources, 1):
            details = self._historical_details(source, message_id, references, original)
            associations.append(
                SourceAssociation(source.source_id, source.dataset_id, index, details)
            )

        return associations

    #
    # Construye los detalles de una fuente citada en una respuesta histórica, con sus columnas originales.
    #
    def _historical_details(self, source, message_id, references, original):
        dataset_id = str(source.dataset_id)
        details = {
            "schema_version": 1,
            "historical_message_id": str(message_id),
            "dataset_version": source.version,
            "dataset_name": source.dataset_name,
            "provenance": source.provenance.model_dump(mode="json"),
            "operation_references": [
                item for item in references if dataset_id in item["dataset_ids"]
            ],
            "columns": [],
            "columns_complete": True,
        }
        if _bytes(details) > self._max_source_details_bytes:
            raise ConversationStorageError("La referencia histórica supera su límite.")

        self._add_columns(details, original.source_columns.get(dataset_id, []))

        return details

    #
    # Agrega columnas a los detalles de una fuente mientras quepan en el límite de metadatos.
    #
    def _add_columns(self, details, columns):
        for column in columns:
            details["columns"].append(column)
            if _bytes(details) > self._max_source_details_bytes:
                details["columns"].pop()
                details["columns_complete"] = False
                break

    #
    # Devuelve las fuentes únicas de la respuesta con los detalles de su procedencia y columnas.
    #
    def _associations(self, state, response):
        selected = (
            {item.dataset_id: item for item in state.planning.selected_datasets}
            if state.planning
            else {}
        )
        result = []
        seen = set()
        for source in response.sources:
            key = (source.source_id, source.dataset_id)
            if key in seen:
                continue
            seen.add(key)
            dataset = _verified_dataset(selected, source)
            details = self._source_details(source, dataset, response.operation_references)
            result.append(
                SourceAssociation(source.source_id, source.dataset_id, len(result) + 1, details)
            )

        return result

    #
    # Construye los detalles de procedencia, referencias de operación y columnas de una fuente.
    #
    def _source_details(self, source, dataset, operation_references):
        references = [
            item for item in operation_references if source.dataset_id in item.dataset_ids
        ]
        details = {
            "schema_version": 1,
            "dataset_version": source.version,
            "dataset_name": source.dataset_name,
            "provenance": source.provenance.model_dump(mode="json"),
            "operation_references": [item.model_dump(mode="json") for item in references],
            "columns": [],
            "columns_complete": True,
        }
        if _bytes(details) > self._max_source_details_bytes:
            raise ConversationStorageError(
                "La procedencia mínima excede el límite de una referencia."
            )

        columns = [
            column.model_dump(mode="json", include=_SOURCE_COLUMN_FIELDS)
            for column in dataset.columns_metadata
        ]
        self._add_columns(details, columns)
        if _bytes(details) > self._max_source_details_bytes:
            raise ConversationStorageError("La referencia excede su límite de metadatos.")

        return details


class _TableBudget:
    #
    # Guarda el máximo de filas y bytes disponibles para las tablas del mensaje.
    #
    def __init__(self, rows, size):
        self._rows = rows
        self._bytes = size

    #
    # Toma las filas que caben en el presupuesto restante y descuenta lo consumido.
    #
    def take(self, rows):
        taken = []
        for row in rows:
            size = _bytes(row) + 1
            if self._rows <= 0 or size > self._bytes:
                break
            taken.append(row)
            self._rows -= 1
            self._bytes -= size

        return taken

    #
    # Conserva las exclusiones de la tabla si caben en el presupuesto o las descarta con un aviso.
    #
    def fit_exclusions(self, table):
        size = _bytes([item.model_dump(mode="json") for item in table.exclusions])
        if size <= self._bytes:
            self._bytes -= size
            return

        table.exclusions = []
        table.exclusions_complete = False
        table.warnings.append(
            "El detalle de exclusiones excede el límite del historial; no se conservó completo."
        )


#
# Verifica que la respuesta conserve la evidencia validada y el contexto del turno.
#
def _check_turn(state, turn, reused):
    response = state.response
    validation = state.validation
    fingerprint = evidence_fingerprint(state.planning, state.execution, state.python_execution)
    if (
        response is None
        or validation is None
        or state.question != turn.question
        or validation.evidence_fingerprint != fingerprint
    ):
        raise ConversationStorageError(
            "La respuesta no conserva la evidencia validada de este turno."
        )
    if not reused and state.conversation_context != turn.history:
        raise ConversationStorageError("El resultado no conserva el contexto enviado al grafo.")
    if not reused and state.conversation_history != turn.structured_history:
        raise ConversationStorageError("El contexto estructurado cambió durante el turno.")


#
# Verifica que las fuentes y referencias de la respuesta correspondan a evidencia autorizada.
#
def _check_references(response, validation, allowed):
    if response.sources != validation.sources or response.validation_status != validation.status:
        raise ConversationStorageError("Las fuentes no corresponden a la respuesta validada.")

    for reference in response.operation_references:
        item = allowed.get(reference.evidence_id)
        if item is None or _reference_identity(reference) != _reference_identity(item):
            raise ConversationStorageError("La respuesta contiene referencias no autorizadas.")


#
# Verifica que cada cita documental corresponda a un fragmento recuperado en este turno.
#
def _check_documents(state, response):
    retrieval = state.document_retrieval
    recovered = {item.chunk_id for item in retrieval.fragments} if retrieval else set()
    recovered |= {
        str(item.get("chunk_id"))
        for message in state.conversation_history
        for item in message.documents
    }
    known = {item.evidence_id for item in response.document_references}
    for reference in response.document_references:
        if reference.chunk_id not in recovered and str(reference.chunk_id) not in recovered:
            raise ConversationStorageError("La respuesta cita un fragmento que no se recuperó.")

    for segment in response.segments:
        if not set(segment.document_ids) <= known:
            raise ConversationStorageError("Un párrafo cita un documento sin referencia.")


#
# Devuelve la identidad de una referencia: operación, herramienta y datasets.
#
def _reference_identity(reference):
    return reference.operation_id, reference.tool, reference.dataset_ids


#
# Crea la respuesta a almacenar sin resultados ni contexto completo de generación.
#
def _stored_response(response):
    generation = response.generation.model_dump(exclude={"context"})
    context = response.generation.context
    if context is not None:
        generation["context_record"] = context.model_dump(include=_CONTEXT_RECORD_FIELDS)
    data = response.model_dump(exclude={"generation", "results"})

    return StoredQueryResponse(**data, generation=StoredGeneration(**generation), results=[])


#
# Devuelve la evidencia autorizada de tipo tabla que respalda una referencia de resultado.
#
def _table_evidence(reference, allowed):
    evidence = allowed.get(reference.evidence_id)
    if (
        evidence is None
        or evidence.kind != "table"
        or evidence.operation_id != reference.operation_id
    ):
        raise ConversationStorageError("La tabla no tiene evidencia autorizada.")

    return evidence


#
# Devuelve el registro de ejecución completo de una tabla o falla si está truncada o incompleta.
#
def _complete_record(state, evidence):
    batch = state.execution if evidence.tool == "sql" else state.python_execution
    record = next(
        (item for item in batch.operations if item.operation_id == evidence.operation_id),
        None,
    )
    if (
        record is None
        or not record.materialized
        or record.truncated
        or record.rows_returned != len(record.rows)
    ):
        raise ConversationStorageError("No se conserva una tabla completa para la respuesta.")

    return record


#
# Construye la tabla almacenada con las filas conservadas y su estado de completitud.
#
def _stored_table(evidence, record, rows):
    complete = len(rows) == record.rows_returned

    return StoredTable(
        evidence_id=evidence.evidence_id,
        operation_id=evidence.operation_id,
        columns=record.columns,
        rows=rows,
        total_rows=record.rows_returned,
        stored_rows=len(rows),
        complete=complete,
        storage=_storage(complete, rows),
        warnings=list(record.warnings),
        rows_processed=record.rows_processed,
        rows_filtered=record.rows_filtered,
        exclusions=record.exclusions,
    )


#
# Indica el modo de almacenamiento de una tabla: completa, vista previa o solo metadatos.
#
def _storage(complete, rows):
    if complete:
        return "inline"

    return "preview" if rows else "metadata_only"


#
# Verifica que cada gráfico de la respuesta coincida con la evidencia validada.
#
def _check_visualizations(state, validation, stored):
    python = state.python_execution
    charts = {item.operation_id: item for item in python.visualizations} if python else {}
    chart_ids = {
        item.operation_id for item in validation.usable_evidence if item.kind == "visualization"
    }
    for chart in stored.visualizations:
        if chart.operation_id not in chart_ids or chart != charts.get(chart.operation_id):
            raise ConversationStorageError(
                "Una visualización no coincide con la evidencia validada."
            )


#
# Devuelve las operaciones del plan con un resumen de su ejecución para guardarlas en el historial.
#
def _stored_operations(state):
    records = {}
    for batch in (state.execution, state.python_execution):
        if batch:
            records.update({item.operation_id: item for item in batch.operations})
    if not state.planning:
        return []

    operations = []
    for step in state.planning.operations:
        value = step.model_dump(mode="json", exclude={"purpose"})
        execution = records.get(step.id)
        value["execution"] = (
            execution.model_dump(mode="json", include=_EXECUTION_FIELDS) if execution else None
        )
        operations.append(value)

    return operations


#
# Devuelve la política de uso del historial y de reutilización de evidencia del turno.
#
def _analysis_context_policy(state, turn, reused):
    resolutions = state.planning.filter_value_resolutions if state.planning else []

    return {
        "history_message_ids": [str(item) for item in turn.history_message_ids],
        "history_messages": len(turn.history),
        "history_used_by": "router,response_generator",
        "followup_resolution": "structured_history",
        "evidence_reused": reused,
        "filter_value_resolutions": [item.model_dump(mode="json") for item in resolutions],
    }


#
# Elimina las filas y exclusiones detalladas de las tablas, conservando sus metadatos.
#
def _drop_table_details(stored):
    for table in stored.results:
        if table.rows:
            table.rows = []
            table.stored_rows = 0
            table.storage = "metadata_only"
            table.complete = table.total_rows == 0
        table.exclusions = []
        table.exclusions_complete = False


#
# Retira los gráficos del snapshot, guarda su resumen y añade una nota de aviso a la respuesta.
#
def _omit_visualizations(snapshot, stored, language):
    snapshot.omitted_visualizations = [
        chart.model_dump(mode="json", include=_OMITTED_CHART_FIELDS)
        for chart in stored.visualizations
    ]
    stored.visualizations = []
    note = (
        "Los gráficos no se guardaron en la conversación por su tamaño."
        if language == "es"
        else "The charts were not kept in the conversation because of their size."
    )
    stored.segments.append(AnswerSegment(kind="note", text=note))
    stored.answer = answer_text(stored.segments)


#
# Indica si la respuesta directa no corresponde al turno o arrastra resultados del flujo de datos.
#
def _differs_from_turn(state, turn, expected_type):
    response = state.response
    routing = state.routing

    return (
        response is None
        or state.question != turn.question
        or response.response_type != expected_type
        or routing.original_question != turn.question
        or response.routing != routing
        or state.conversation_history != turn.structured_history
        or any(
            item is not None
            for item in (
                state.retrieval,
                state.planning,
                state.execution,
                state.python_execution,
                state.validation,
            )
        )
        or response.results
        or response.visualizations
        or response.validation_status is not None
    )


#
# Verifica que una respuesta general no contenga fuentes ni referencias.
#
def _check_without_references(response):
    if response.sources or response.operation_references or response.historical_message_ids:
        raise ConversationStorageError("Una respuesta general no puede inventar referencias.")


#
# Devuelve el dataset seleccionado que conserva la versión verificada de la fuente citada.
#
def _verified_dataset(selected, source):
    dataset = selected.get(source.dataset_id)
    if (
        dataset is None
        or dataset.source_id != source.source_id
        or dataset.verified_version != source.version
    ):
        raise ConversationStorageError("La fuente no conserva su versión verificada.")

    return dataset
