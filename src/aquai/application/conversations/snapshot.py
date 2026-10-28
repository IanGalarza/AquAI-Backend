"""Fotografía JSON de una respuesta autorizada: sin prompts ni referencias efímeras."""

import json

from aquai.application.contracts.conversations import (
    AssistantRecord, AssistantSnapshot, SourceAssociation, StoredGeneration, StoredQueryResponse, StoredTable,
)
from aquai.application.errors import ConversationStorageError
from aquai.application.execution import evidence_fingerprint


def _bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def failure_record(turn, code, message, *, cancelled=False):
    return AssistantRecord(content=message, content_data={"schema_version": 1, "kind": "generation_failure",
        "user_message_id": str(turn.user_message_id), "error": {"code": code, "message": message}},
        sources=[], status_name="cancelled" if cancelled else "failed")


class ConversationSnapshotBuilder:
    def __init__(self, *, max_message_bytes, max_table_rows, max_table_bytes, max_source_details_bytes):
        self._max_message_bytes = max_message_bytes
        self._max_table_rows = max_table_rows
        self._max_table_bytes = max_table_bytes
        self._max_source_details_bytes = max_source_details_bytes

    def build(self, state, turn, *, reused=False):
        response, validation = state.response, state.validation
        if (response is None or validation is None or state.question != turn.question or
                validation.evidence_fingerprint != evidence_fingerprint(state.planning, state.execution, state.python_execution)):
            raise ConversationStorageError("La respuesta no conserva la evidencia validada de este turno.")
        if not reused and state.conversation_context != turn.history:
            raise ConversationStorageError("El resultado no conserva el contexto enviado al grafo.")
        allowed = {item.evidence_id: item for item in validation.usable_evidence}
        if response.sources != validation.sources or response.validation_status != validation.status:
            raise ConversationStorageError("Las fuentes no corresponden a la respuesta validada.")
        for reference in response.operation_references:
            item = allowed.get(reference.evidence_id)
            if item is None or (reference.operation_id, reference.tool, reference.dataset_ids) != (item.operation_id, item.tool, item.dataset_ids):
                raise ConversationStorageError("La respuesta contiene referencias no autorizadas.")
        generation = response.generation.model_dump(exclude={"context"})
        if response.generation.context is not None:
            context = response.generation.context
            generation["context_record"] = context.model_dump(include={
                "prompt_version", "sha256", "size_bytes", "budget_bytes", "omitted_rows", "sent_to_model",
            })
        data = response.model_dump(exclude={"generation", "results"})
        stored = StoredQueryResponse(**data, generation=StoredGeneration(**generation), results=[])
        warnings = []
        remaining_rows, remaining_bytes = self._max_table_rows, self._max_table_bytes
        for reference in response.results:
            evidence = allowed.get(reference.evidence_id)
            if evidence is None or evidence.kind != "table" or evidence.operation_id != reference.operation_id:
                raise ConversationStorageError("La tabla no tiene evidencia autorizada.")
            batch = state.execution if evidence.tool == "sql" else state.python_execution
            record = next((item for item in batch.operations if item.operation_id == evidence.operation_id), None)
            if record is None or not record.materialized or record.truncated or record.rows_returned != len(record.rows):
                raise ConversationStorageError("No se conserva una tabla completa para la respuesta.")
            serialized = record.model_dump(mode="json")
            rows = []
            for row in serialized["rows"]:
                size = _bytes(row) + 1
                if remaining_rows <= 0 or size > remaining_bytes:
                    break
                rows.append(row)
                remaining_rows -= 1
                remaining_bytes -= size
            complete = len(rows) == record.rows_returned
            table = StoredTable(evidence_id=evidence.evidence_id, operation_id=evidence.operation_id, columns=record.columns,
                rows=rows, total_rows=record.rows_returned, stored_rows=len(rows),
                storage="inline" if complete else "preview" if rows else "metadata_only", complete=complete,
                warnings=list(record.warnings), rows_processed=record.rows_processed, rows_filtered=record.rows_filtered,
                exclusions=record.exclusions)
            if _bytes([item.model_dump(mode="json") for item in table.exclusions]) > remaining_bytes:
                table.exclusions, table.exclusions_complete = [], False
                table.warnings.append("El detalle de exclusiones excede el límite del historial; no se conservó completo.")
            else:
                remaining_bytes -= _bytes([item.model_dump(mode="json") for item in table.exclusions])
            if not complete:
                warnings.append(f"La tabla {evidence.evidence_id} conserva {len(rows)} de {record.rows_returned} filas; las restantes no están almacenadas ni disponibles mediante un recurso HTTP.")
            stored.results.append(table)
        # Cada referencia de figura debe apuntar al objeto original autorizado.
        charts = {item.operation_id: item for item in state.python_execution.visualizations} if state.python_execution else {}
        chart_ids = {item.operation_id for item in validation.usable_evidence if item.kind == "visualization"}
        if any(chart.operation_id not in chart_ids or chart != charts.get(chart.operation_id) for chart in stored.visualizations):
            raise ConversationStorageError("Una visualización no coincide con la evidencia validada.")
        records = {}
        for batch in (state.execution, state.python_execution):
            if batch:
                records.update({item.operation_id: item for item in batch.operations})
        operations = []
        if state.planning:
            for step in state.planning.operations:
                value = step.model_dump(mode="json", exclude={"purpose"})
                execution = records.get(step.id)
                value["execution"] = execution.model_dump(mode="json", include={
                    "status", "materialized", "truncated", "rows_returned", "rows_processed", "rows_filtered", "duration_ms", "error",
                }) if execution else None
                operations.append(value)
        snapshot = AssistantSnapshot(response=stored, operations=operations,
            context_policy={"history_message_ids": [str(item) for item in turn.history_message_ids],
                "history_messages": len(turn.history), "history_used_by": "response_generator",
                "followup_resolution": "not_supported", "question_requirement": "standalone", "evidence_reused": reused},
            validated_at=validation.verified_at)

        def size():
            return len(snapshot.model_dump_json().encode("utf-8"))

        if size() > self._max_message_bytes:
            for table in stored.results:
                if table.rows:
                    table.rows, table.stored_rows = [], 0
                    table.storage, table.complete = "metadata_only", table.total_rows == 0
                table.exclusions, table.exclusions_complete = [], False
            warnings.append("Las tablas y exclusiones detalladas se omitieron por el límite del historial; se conservan sus metadatos.")
        if size() > self._max_message_bytes:
            snapshot.operations = [{key: value[key] for key in ("id", "tool", "operation", "depends_on", "execution")} for value in operations]
            warnings.append("Se conserva un resumen de las operaciones por el límite del historial.")
        if size() > self._max_message_bytes:
            snapshot.omitted_visualizations = [chart.model_dump(mode="json", include={
                "operation_id", "chart_type", "title", "datasets", "input_operation_ids", "input_rows", "excluded_rows",
            }) for chart in stored.visualizations]
            stored.visualizations = []
            warnings.append("Las figuras exceden el límite del historial y no se conservaron; no existe un archivo o enlace para recuperarlas.")
        if warnings:
            snapshot.storage_warnings = list(dict.fromkeys(warnings))
            stored.limitations = list(dict.fromkeys([*stored.limitations, *snapshot.storage_warnings]))
            stored.answer += "\n\n" + " ".join(snapshot.storage_warnings)
        if size() > self._max_message_bytes:
            raise ConversationStorageError("Los metadatos mínimos exceden el límite de almacenamiento del mensaje.")
        associations = self._associations(state, response)
        return AssistantRecord(content=stored.answer,
            content_data=snapshot.model_dump(mode="json", exclude={"response": {"generation": {"context"}}}),
            sources=associations, status_name="failed" if response.generation.status == "failed" else "completed")

    def _associations(self, state, response):
        selected = {item.dataset_id: item for item in state.planning.selected_datasets} if state.planning else {}
        result, seen = [], set()
        for source in response.sources:
            key = (source.source_id, source.dataset_id)
            if key in seen:
                continue
            seen.add(key)
            dataset = selected.get(source.dataset_id)
            if dataset is None or dataset.source_id != source.source_id or dataset.verified_version != source.version:
                raise ConversationStorageError("La fuente no conserva su versión verificada.")
            references = [item for item in response.operation_references if source.dataset_id in item.dataset_ids]
            details = {"schema_version": 1, "dataset_version": source.version, "dataset_name": source.dataset_name,
                "provenance": source.provenance.model_dump(mode="json"),
                "operation_references": [item.model_dump(mode="json") for item in references],
                "columns": [], "columns_complete": True}
            if _bytes(details) > self._max_source_details_bytes:
                raise ConversationStorageError("La procedencia mínima excede el límite de una referencia.")
            for column in dataset.columns_metadata:
                value = column.model_dump(mode="json", include={"sql_name", "original_name", "sql_type", "unit", "unit_evidence",
                    "numeric_eligible", "derived_from", "representation", "excluded_count", "source_column", "source_row"})
                details["columns"].append(value)
                if _bytes(details) > self._max_source_details_bytes:
                    details["columns"].pop()
                    details["columns_complete"] = False
                    break
            if _bytes(details) > self._max_source_details_bytes:
                raise ConversationStorageError("La referencia excede su límite de metadatos.")
            result.append(SourceAssociation(source.source_id, source.dataset_id, len(result) + 1, details))
        return result
