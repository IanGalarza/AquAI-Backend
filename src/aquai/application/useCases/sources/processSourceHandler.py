"""Prepara y confirma SQL antes de cualquier dependencia semántica externa."""
import asyncio
from uuid import UUID
from aquai.application.errors import SemanticIndexError

from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.interfaces.tabular import SemanticIndex, SemanticProvider, TabularProcessor
from aquai.application.utils.background import complete_write


class ProcessSourceHandler:
    def __init__(self, commands: SourceCommands, queries: SourceQueries,
                 processor: TabularProcessor, semantics: SemanticProvider,
                 index: SemanticIndex) -> None:
        self._commands = commands
        self._queries = queries
        self._processor = processor
        self._semantics = semantics
        self._index = index

    async def handle(self, source_id: UUID) -> None:
        source = await asyncio.to_thread(self._queries.get, source_id, include_content=True)
        if source["datasets"]:
            await self.reindex(source_id)
            return
        prepared = await asyncio.to_thread(
            self._processor.prepare, source["original_filename"], source["file_content"],
        )
        warnings = list(dict.fromkeys(
            prepared.warnings + [warning for table in prepared.tables for warning in table.warnings]
        ))
        report = {
            "summary": f"{len(prepared.tables)} conjuntos preparados.",
            "warnings": [item[:500] for item in warnings[:100]],
            "warning_count": len(warnings),
            "errors": [],
            "dataset_count": len(prepared.tables),
            "row_count": sum(len(table.rows) for table in prepared.tables),
            "steps": ["original_saved", "tables_prepared", "sql_persisted"],
        }
        await complete_write(self._commands.save_prepared, source_id, prepared.tables, report)
        if not prepared.tables:
            return

        # El catálogo usa el mismo orden y locator que las tablas preparadas.
        persisted = await asyncio.to_thread(self._queries.get, source_id)
        for table in prepared.tables:
            dataset = next(
                item for item in persisted["datasets"] if item["source_locator"] == table.source_locator
            )
            try:
                result = await self._semantics.describe(table)
                description, columns, semantic_warnings = (
                    result.description, result.columns or table.columns, result.warnings,
                )
            except Exception as error:
                description, columns = dataset["description"], table.columns
                semantic_warnings = [
                    f"Descripción automática no disponible ({type(error).__name__}); "
                    "se conserva la descripción estructural."
                ]
            await complete_write(
                self._commands.save_dataset_semantics, dataset["id"],
                description, columns, warnings=semantic_warnings,
            )
        await self.reindex(source_id)

    async def reindex(self, source_id: UUID) -> None:
        await complete_write(self._commands.set_index_status, source_id, "processing")
        try:
            source = await asyncio.to_thread(self._queries.get, source_id)
            await self._index.index_source(source)
        except asyncio.CancelledError:
            await complete_write(
                self._commands.set_index_status, source_id, "pending",
                error="Indexación interrumpida; puede reconstruirse desde PostgreSQL.",
            )
            raise
        except Exception as error:
            detail = str(error) if isinstance(error, SemanticIndexError) else type(error).__name__
            await complete_write(
                self._commands.set_index_status, source_id, "failed",
                error=f"Índice semántico no disponible: {detail}",
            )
        else:
            await complete_write(
                self._commands.set_index_status, source_id, "ready",
                expected_active=source["is_active"],
            )
