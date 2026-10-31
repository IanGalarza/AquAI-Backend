import asyncio
from uuid import UUID

from aquai.application.errors import SemanticIndexError
from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.interfaces.tabular import (
    SemanticIndex,
    SemanticProvider,
    TabularProcessor,
)
from aquai.application.utils.background import complete_write


class ProcessSourceService:
    def __init__(
        self,
        commands: SourceCommands,
        queries: SourceQueries,
        processor: TabularProcessor,
        semantics: SemanticProvider,
        index: SemanticIndex,
    ) -> None:
        self._commands = commands
        self._queries = queries
        self._processor = processor
        self._semantics = semantics
        self._index = index

    async def process(self, source_id: UUID, *, reprocess: bool = False) -> None:
        source = await asyncio.to_thread(self._queries.get, source_id, include_content=True)
        if source["datasets"] and not reprocess:
            await self.reindex(source_id)
            return

        prepared = await asyncio.to_thread(
            self._processor.prepare,
            source["original_filename"],
            source["file_content"],
        )

        structure_report = {"attempted": False, "suggestions": 0, "accepted": 0}
        if prepared.structure_candidates:
            prepared = await self._assist_structure(source, prepared, structure_report)

        report = _preparation_report(prepared, structure_report)
        await self._save_prepared(source_id, prepared.tables, report, reprocess=reprocess)

        if not prepared.tables:
            return

        await self._describe_datasets(source_id, prepared.tables)
        await self.reindex(source_id)

    async def reindex(self, source_id: UUID) -> None:
        await complete_write(self._commands.set_index_status, source_id, "processing")

        try:
            source = await asyncio.to_thread(self._queries.get, source_id)
            await self._index.index_source(source)
        except asyncio.CancelledError:
            await complete_write(
                self._commands.set_index_status,
                source_id,
                "pending",
                error="Indexación interrumpida; puede reconstruirse desde PostgreSQL.",
            )
            raise
        except Exception as error:
            detail = str(error) if isinstance(error, SemanticIndexError) else type(error).__name__
            await complete_write(
                self._commands.set_index_status,
                source_id,
                "failed",
                error=f"Índice semántico no disponible: {detail}",
            )
        else:
            await complete_write(
                self._commands.set_index_status,
                source_id,
                "ready",
                expected_active=source["is_active"],
            )

    async def _assist_structure(self, source, prepared, structure_report):
        structure_report["attempted"] = True

        try:
            suggestions = await self._semantics.suggest_structure(prepared.structure_candidates)
            structure_report["suggestions"] = len(suggestions)

            if suggestions:
                prepared = await asyncio.to_thread(
                    self._processor.prepare,
                    source["original_filename"],
                    source["file_content"],
                    suggestions,
                )

            structure_report["accepted"] = sum(
                "structure_assistance" in table.source_locator for table in prepared.tables
            )
            if structure_report["accepted"] < len(prepared.structure_candidates):
                prepared.warnings.append(
                    "Persisten regiones cuya estructura no pudo verificarse; se conservan en el original."
                )
        except Exception as error:
            prepared.warnings.append(
                f"Asistencia estructural no disponible ({type(error).__name__}); se conserva la preparación determinística."
            )

        return prepared

    async def _save_prepared(self, source_id, tables, report, *, reprocess) -> None:
        if reprocess:
            await complete_write(
                self._commands.save_prepared,
                source_id,
                tables,
                report,
                replace_existing=True,
            )
            return

        await complete_write(self._commands.save_prepared, source_id, tables, report)

    async def _describe_datasets(self, source_id, tables) -> None:
        # El catálogo usa el mismo orden y locator que las tablas preparadas.
        persisted = await asyncio.to_thread(self._queries.get, source_id)

        for table in tables:
            dataset = _persisted_dataset(persisted, table)
            description, columns, semantic_warnings = await self._describe(table, dataset)
            await complete_write(
                self._commands.save_dataset_semantics,
                dataset["id"],
                description,
                columns,
                warnings=semantic_warnings,
            )

    async def _describe(self, table, dataset):
        try:
            result = await self._semantics.describe(table)
            return result.description, result.columns or table.columns, result.warnings
        except Exception as error:
            fallback_warning = (
                f"Descripción automática no disponible ({type(error).__name__}); "
                "se conserva la descripción estructural."
            )
            return dataset["description"], table.columns, [fallback_warning]


def _preparation_report(prepared, structure_report) -> dict:
    warnings = _unique_warnings(prepared)

    return {
        "summary": f"{len(prepared.tables)} conjuntos preparados.",
        "warnings": [item[:500] for item in warnings[:100]],
        "warning_count": len(warnings),
        "errors": [],
        "dataset_count": len(prepared.tables),
        "row_count": sum(len(table.rows) for table in prepared.tables),
        "steps": ["original_saved", "tables_prepared", "sql_persisted"],
        "structure_assistance": structure_report,
    }


def _unique_warnings(prepared) -> list:
    warnings = list(prepared.warnings)
    for table in prepared.tables:
        warnings.extend(table.warnings)

    return list(dict.fromkeys(warnings))


def _persisted_dataset(persisted, table) -> dict:
    return next(
        item for item in persisted["datasets"] if item["source_locator"] == table.source_locator
    )
