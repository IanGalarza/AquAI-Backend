import asyncio
from uuid import UUID

from aquai.application.errors import SemanticIndexError
from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.interfaces.tabular import SemanticIndex
from aquai.application.utils.background import complete_write


class IndexSourceService:
    #
    # Guarda los puertos de comandos, consultas e índice que usa el servicio.
    #
    def __init__(
        self, commands: SourceCommands, queries: SourceQueries, index: SemanticIndex
    ) -> None:
        self._commands = commands
        self._queries = queries
        self._index = index

    #
    # Reconstruye el índice semántico de una fuente y registra su estado.
    #
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
