import asyncio
from collections.abc import Callable
from uuid import UUID

from aquai.application.errors import (
    SemanticIndexError,
    SourceConflictError,
    SourcesUnavailableError,
)
from aquai.application.interfaces.documents import DocumentCommands, DocumentIndex
from aquai.application.interfaces.sources import SourceCommands
from aquai.application.utils.background import complete_write


class DeleteSourceHandler:
    #
    # Guarda los comandos, el índice y la notificación que usa el manejador.
    #
    def __init__(
        self,
        commands: DocumentCommands,
        sources: SourceCommands,
        index: DocumentIndex,
        *,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._sources = sources
        self._index = index
        self._notify = notify

    #
    # Elimina primero el índice: si Qdrant falla, la fuente y su contenido quedan intactos.
    #
    async def handle(self, source_id: UUID) -> None:
        await asyncio.to_thread(self._commands.ensure_removable, source_id)

        try:
            await self._index.delete_source(source_id)
        except SemanticIndexError as error:
            raise SourcesUnavailableError("No se pudo eliminar la fuente del índice.") from error

        try:
            await complete_write(self._commands.delete_source, source_id)
        except SourceConflictError:
            await complete_write(self._sources.request_reindex, source_id)
            raise
        finally:
            if self._notify:
                self._notify()
