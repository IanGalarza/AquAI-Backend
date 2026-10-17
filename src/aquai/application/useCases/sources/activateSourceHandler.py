"""La activación expresa preferencia; la consultabilidad depende del catálogo SQL."""
import asyncio
from collections.abc import Callable
from uuid import UUID

from aquai.application.dtos.responses.activateSource import SourceActivationResponse
from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.useCases.sources.getSourceHandler import GetSourceHandler


class ActivateSourceHandler:
    def __init__(self, commands: SourceCommands, queries: SourceQueries,
                 *, notify: Callable[[], None] | None = None) -> None:
        self._commands = commands
        self._queries = queries
        self._notify = notify

    async def handle(self, source_id: UUID, active: bool) -> SourceActivationResponse:
        await asyncio.to_thread(self._commands.set_active, source_id, active)
        if self._notify:
            self._notify()
        detail = await GetSourceHandler(self._queries).handle(source_id)
        return SourceActivationResponse.model_validate(detail)
