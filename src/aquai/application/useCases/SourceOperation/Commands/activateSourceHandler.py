import asyncio
from collections.abc import Callable
from uuid import UUID

from aquai.application.dtos.responses.activateSource import SourceActivationResponse
from aquai.application.interfaces.sources import SourceCommands
from aquai.application.Services.sourceDetailService import SourceDetailService


class ActivateSourceHandler:
    def __init__(
        self,
        commands: SourceCommands,
        details: SourceDetailService,
        *,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._details = details
        self._notify = notify

    async def handle(self, source_id: UUID, active: bool) -> SourceActivationResponse:
        await asyncio.to_thread(self._commands.set_active, source_id, active)

        if self._notify:
            self._notify()

        detail = await self._details.get(source_id)

        return SourceActivationResponse.model_validate(detail)
