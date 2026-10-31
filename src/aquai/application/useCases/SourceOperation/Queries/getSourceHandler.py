from uuid import UUID

from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.Services.sourceDetailService import SourceDetailService


class GetSourceHandler:
    def __init__(self, details: SourceDetailService) -> None:
        self._details = details

    async def handle(self, source_id: UUID) -> SourceDetailResponse:
        return await self._details.get(source_id)
