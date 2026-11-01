from uuid import UUID

from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.Services.sourceDetailService import SourceDetailService


class GetSourceHandler:
    #
    # Guarda el servicio de detalle de fuentes que usa el manejador.
    #
    def __init__(self, details: SourceDetailService) -> None:
        self._details = details

    #
    # Devuelve el detalle de una fuente.
    #
    async def handle(self, source_id: UUID) -> SourceDetailResponse:
        return await self._details.get(source_id)
