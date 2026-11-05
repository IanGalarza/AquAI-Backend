import asyncio
from collections.abc import Callable
from uuid import UUID

from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.application.interfaces.documents import DocumentCommands, DocumentProcessor
from aquai.application.utils.background import complete_write


class ReplaceSourceFileHandler:
    #
    # Guarda los comandos, el procesador y la notificación que usa el manejador.
    #
    def __init__(
        self,
        commands: DocumentCommands,
        processor: DocumentProcessor,
        *,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._processor = processor
        self._notify = notify

    #
    # Valida el archivo nuevo y lo deja en cola; un archivo idéntico no cambia nada.
    #
    async def handle(self, source_id: UUID, request: UploadSourceRequest) -> SourceAcceptedResponse:
        filename = await asyncio.to_thread(
            self._processor.validate_upload,
            request.filename,
            request.content,
        )
        changed = await complete_write(
            self._commands.replace_file,
            source_id,
            filename,
            request.content,
        )

        if self._notify:
            self._notify()

        return SourceAcceptedResponse(id=source_id, duplicate=not changed)
