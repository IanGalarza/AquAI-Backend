import asyncio
from collections.abc import Callable
from hashlib import sha256

from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.application.interfaces.sources import SourceCommands
from aquai.application.interfaces.tabular import TabularProcessor
from aquai.domain.models.source import Source

MIME_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


class UploadSourceHandler:
    #
    # Guarda los comandos, el procesador y el límite de tamaño de carga.
    #
    def __init__(
        self,
        commands: SourceCommands,
        processor: TabularProcessor,
        *,
        max_upload_bytes: int,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._processor = processor
        self._notify = notify
        self.max_upload_bytes = max_upload_bytes

    #
    # Valida y registra un archivo cargado como fuente, detectando duplicados.
    #
    async def handle(self, request: UploadSourceRequest) -> SourceAcceptedResponse:
        filename = await asyncio.to_thread(
            self._processor.validate_upload,
            request.filename,
            request.content,
        )
        file_type = filename.rsplit(".", 1)[-1].lower()

        source = Source(
            name=filename,
            original_filename=filename,
            file_type=file_type,
            mime_type=MIME_TYPES[file_type],
            file_size_bytes=len(request.content),
            file_content=request.content,
            content_hash=sha256(request.content).hexdigest(),
            is_active=True,
        )
        source_id, created = await asyncio.to_thread(self._commands.create_upload, source)

        if self._notify:
            self._notify()

        return SourceAcceptedResponse(id=source_id, duplicate=not created)
