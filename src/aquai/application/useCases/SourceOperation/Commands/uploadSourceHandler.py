import asyncio
from collections.abc import Callable
from hashlib import sha256

from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.application.interfaces.documents import DocumentProcessor
from aquai.application.interfaces.sources import SourceCommands
from aquai.application.interfaces.tabular import TabularProcessor
from aquai.application.Rules.documents import (
    DOCUMENT_FILE_TYPES,
    DOCUMENT_MIME_TYPES,
    file_suffix,
)
from aquai.domain.models.source import Source

MIME_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    **DOCUMENT_MIME_TYPES,
}


class UploadSourceHandler:
    #
    # Guarda los comandos, los procesadores y el límite de tamaño de carga.
    #
    def __init__(
        self,
        commands: SourceCommands,
        processor: TabularProcessor,
        *,
        max_upload_bytes: int,
        notify: Callable[[], None] | None = None,
        documents: DocumentProcessor | None = None,
    ) -> None:
        self._commands = commands
        self._processor = processor
        self._documents = documents
        self._notify = notify
        self.max_upload_bytes = max_upload_bytes

    #
    # Valida y registra un archivo cargado como fuente, detectando duplicados.
    #
    async def handle(self, request: UploadSourceRequest) -> SourceAcceptedResponse:
        filename = await asyncio.to_thread(self._validate, request.filename, request.content)
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

    #
    # Valida el archivo con el procesador que corresponde a su extensión.
    #
    def _validate(self, filename: str | None, content: bytes) -> str:
        if self._documents is not None and file_suffix(filename) in DOCUMENT_FILE_TYPES:
            return self._documents.validate_upload(filename, content)

        return self._processor.validate_upload(filename, content)
