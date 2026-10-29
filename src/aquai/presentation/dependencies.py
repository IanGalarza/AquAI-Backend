"""Dependencias HTTP de cada handler y lectura acotada de archivos."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, File, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.errors import FileTooLargeError
from aquai.application.useCases.sources import (
    GetSourceHandler,
    ListSourcesHandler,
    ActivateSourceHandler,
    UploadSourceHandler,
)
from aquai.infrastructure.providers.sources import SourceRuntime
from aquai.application.useCases.commands.createConversationHandler import CreateConversationHandler
from aquai.application.useCases.commands.sendMessageHandler import SendMessageHandler
from aquai.application.useCases.queries.conversationHistoryHandler import ConversationHistoryHandler
from aquai.infrastructure.providers.conversations import create_conversation_runtime


def _get_source_runtime(request: Request) -> SourceRuntime:
    sources = getattr(request.app.state, "sources", None)
    if sources is None:
        raise HTTPException(status_code=503, detail="La gestión de fuentes no está disponible.")
    return sources


def get_upload_source_handler(request: Request) -> UploadSourceHandler:
    return _get_source_runtime(request).upload


def get_list_sources_handler(request: Request) -> ListSourcesHandler:
    return _get_source_runtime(request).list


def get_source_handler(request: Request) -> GetSourceHandler:
    return _get_source_runtime(request).get


def get_activate_source_handler(request: Request) -> ActivateSourceHandler:
    return _get_source_runtime(request).activate


UploadSourceDependency = Annotated[UploadSourceHandler, Depends(get_upload_source_handler)]
ListSourcesDependency = Annotated[ListSourcesHandler, Depends(get_list_sources_handler)]
GetSourceDependency = Annotated[GetSourceHandler, Depends(get_source_handler)]
ActivateSourceDependency = Annotated[ActivateSourceHandler, Depends(get_activate_source_handler)]


async def read_source_file(
    request: Request,
    file: Annotated[UploadFile, File(description="Archivo original CSV o XLSX.")],
    handler: UploadSourceDependency,
) -> AsyncIterator[UploadSourceRequest]:
    form = await request.form()
    try:
        parts = list(form.multi_items())
        if request.query_params or len(parts) != 1 or parts[0][0] != "file":
            raise HTTPException(
                status_code=400,
                detail="La carga admite únicamente un archivo en el campo file, sin parámetros adicionales.",
            )
        content = await file.read(handler.max_upload_bytes + 1)
        if len(content) > handler.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño máximo permitido.")
        yield UploadSourceRequest(filename=file.filename, content=content)
    finally:
        for _, part in form.multi_items():
            if isinstance(part, StarletteUploadFile):
                await part.close()


SourceFileDependency = Annotated[UploadSourceRequest, Depends(read_source_file)]


def _get_conversation_runtime(request: Request):
    runtime = getattr(request.app.state, "conversations", None)
    if runtime is None:
        runtime = create_conversation_runtime()
        request.app.state.conversations = runtime
    return runtime


def get_conversation_history_handler(request: Request) -> ConversationHistoryHandler:
    return _get_conversation_runtime(request).history


def get_send_message_handler(request: Request) -> SendMessageHandler:
    return _get_conversation_runtime(request).send


def get_create_conversation_handler(request: Request) -> CreateConversationHandler:
    return _get_conversation_runtime(request).create


ConversationHistoryDependency = Annotated[
    ConversationHistoryHandler, Depends(get_conversation_history_handler)
]
SendMessageDependency = Annotated[SendMessageHandler, Depends(get_send_message_handler)]
CreateConversationDependency = Annotated[
    CreateConversationHandler, Depends(get_create_conversation_handler)
]
