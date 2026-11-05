from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, File, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.errors import FileTooLargeError
from aquai.application.useCases.ConversationOperation.Commands.createConversationHandler import (
    CreateConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.deleteConversationHandler import (
    DeleteConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.renameConversationHandler import (
    RenameConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Commands.sendMessageHandler import (
    SendMessageHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.getConversationHandler import (
    GetConversationHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.getMessagesHandler import (
    GetMessagesHandler,
)
from aquai.application.useCases.ConversationOperation.Queries.listConversationsHandler import (
    ListConversationsHandler,
)
from aquai.application.useCases.SourceOperation.Commands.activateSourceHandler import (
    ActivateSourceHandler,
)
from aquai.application.useCases.SourceOperation.Commands.deleteSourceHandler import (
    DeleteSourceHandler,
)
from aquai.application.useCases.SourceOperation.Commands.replaceSourceFileHandler import (
    ReplaceSourceFileHandler,
)
from aquai.application.useCases.SourceOperation.Commands.uploadSourceHandler import (
    UploadSourceHandler,
)
from aquai.application.useCases.SourceOperation.Queries.getSourceHandler import (
    GetSourceHandler,
)
from aquai.application.useCases.SourceOperation.Queries.listSourcesHandler import (
    ListSourcesHandler,
)
from aquai.infrastructure.providers.conversations import (
    ConversationRuntime,
    create_conversation_runtime,
)
from aquai.infrastructure.providers.sources import SourceRuntime

SINGLE_FILE_FIELD_ERROR = (
    "La carga admite únicamente un archivo en el campo file, sin parámetros adicionales."
)
DOCUMENTS_UNAVAILABLE_ERROR = "La gestión de documentos no está disponible."


#
# Devuelve el runtime de fuentes de la aplicación o responde 503 si no está disponible.
#
def _get_source_runtime(request: Request) -> SourceRuntime:
    sources = getattr(request.app.state, "sources", None)

    if sources is None:
        raise HTTPException(status_code=503, detail="La gestión de fuentes no está disponible.")

    return sources


#
# Devuelve el manejador que carga fuentes.
#
def get_upload_source_handler(request: Request) -> UploadSourceHandler:
    return _get_source_runtime(request).upload


#
# Devuelve el manejador que lista fuentes.
#
def get_list_sources_handler(request: Request) -> ListSourcesHandler:
    return _get_source_runtime(request).list


#
# Devuelve el manejador que consulta una fuente.
#
def get_source_handler(request: Request) -> GetSourceHandler:
    return _get_source_runtime(request).get


#
# Devuelve el manejador que activa o desactiva fuentes.
#
def get_activate_source_handler(request: Request) -> ActivateSourceHandler:
    return _get_source_runtime(request).activate


#
# Devuelve el manejador que elimina fuentes documentales o responde 503 si no está disponible.
#
def get_delete_source_handler(request: Request) -> DeleteSourceHandler:
    handler = _get_source_runtime(request).delete

    if handler is None:
        raise HTTPException(status_code=503, detail=DOCUMENTS_UNAVAILABLE_ERROR)

    return handler


#
# Devuelve el manejador que reemplaza archivos o responde 503 si no está disponible.
#
def get_replace_source_file_handler(request: Request) -> ReplaceSourceFileHandler:
    handler = _get_source_runtime(request).replace_file

    if handler is None:
        raise HTTPException(status_code=503, detail=DOCUMENTS_UNAVAILABLE_ERROR)

    return handler


UploadSourceDependency = Annotated[UploadSourceHandler, Depends(get_upload_source_handler)]
ListSourcesDependency = Annotated[ListSourcesHandler, Depends(get_list_sources_handler)]
GetSourceDependency = Annotated[GetSourceHandler, Depends(get_source_handler)]
ActivateSourceDependency = Annotated[ActivateSourceHandler, Depends(get_activate_source_handler)]
DeleteSourceDependency = Annotated[DeleteSourceHandler, Depends(get_delete_source_handler)]
ReplaceSourceFileDependency = Annotated[
    ReplaceSourceFileHandler, Depends(get_replace_source_file_handler)
]


#
# Valida que la carga sea un único archivo dentro del tamaño máximo y lo entrega como solicitud.
#
async def read_source_file(
    request: Request,
    file: Annotated[UploadFile, File(description="Archivo original CSV, XLSX, PDF o DOCX.")],
    handler: UploadSourceDependency,
) -> AsyncIterator[UploadSourceRequest]:
    form = await request.form()

    try:
        parts = list(form.multi_items())
        only_file_field = len(parts) == 1 and parts[0][0] == "file"

        if request.query_params or not only_file_field:
            raise HTTPException(status_code=400, detail=SINGLE_FILE_FIELD_ERROR)

        content = await file.read(handler.max_upload_bytes + 1)

        if len(content) > handler.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño máximo permitido.")

        yield UploadSourceRequest(filename=file.filename, content=content)
    finally:
        for _, part in form.multi_items():
            if isinstance(part, StarletteUploadFile):
                await part.close()


SourceFileDependency = Annotated[UploadSourceRequest, Depends(read_source_file)]


#
# Devuelve el runtime de conversaciones y lo crea si aún no existe.
#
def _get_conversation_runtime(request: Request) -> ConversationRuntime:
    runtime = getattr(request.app.state, "conversations", None)

    if runtime is None:
        runtime = create_conversation_runtime()
        request.app.state.conversations = runtime

    return runtime


#
# Devuelve el manejador que crea conversaciones.
#
def get_create_conversation_handler(request: Request) -> CreateConversationHandler:
    return _get_conversation_runtime(request).create


#
# Devuelve el manejador que renombra conversaciones.
#
def get_rename_conversation_handler(request: Request) -> RenameConversationHandler:
    return _get_conversation_runtime(request).rename


#
# Devuelve el manejador que elimina conversaciones.
#
def get_delete_conversation_handler(request: Request) -> DeleteConversationHandler:
    return _get_conversation_runtime(request).delete


#
# Devuelve el manejador que envía mensajes y genera respuestas.
#
def get_send_message_handler(request: Request) -> SendMessageHandler:
    return _get_conversation_runtime(request).send


#
# Devuelve el manejador que lista conversaciones.
#
def get_list_conversations_handler(request: Request) -> ListConversationsHandler:
    return _get_conversation_runtime(request).list


#
# Devuelve el manejador que consulta una conversación.
#
def get_conversation_handler(request: Request) -> GetConversationHandler:
    return _get_conversation_runtime(request).get


#
# Devuelve el manejador que consulta el historial de mensajes.
#
def get_messages_handler(request: Request) -> GetMessagesHandler:
    return _get_conversation_runtime(request).messages


CreateConversationDependency = Annotated[
    CreateConversationHandler, Depends(get_create_conversation_handler)
]
RenameConversationDependency = Annotated[
    RenameConversationHandler, Depends(get_rename_conversation_handler)
]
DeleteConversationDependency = Annotated[
    DeleteConversationHandler, Depends(get_delete_conversation_handler)
]
SendMessageDependency = Annotated[SendMessageHandler, Depends(get_send_message_handler)]
ListConversationsDependency = Annotated[
    ListConversationsHandler, Depends(get_list_conversations_handler)
]
GetConversationDependency = Annotated[GetConversationHandler, Depends(get_conversation_handler)]
GetMessagesDependency = Annotated[GetMessagesHandler, Depends(get_messages_handler)]
