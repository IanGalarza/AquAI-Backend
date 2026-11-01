from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response

from aquai.application.contracts.conversations import (
    ConversationPage,
    ConversationView,
    CreateConversationRequest,
    MessagePage,
    MessageView,
    RenameConversationRequest,
    SendMessageRequest,
)
from aquai.presentation.dependencies import (
    CreateConversationDependency,
    DeleteConversationDependency,
    GetConversationDependency,
    GetMessagesDependency,
    ListConversationsDependency,
    RenameConversationDependency,
    SendMessageDependency,
)

router = APIRouter(prefix="/conversations", tags=["Conversaciones"])
_ERRORS = {
    404: {"description": "Conversación inexistente."},
    409: {"description": "La conversación ya tiene una generación activa o el turno ya finalizó."},
    503: {"description": "La persistencia de conversaciones no está disponible."},
}
_OUTCOME_STATUS_CODES = {
    "completed": 201,
    "generation_failed": 502,
    "execution_failed": 500,
    "timed_out": 504,
}


#
# Crea una conversación (POST /conversations).
#
@router.post(
    "",
    status_code=201,
    response_model=ConversationView,
    responses=_ERRORS,
    summary="Crear conversación",
)
async def create_conversation(
    request: CreateConversationRequest,
    handler: CreateConversationDependency,
):
    return await handler.handle(request)


#
# Lista las conversaciones de forma paginada (GET /conversations).
#
@router.get("", response_model=ConversationPage, responses=_ERRORS, summary="Listar conversaciones")
async def list_conversations(
    handler: ListConversationsDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return await handler.handle(page=page, page_size=page_size)


#
# Obtiene una conversación (GET /conversations/{id}).
#
@router.get(
    "/{id}", response_model=ConversationView, responses=_ERRORS, summary="Obtener conversación"
)
async def get_conversation(id: UUID, handler: GetConversationDependency):
    return await handler.handle(id)


#
# Renombra una conversación (PATCH /conversations/{id}).
#
@router.patch(
    "/{id}", response_model=ConversationView, responses=_ERRORS, summary="Renombrar conversación"
)
async def rename_conversation(
    id: UUID,
    request: RenameConversationRequest,
    handler: RenameConversationDependency,
):
    return await handler.handle(id, request)


#
# Elimina una conversación con sus mensajes (DELETE /conversations/{id}).
#
@router.delete(
    "/{id}",
    status_code=204,
    responses=_ERRORS,
    summary="Eliminar conversación",
    description="Elimina la conversación con sus mensajes. Se rechaza mientras se genera una respuesta.",
)
async def delete_conversation(id: UUID, handler: DeleteConversationDependency):
    await handler.handle(id)


#
# Obtiene el historial paginado de mensajes de una conversación (GET /conversations/{id}/messages).
#
@router.get(
    "/{id}/messages", response_model=MessagePage, responses=_ERRORS, summary="Obtener historial"
)
async def get_messages(
    id: UUID,
    handler: GetMessagesDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=20)] = 10,
):
    return await handler.handle(id, page=page, page_size=page_size)


#
# Envía una pregunta y devuelve su respuesta (POST /conversations/{id}/messages).
#
@router.post(
    "/{id}/messages",
    status_code=201,
    response_model=MessageView,
    responses={
        **_ERRORS,
        500: {"model": MessageView, "description": "Fallo de ejecución registrado."},
        502: {"model": MessageView, "description": "Fallo de generación registrado."},
        504: {"model": MessageView, "description": "Tiempo agotado; el turno sigue registrado."},
    },
    summary="Enviar pregunta y esperar respuesta",
    description="Interpreta la pregunta y su contexto persistido, responde y conserva el turno. "
    "Las referencias ambiguas requieren aclaración. El historial es compartido.",
)
async def send_message(
    id: UUID,
    request: SendMessageRequest,
    response: Response,
    handler: SendMessageDependency,
):
    result = await handler.handle(id, request)
    response.status_code = _OUTCOME_STATUS_CODES[result.outcome]

    return result.message
