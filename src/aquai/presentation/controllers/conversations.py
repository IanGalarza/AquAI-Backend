from collections.abc import AsyncIterable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent

from aquai.application.contracts.conversations import (
    ConversationPage,
    ConversationView,
    CreateConversationRequest,
    MessagePage,
    MessageView,
    RenameConversationRequest,
    SendMessageRequest,
)
from aquai.application.Services.streamTurnService import TurnEvent
from aquai.presentation.dependencies import (
    CancelGenerationDependency,
    CreateConversationDependency,
    DeleteConversationDependency,
    GenerationStreamDependency,
    GetConversationDependency,
    GetMessagesDependency,
    ListConversationsDependency,
    MessageStreamDependency,
    RenameConversationDependency,
    SendMessageDependency,
)
from aquai.presentation.errors import turn_error_event

router = APIRouter(prefix="/conversations", tags=["Conversaciones"])
_ERRORS = {
    404: {"description": "Conversación inexistente."},
    409: {"description": "La conversación ya tiene una generación activa o el turno ya finalizó."},
    503: {"description": "La persistencia de conversaciones no está disponible."},
}
_GENERATION_ERRORS = {
    404: {"description": "La respuesta ya no se está generando en este servidor."},
}
_STREAM_EVENTS = (
    "Eventos: turn (mensajes reservados), stage (etapa del análisis que comienza), "
    "done (resultado y mensaje guardado del asistente) y error (el turno no pudo cerrarse)."
)
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


#
# Envía una pregunta y transmite por SSE las etapas del análisis y la respuesta validada
# (POST /conversations/{id}/messages/stream).
#
@router.post(
    "/{id}/messages/stream",
    response_class=EventSourceResponse,
    responses=_ERRORS,
    summary="Enviar pregunta y seguir su progreso",
    description="Reserva el turno y lo responde en segundo plano: cerrar la conexión no lo "
    "interrumpe y la respuesta se conserva igual que en POST /conversations/{id}/messages. "
    + _STREAM_EVENTS,
)
async def stream_message(events: MessageStreamDependency) -> AsyncIterable[ServerSentEvent]:
    async for event in events.follow():
        yield _server_event(event)


#
# Transmite por SSE una respuesta en curso desde su inicio
# (GET /conversations/{id}/messages/{message_id}/events).
#
@router.get(
    "/{id}/messages/{message_id}/events",
    response_class=EventSourceResponse,
    responses=_GENERATION_ERRORS,
    summary="Seguir una respuesta en curso",
    description="Repite los eventos ya emitidos del turno y continúa con los siguientes. "
    + _STREAM_EVENTS,
)
async def follow_generation(events: GenerationStreamDependency) -> AsyncIterable[ServerSentEvent]:
    async for event in events.follow():
        yield _server_event(event)


#
# Cancela una respuesta en curso (POST /conversations/{id}/messages/{message_id}/cancel).
#
@router.post(
    "/{id}/messages/{message_id}/cancel",
    status_code=204,
    responses=_GENERATION_ERRORS,
    summary="Cancelar una respuesta en curso",
    description="El turno se cierra como cancelado y quienes lo siguen reciben el evento done.",
)
async def cancel_generation(id: UUID, message_id: UUID, handler: CancelGenerationDependency):
    await handler.handle(id, message_id)


#
# Convierte un evento del turno en un evento SSE; un error viaja con su código HTTP equivalente.
#
def _server_event(event: TurnEvent) -> ServerSentEvent:
    data = turn_error_event(event.data) if event.kind == "error" else event.data

    return ServerSentEvent(event=event.kind, data=data)
