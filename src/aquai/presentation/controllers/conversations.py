"""API REST de historial; los GET sólo consultan PostgreSQL."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Response

from aquai.application.contracts.conversations import (
    ConversationPage, ConversationView, CreateConversationRequest, MessagePage, MessageView, SendMessageRequest,
)
from aquai.presentation.dependencies import ConversationHistoryDependency, SendMessageDependency


router = APIRouter(prefix="/conversations", tags=["Conversaciones"])
_ERRORS = {404: {"description": "Conversación inexistente."},
           409: {"description": "La conversación ya tiene una generación activa o el turno ya finalizó."},
           503: {"description": "La persistencia de conversaciones no está disponible."}}


@router.post("", status_code=201, response_model=ConversationView, responses=_ERRORS,
             summary="Crear conversación")
async def create_conversation(request: CreateConversationRequest, handler: ConversationHistoryDependency):
    return await handler.create(request)


@router.get("", response_model=ConversationPage, responses=_ERRORS, summary="Listar conversaciones")
async def list_conversations(handler: ConversationHistoryDependency,
        page: Annotated[int, Query(ge=1)] = 1, page_size: Annotated[int, Query(ge=1, le=100)] = 20):
    return await handler.list(page=page, page_size=page_size)


@router.get("/{id}", response_model=ConversationView, responses=_ERRORS, summary="Obtener conversación")
async def get_conversation(id: UUID, handler: ConversationHistoryDependency):
    return await handler.get(id)


@router.get("/{id}/messages", response_model=MessagePage, responses=_ERRORS, summary="Obtener historial")
async def get_messages(id: UUID, handler: ConversationHistoryDependency,
        page: Annotated[int, Query(ge=1)] = 1, page_size: Annotated[int, Query(ge=1, le=20)] = 10):
    return await handler.messages(id, page=page, page_size=page_size)


@router.post("/{id}/messages", status_code=201, response_model=MessageView,
    responses={**_ERRORS, 500: {"model": MessageView, "description": "Fallo de ejecución registrado en el historial."},
               502: {"model": MessageView, "description": "Fallo de generación registrado en el historial."},
               504: {"model": MessageView, "description": "Tiempo de generación agotado; el turno permanece registrado."}},
    summary="Enviar pregunta y esperar respuesta",
    description="Genera una respuesta y persiste el turno. Las preguntas deben ser autocontenidas; "
                "el historial actual no resuelve referencias como 'ahora mostralo'. No hay aislamiento entre usuarios.")
async def send_message(id: UUID, request: SendMessageRequest, response: Response, handler: SendMessageDependency):
    result = await handler.handle(id, request)
    response.status_code = {"completed": 201, "generation_failed": 502, "execution_failed": 500, "timed_out": 504}[result.outcome]
    return result.message
