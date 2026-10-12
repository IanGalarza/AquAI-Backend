"""Adaptación HTTP de los casos de uso de fuentes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from aquai.presentation.dependencies import (
    GetSourceDependency,
    ListSourcesDependency,
    ReviewSourceDependency,
    SourceFileDependency,
    UploadSourceDependency,
)
from aquai.presentation.errors import ERROR_RESPONSES
from aquai.presentation.schemas.sources import (
    ProcessingStatus,
    SourceAcceptedResponse,
    SourceDecisionRequest,
    SourceDecisionResponse,
    SourceDetailResponse,
    SourceListResponse,
    ValidationStatus,
)


router = APIRouter(prefix="/sources", tags=["Fuentes"])


@router.post(
    "",
    status_code=202,
    response_model=SourceAcceptedResponse,
    summary="Cargar una fuente",
    description=(
        "Recibe únicamente el archivo CSV/XLSX. Guarda el original antes de responder "
        "y continúa en segundo plano: inspección, receta, ejecución y validación. "
        "Modelo, contexto y reglas se resuelven en el backend."
    ),
    response_description="Original y estado inicial persistidos.",
    responses={
        **ERROR_RESPONSES,
        413: {"description": "El original supera el tamaño permitido."},
    },
)
async def upload_source(
    file: SourceFileDependency,
    handler: UploadSourceDependency,
) -> dict:
    return await handler.handle(file.filename, file.content)


@router.get(
    "",
    response_model=SourceListResponse,
    summary="Listar fuentes",
    responses=ERROR_RESPONSES,
)
async def list_sources(
    handler: ListSourcesDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    name: Annotated[
        str | None, Query(max_length=255, description="Búsqueda parcial por nombre.")
    ] = None,
    processing_status: Annotated[ProcessingStatus | None, Query()] = None,
    validation_status: Annotated[ValidationStatus | None, Query()] = None,
) -> dict:
    return await handler.handle(
        page=page,
        page_size=page_size,
        name=name,
        processing_status=processing_status,
        validation_status=validation_status,
    )


@router.get(
    "/{id}",
    response_model=SourceDetailResponse,
    summary="Consultar una fuente",
    responses=ERROR_RESPONSES,
)
async def get_source(id: UUID, handler: GetSourceDependency) -> dict:
    return await handler.handle(id)


@router.post(
    "/{id}/reviews/{review_id}/decisions",
    status_code=202,
    response_model=SourceDecisionResponse,
    summary="Resolver una revisión vigente",
    description=(
        "Registra una decisión con alcance de esta fuente. Conserva la propuesta original, "
        "reanuda las etapas necesarias y vuelve a validar. Una decisión idéntica "
        "reenviada no duplica su efecto. Las opciones admitidas se obtienen del detalle de la fuente."
    ),
    response_description="Decisión registrada; reanudación pendiente.",
    responses=ERROR_RESPONSES,
)
async def decide_source_review(
    id: UUID,
    review_id: str,
    decision: SourceDecisionRequest,
    handler: ReviewSourceDependency,
) -> dict:
    return await handler.handle(id, review_id, decision.action, decision.values)
