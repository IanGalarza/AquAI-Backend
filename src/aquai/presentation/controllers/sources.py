"""Cuatro operaciones HTTP para originales y catálogo consultable."""
from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Query

from aquai.application.dtos.requests.activateSource import ActivateSourceRequest
from aquai.application.dtos.requests.listSources import ListSourcesRequest, SourceSort
from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.dtos.responses.listSources import SourceListResponse
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.presentation.dependencies import (
    ActivateSourceDependency, GetSourceDependency, ListSourcesDependency,
    SourceFileDependency, UploadSourceDependency,
)
from aquai.presentation.errors import ERROR_RESPONSES

router = APIRouter(prefix="/sources", tags=["Fuentes"])


@router.post("", status_code=202, response_model=SourceAcceptedResponse,
             summary="Cargar CSV/XLSX", responses=ERROR_RESPONSES,
             description="Guarda el original y prepara tablas SQL en segundo plano. "
                         "La descripción y el índice semántico tienen estados independientes.")
async def upload_source(file: SourceFileDependency,
                        handler: UploadSourceDependency) -> SourceAcceptedResponse:
    return await handler.handle(file)


@router.get("", response_model=SourceListResponse, summary="Listar fuentes",
            responses=ERROR_RESPONSES)
async def list_sources(
    handler: ListSourcesDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    name: Annotated[str | None, Query(max_length=255)] = None,
    processing_status: Annotated[ProcessingStatus | None, Query()] = None,
    sort: Annotated[SourceSort, Query(description="Orden del listado, aplicado antes de paginar.")] = SourceSort.NEWEST,
) -> SourceListResponse:
    return await handler.handle(ListSourcesRequest(
        page=page, page_size=page_size, name=name, processing_status=processing_status, sort=sort,
    ))


@router.get("/{id}", response_model=SourceDetailResponse, summary="Consultar fuente y datasets",
            responses=ERROR_RESPONSES)
async def get_source(id: UUID, handler: GetSourceDependency) -> SourceDetailResponse:
    return await handler.handle(id)


@router.patch("/{id}", response_model=SourceDetailResponse, summary="Activar o desactivar fuente",
              responses=ERROR_RESPONSES)
async def activate_source(id: UUID, request: ActivateSourceRequest,
                          handler: ActivateSourceDependency) -> SourceDetailResponse:
    return await handler.handle(id, request.is_active)
