from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from aquai.application.dtos.requests.activateSource import ActivateSourceRequest
from aquai.application.dtos.requests.listSources import ListSourcesRequest, SourceSort
from aquai.application.dtos.responses.activateSource import SourceActivationResponse
from aquai.application.dtos.responses.getSource import SourceDetailResponse
from aquai.application.dtos.responses.listSources import SourceListResponse
from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.presentation.dependencies import (
    ActivateSourceDependency,
    DeleteSourceDependency,
    GetSourceDependency,
    ListSourcesDependency,
    ReplaceSourceFileDependency,
    SourceFileDependency,
    UploadSourceDependency,
)
from aquai.presentation.errors import ERROR_RESPONSES

router = APIRouter(prefix="/sources", tags=["Fuentes"])


#
# Carga un archivo CSV, XLSX, PDF o DOCX como fuente (POST /sources).
#
@router.post(
    "",
    status_code=202,
    response_model=SourceAcceptedResponse,
    summary="Cargar CSV, XLSX, PDF o DOCX",
    responses=ERROR_RESPONSES,
    description="Guarda el original y lo procesa en segundo plano: tablas SQL para planillas, "
    "fragmentos indexados para documentos. El índice semántico tiene un estado independiente.",
)
async def upload_source(
    file: SourceFileDependency, handler: UploadSourceDependency
) -> SourceAcceptedResponse:
    return await handler.handle(file)


#
# Lista las fuentes cargadas de forma paginada y filtrable (GET /sources).
#
@router.get(
    "", response_model=SourceListResponse, summary="Listar fuentes", responses=ERROR_RESPONSES
)
async def list_sources(
    handler: ListSourcesDependency,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    name: Annotated[str | None, Query(max_length=255)] = None,
    processing_status: Annotated[ProcessingStatus | None, Query()] = None,
    sort: Annotated[
        SourceSort, Query(description="Orden del listado, aplicado antes de paginar.")
    ] = SourceSort.NEWEST,
) -> SourceListResponse:
    return await handler.handle(
        ListSourcesRequest(
            page=page,
            page_size=page_size,
            name=name,
            processing_status=processing_status,
            sort=sort,
        )
    )


#
# Consulta una fuente y sus datasets (GET /sources/{id}).
#
@router.get(
    "/{id}",
    response_model=SourceDetailResponse,
    summary="Consultar fuente y datasets",
    responses=ERROR_RESPONSES,
)
async def get_source(id: UUID, handler: GetSourceDependency) -> SourceDetailResponse:
    return await handler.handle(id)


#
# Activa o desactiva una fuente (PATCH /sources/{id}).
#
@router.patch(
    "/{id}",
    response_model=SourceActivationResponse,
    summary="Activar o desactivar fuente",
    responses=ERROR_RESPONSES,
)
async def activate_source(
    id: UUID, request: ActivateSourceRequest, handler: ActivateSourceDependency
) -> SourceActivationResponse:
    return await handler.handle(id, request.is_active)


#
# Reemplaza el archivo de una fuente documental y la vuelve a procesar (PUT /sources/{id}/file).
#
@router.put(
    "/{id}/file",
    status_code=202,
    response_model=SourceAcceptedResponse,
    summary="Reemplazar el archivo de un documento",
    responses=ERROR_RESPONSES,
    description="Sólo para PDF y DOCX. El contenido extraído anterior se elimina y el nuevo "
    "original se procesa en segundo plano. Las planillas conservan su original.",
)
async def replace_source_file(
    id: UUID, file: SourceFileDependency, handler: ReplaceSourceFileDependency
) -> SourceAcceptedResponse:
    return await handler.handle(id, file)


#
# Elimina una fuente documental con su contenido y sus puntos del índice (DELETE /sources/{id}).
#
@router.delete(
    "/{id}",
    status_code=204,
    summary="Eliminar un documento",
    responses=ERROR_RESPONSES,
    description="Sólo para PDF y DOCX. Las planillas no se eliminan por esta vía.",
)
async def delete_source(id: UUID, handler: DeleteSourceDependency) -> None:
    await handler.handle(id)
