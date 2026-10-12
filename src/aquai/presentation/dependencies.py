"""Dependencias HTTP de cada handler y lectura acotada de archivos."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, File, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from aquai.application.errors import FileTooLargeError
from aquai.application.useCases.sources import (
    GetSourceHandler,
    ListSourcesHandler,
    ReviewSourceHandler,
    UploadSourceHandler,
)
from aquai.infrastructure.providers.sources import SourceRuntime


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


def get_review_source_handler(request: Request) -> ReviewSourceHandler:
    return _get_source_runtime(request).review


UploadSourceDependency = Annotated[UploadSourceHandler, Depends(get_upload_source_handler)]
ListSourcesDependency = Annotated[ListSourcesHandler, Depends(get_list_sources_handler)]
GetSourceDependency = Annotated[GetSourceHandler, Depends(get_source_handler)]
ReviewSourceDependency = Annotated[ReviewSourceHandler, Depends(get_review_source_handler)]


@dataclass(frozen=True)
class SourceFile:
    filename: str | None
    content: bytes


async def read_source_file(
    request: Request,
    file: Annotated[UploadFile, File(description="Archivo original CSV o XLSX.")],
    handler: UploadSourceDependency,
) -> AsyncIterator[SourceFile]:
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
        yield SourceFile(filename=file.filename, content=content)
    finally:
        for _, part in form.multi_items():
            if isinstance(part, StarletteUploadFile):
                await part.close()


SourceFileDependency = Annotated[SourceFile, Depends(read_source_file)]
