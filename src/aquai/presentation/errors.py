from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from aquai.application.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
    ConversationsUnavailableError,
    FileTooLargeError,
    SourceConflictError,
    SourceNotFoundError,
    SourcesUnavailableError,
    TabularInputError,
)

ERROR_RESPONSES = {
    400: {"description": "Archivo o solicitud inválidos."},
    404: {"description": "Fuente inexistente."},
    409: {"description": "Estado incompatible con la operación."},
    413: {"description": "El archivo supera el límite permitido."},
    503: {"description": "La persistencia o el procesamiento de fuentes no está disponible."},
}

_ERROR_STATUS_CODES = {
    FileTooLargeError: 413,
    SourceNotFoundError: 404,
    SourceConflictError: 409,
    SourcesUnavailableError: 503,
    TabularInputError: 400,
    ConversationNotFoundError: 404,
    ConversationConflictError: 409,
    ConversationsUnavailableError: 503,
}


def _error_handler(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handle


def register_exception_handlers(app: FastAPI) -> None:
    for error_type, status_code in _ERROR_STATUS_CODES.items():
        app.add_exception_handler(error_type, _error_handler(status_code))

    async def database_unavailable(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": "La persistencia no está disponible."}
        )

    app.add_exception_handler(SQLAlchemyError, database_unavailable)
