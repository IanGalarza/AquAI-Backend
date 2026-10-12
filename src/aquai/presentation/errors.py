"""Traducción de errores de Application al contrato HTTP de fuentes."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aquai.application.errors import (
    FileTooLargeError,
    SourceConflictError,
    SourceNotFoundError,
    SourcesUnavailableError,
    TabularInputError,
)


ERROR_RESPONSES = {
    400: {"description": "Carga o decisión inválida."},
    404: {"description": "Fuente o revisión inexistente."},
    409: {"description": "La revisión ya no corresponde al estado actual."},
    503: {"description": "La persistencia o el procesamiento de fuentes no está disponible."},
}

_ERROR_STATUS_CODES = {
    FileTooLargeError: 413,
    SourceNotFoundError: 404,
    SourceConflictError: 409,
    SourcesUnavailableError: 503,
    TabularInputError: 400,
}


def _error_handler(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handle


def register_exception_handlers(app: FastAPI) -> None:
    for error_type, status_code in _ERROR_STATUS_CODES.items():
        app.add_exception_handler(error_type, _error_handler(status_code))
