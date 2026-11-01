from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MULTIPART_ENVELOPE_BYTES = 65536


class SourceUploadSizeLimit:
    #
    # Guarda la aplicación envuelta y la función que da el tamaño máximo de carga.
    #
    def __init__(self, app: ASGIApp, max_upload_bytes: Callable[[], int | None]) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    #
    # Rechaza con 413 las cargas de fuentes que superan el tamaño máximo antes de procesarlas.
    #
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_source_upload(scope):
            await self.app(scope, receive, send)
            return

        maximum = self.max_upload_bytes()
        if maximum is None:
            await self.app(scope, receive, send)
            return

        maximum += MULTIPART_ENVELOPE_BYTES
        chunks = []
        size = 0

        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return

            chunk = message.get("body", b"")
            size += len(chunk)
            if size > maximum:
                response = _too_large_response()
                await response(scope, receive, send)
                return

            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)

        await self.app(scope, _replaying(body, receive), send)


#
# Indica si la solicitud es una carga de fuente (POST /sources).
#
def _is_source_upload(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope["method"] == "POST"
        and scope["path"].rstrip("/") == "/sources"
    )


#
# Construye la respuesta 413 que indica que el archivo supera el tamaño máximo.
#
def _too_large_response() -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": "El archivo supera el tamaño máximo permitido."},
    )


#
# Devuelve una función de recepción que reentrega el cuerpo ya leído a la aplicación.
#
def _replaying(body: bytes, receive: Receive) -> Receive:
    delivered = False

    async def replay() -> dict:
        nonlocal delivered
        if delivered:
            return await receive()

        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return replay
