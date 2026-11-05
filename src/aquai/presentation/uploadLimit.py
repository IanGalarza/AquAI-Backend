import re
from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MULTIPART_ENVELOPE_BYTES = 65536
SOURCE_FILE_PATH = re.compile(r"/sources/[^/]+/file")


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
# Indica si la solicitud carga un archivo (POST /sources o PUT /sources/{id}/file).
#
def _is_source_upload(scope: Scope) -> bool:
    if scope["type"] != "http":
        return False

    path = scope["path"].rstrip("/")
    is_new_source = scope["method"] == "POST" and path == "/sources"
    is_replacement = scope["method"] == "PUT" and SOURCE_FILE_PATH.fullmatch(path) is not None

    return is_new_source or is_replacement


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
