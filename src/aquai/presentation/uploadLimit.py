"""Acota el multipart antes de que el parser escriba archivos temporales."""

from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class SourceUploadSizeLimit:
    def __init__(self, app: ASGIApp, max_upload_bytes: Callable[[], int | None]) -> None:
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] != "http" or scope["method"] != "POST"
                or scope["path"].rstrip("/") != "/sources"):
            await self.app(scope, receive, send)
            return
        maximum = self.max_upload_bytes()
        if maximum is None:
            await self.app(scope, receive, send)
            return
        # El archivo tiene su propio límite; el sobre multipart admite 64 KiB.
        maximum += 65536
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > maximum:
                await JSONResponse(
                    status_code=413, content={"detail": "El archivo supera el tamaño máximo permitido."},
                )(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
