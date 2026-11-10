from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


#
# Rechaza los cambios pedidos desde páginas de otro origen: el navegador envía igual un formulario
# o una carga simple aunque CORS no le deje leer la respuesta.
#
class TrustedOrigins:
    #
    # Guarda la aplicación envuelta y los orígenes del frontend autorizados.
    #
    def __init__(self, app: ASGIApp, origins: list[str]) -> None:
        self._app = app
        self._origins = set(origins)

    #
    # Deja pasar las lecturas y los pedidos sin Origin; rechaza los cambios de un origen ajeno.
    #
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in _STATE_CHANGING:
            await self._app(scope, receive, send)
            return

        origin = dict(scope["headers"]).get(b"origin")
        if origin is None or origin.decode("latin-1") in self._origins:
            await self._app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=403,
            content={"detail": "El origen de la solicitud no está autorizado para modificar datos."},
        )
        await response(scope, receive, send)
