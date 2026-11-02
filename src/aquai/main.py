import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aquai.infrastructure.providers.conversations import ConversationRuntime
from aquai.infrastructure.providers.llm.ollama import warm_up_models
from aquai.infrastructure.providers.llm.settings import get_llm_settings
from aquai.infrastructure.providers.sources import SourceRuntime, create_source_runtime
from aquai.presentation.controllers.conversations import router as conversations_router
from aquai.presentation.controllers.sources import router as sources_router
from aquai.presentation.errors import register_exception_handlers
from aquai.presentation.uploadLimit import SourceUploadSizeLimit


#
# Configura el registro de la aplicación para mostrar los mensajes y duraciones por etapa.
#
def _configure_logging() -> None:
    logger = logging.getLogger("aquai")
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


#
# Crea la aplicación FastAPI con su ciclo de vida, middleware, manejadores de errores y rutas.
#
def create_app(
    *, sources: SourceRuntime | None = None, conversations: ConversationRuntime | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = sources or create_source_runtime()
        app.state.sources = runtime
        await runtime.worker.start()

        warm_up = None if sources else asyncio.create_task(warm_up_models(get_llm_settings()))

        try:
            yield
        finally:
            if warm_up is not None:
                warm_up.cancel()
            await runtime.worker.stop()

    def max_upload_bytes() -> int | None:
        return app.state.sources.upload.max_upload_bytes if app.state.sources else None

    app = FastAPI(title="AquAI Backend", version="0.1.0", lifespan=lifespan)
    app.state.sources = sources
    app.state.conversations = conversations

    app.add_middleware(SourceUploadSizeLimit, max_upload_bytes=max_upload_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )
    register_exception_handlers(app)
    app.include_router(sources_router)
    app.include_router(conversations_router)

    return app


_configure_logging()
app = create_app()
