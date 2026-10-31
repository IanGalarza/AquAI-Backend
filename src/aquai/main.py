import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aquai.infrastructure.providers.llm.ollama import warm_up_models
from aquai.infrastructure.providers.sources import SourceRuntime, create_source_runtime
from aquai.infrastructure.providers.tabular import get_tabular_settings
from aquai.presentation.controllers.sources import router as sources_router
from aquai.presentation.controllers.conversations import router as conversations_router
from aquai.infrastructure.providers.conversations import ConversationRuntime
from aquai.presentation.errors import register_exception_handlers
from aquai.presentation.uploadLimit import SourceUploadSizeLimit


def _configure_logging() -> None:
    """uvicorn sólo configura sus loggers: sin esto no se ven las duraciones por etapa."""
    logger = logging.getLogger("aquai")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def create_app(*, sources: SourceRuntime | None = None, conversations: ConversationRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = sources or create_source_runtime()
        app.state.sources = runtime
        await runtime.worker.start()
        # Sólo con la composición real: la carga del modelo no bloquea el arranque.
        warm_up = None if sources else asyncio.create_task(warm_up_models(get_tabular_settings()))
        try:
            yield
        finally:
            if warm_up is not None:
                warm_up.cancel()
            await runtime.worker.stop()

    app = FastAPI(title="AquAI Backend", version="0.1.0", lifespan=lifespan)
    app.state.sources = sources
    app.state.conversations = conversations
    app.add_middleware(
        SourceUploadSizeLimit,
        max_upload_bytes=lambda: (
            app.state.sources.upload.max_upload_bytes if app.state.sources else None
        ),
    )
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
