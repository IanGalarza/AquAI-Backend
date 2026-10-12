from contextlib import asynccontextmanager

from fastapi import FastAPI

from aquai.infrastructure.providers.sources import SourceRuntime, create_source_runtime
from aquai.presentation.controllers.sources import router as sources_router
from aquai.presentation.errors import register_exception_handlers


def create_app(*, sources: SourceRuntime | None = None) -> FastAPI:
    sources = sources or create_source_runtime()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await sources.worker.start()
        try:
            yield
        finally:
            await sources.worker.stop()

    app = FastAPI(title="AquAI Backend", version="0.1.0", lifespan=lifespan)
    app.state.sources = sources
    register_exception_handlers(app)
    app.include_router(sources_router)
    return app


app = create_app()
