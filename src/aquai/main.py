from contextlib import asynccontextmanager

from fastapi import FastAPI

from aquai.infrastructure.providers.sources import SourceRuntime, create_source_runtime
from aquai.presentation.controllers.sources import router as sources_router
from aquai.presentation.errors import register_exception_handlers
from aquai.presentation.uploadLimit import SourceUploadSizeLimit


def create_app(*, sources: SourceRuntime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = sources or create_source_runtime()
        app.state.sources = runtime
        await runtime.worker.start()
        try:
            yield
        finally:
            await runtime.worker.stop()

    app = FastAPI(title="AquAI Backend", version="0.1.0", lifespan=lifespan)
    app.state.sources = sources
    app.add_middleware(
        SourceUploadSizeLimit,
        max_upload_bytes=lambda: (
            app.state.sources.upload.max_upload_bytes if app.state.sources else None
        ),
    )
    register_exception_handlers(app)
    app.include_router(sources_router)
    return app


app = create_app()
