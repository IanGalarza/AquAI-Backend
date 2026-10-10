from fastapi import FastAPI

from aquai.presentation.controllers.tabular import router as tabular_router


def create_app() -> FastAPI:
    app = FastAPI(title="AquAI Backend", version="0.1.0")
    app.include_router(tabular_router)
    return app


app = create_app()
