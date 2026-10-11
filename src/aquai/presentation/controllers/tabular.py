from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from aquai.application.errors import FileTooLargeError, TabularInputError
from aquai.application.use_cases.evaluate_tabular_file import EvaluateTabularFile
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.tabular import get_tabular_evaluator, get_tabular_settings
from aquai.presentation.schemas.tabular import TabularEvaluationResponse


router = APIRouter(prefix="/experimental/tabular", tags=["Procesamiento tabular experimental"])


@router.post("/evaluate", response_model=TabularEvaluationResponse)
async def evaluate_tabular_file(
    file: Annotated[UploadFile, File(description="Archivo original CSV o XLSX.")],
    evaluator: Annotated[EvaluateTabularFile, Depends(get_tabular_evaluator)],
    settings: Annotated[TabularSettings, Depends(get_tabular_settings)],
    model: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    known_rules: Annotated[str | None, Form(max_length=4000)] = None,
    rule_profile: Annotated[str | None, Form(max_length=80, description=(
        "Perfil opcional: biological_counts, self_sampling, belen_cyanobacteria "
        "o cyanobacterial_alerts. Las reglas comunes siempre se aplican."
    ))] = None,
) -> TabularEvaluationResponse:
    try:
        content = await file.read(settings.max_upload_bytes + 1)
        if len(content) > settings.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño máximo permitido.")
        result = await evaluator.execute(
            file.filename, content, model=model, known_rules=known_rules, rule_profile=rule_profile
        )
        return TabularEvaluationResponse.model_validate(result.model_dump())
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except TabularInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()
