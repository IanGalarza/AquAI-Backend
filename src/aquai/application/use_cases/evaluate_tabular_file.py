import asyncio
from time import perf_counter

from aquai.application.dtos.tabular import (
    EvaluationMetrics,
    TabularEvaluation,
    ValidationResult,
)
from aquai.application.errors import FileTooLargeError, LLMProviderError, TabularInputError
from aquai.application.interfaces.tabular import LLMProvider, TabularProcessor
from aquai.application.services.tabular_recipe import (
    RecipeParseError,
    parse_preparation_recipe_with_metrics,
)
from aquai.application.services.tabular_validation import TabularValidationService
from aquai.domain.enums.tabular import EvaluationStatus


class EvaluateTabularFile:
    def __init__(
        self,
        processor: TabularProcessor,
        llm_provider: LLMProvider,
        *,
        default_model: str,
        max_upload_bytes: int,
        validator: TabularValidationService | None = None,
    ) -> None:
        self._processor = processor
        self._llm_provider = llm_provider
        self._default_model = default_model
        self._max_upload_bytes = max_upload_bytes
        self._validator = validator or TabularValidationService()

    async def execute(
        self,
        file_name: str | None,
        content: bytes,
        *,
        model: str | None = None,
        known_rules: str | None = None,
    ) -> TabularEvaluation:
        started = perf_counter()
        file_name = self._validate_file(file_name, content)
        selected_model = self._default_model if model is None else model.strip()
        if not selected_model or len(selected_model) > 200:
            raise TabularInputError("El modelo debe tener entre 1 y 200 caracteres.")
        if known_rules is not None and len(known_rules) > 4000:
            raise TabularInputError("Las reglas conocidas no pueden superar 4000 caracteres.")

        metrics = EvaluationMetrics()
        stage_started = perf_counter()
        summary = await asyncio.to_thread(self._processor.inspect, file_name, content)
        metrics.inspection_duration_ms = self._elapsed_ms(stage_started)
        result = TabularEvaluation(
            model=selected_model,
            structural_summary=summary,
            validation=ValidationResult(status=EvaluationStatus.FAILED),
            metrics=metrics,
        )

        stage_started = perf_counter()
        try:
            result.llm_response = await self._llm_provider.generate_recipe(
                summary, selected_model, known_rules
            )
        except LLMProviderError as exc:
            result.validation.errors.append(str(exc))
        finally:
            metrics.llm_duration_ms = self._elapsed_ms(stage_started)

        if result.llm_response is not None:
            result.model = result.llm_response.model
            try:
                parsed = parse_preparation_recipe_with_metrics(result.llm_response.content)
                result.recipe = parsed.recipe
                result.response_format = parsed.response_format
            except RecipeParseError as exc:
                result.response_format = exc.response_format
                result.validation.errors.append(str(exc))

        if result.recipe is not None:
            if result.recipe.status != "failed":
                stage_started = perf_counter()
                result.execution = await asyncio.to_thread(
                    self._processor.execute, file_name, content, result.recipe, summary
                )
                metrics.execution_duration_ms = self._elapsed_ms(stage_started)
            stage_started = perf_counter()
            result.validation = self._validator.validate(summary, result.recipe, result.execution)
            metrics.validation_duration_ms = self._elapsed_ms(stage_started)
            result.response_format.recipe_executable = self._is_executable(result)

        metrics.total_duration_ms = self._elapsed_ms(started)
        return result

    def _validate_file(self, file_name: str | None, content: bytes) -> str:
        if not file_name or not file_name.strip():
            raise TabularInputError("Debe adjuntar un archivo con nombre.")
        normalized_name = file_name.replace("\\", "/").rsplit("/", 1)[-1]
        if len(normalized_name) > 255:
            raise TabularInputError("El nombre del archivo no puede superar 255 caracteres.")
        if normalized_name.rsplit(".", 1)[-1].lower() not in {"csv", "xlsx"}:
            raise TabularInputError("Sólo se admiten archivos CSV y XLSX.")
        if not content:
            raise TabularInputError("El archivo está vacío.")
        if len(content) > self._max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño máximo permitido.")
        return normalized_name

    @staticmethod
    def _is_executable(result: TabularEvaluation) -> bool:
        recipe = result.recipe
        if recipe is None or recipe.status == "failed" or not recipe.sheets or result.validation.errors:
            return False
        planned_names = {sheet.sheet_name for sheet in recipe.sheets}
        executed_names = {sheet.sheet_name for sheet in result.execution.sheets}
        return (
            planned_names == executed_names
            and len(recipe.sheets) == len(result.execution.sheets)
            and all(
                sheet.rows_after > 0 and bool(sheet.columns_after) and not sheet.errors
                for sheet in result.execution.sheets
            )
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 3)
