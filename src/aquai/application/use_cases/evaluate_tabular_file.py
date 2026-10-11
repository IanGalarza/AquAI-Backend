import asyncio
import json
from hashlib import sha256
from collections import Counter
from time import perf_counter

from aquai.application.dtos.tabular import (
    EvaluationMetrics,
    TabularEvaluation,
    TabularFinding,
    ValidationResult,
)
from aquai.application.errors import FileTooLargeError, LLMProviderError, TabularInputError
from aquai.application.interfaces.tabular import LLMProvider, TabularProcessor
from aquai.application.services.tabular_recipe import (
    RecipeParseError,
    parse_preparation_recipe_with_metrics,
)
from aquai.application.services.tabular_validation import TabularValidationService
from aquai.application.services.tabular_rules import TabularRuleService
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
        rule_service: TabularRuleService | None = None,
    ) -> None:
        self._processor = processor
        self._llm_provider = llm_provider
        self._default_model = default_model
        self._max_upload_bytes = max_upload_bytes
        self._validator = validator or TabularValidationService()
        self._rule_service = rule_service

    async def execute(
        self,
        file_name: str | None,
        content: bytes,
        *,
        model: str | None = None,
        known_rules: str | None = None,
        rule_profile: str | None = None,
    ) -> TabularEvaluation:
        started = perf_counter()
        file_name = self._validate_file(file_name, content)
        selected_model = self._default_model if model is None else model.strip()
        if not selected_model or len(selected_model) > 200:
            raise TabularInputError("El modelo debe tener entre 1 y 200 caracteres.")
        if known_rules is not None and len(known_rules) > 4000:
            raise TabularInputError("Las reglas conocidas no pueden superar 4000 caracteres.")
        if self._rule_service is not None:
            self._rule_service.validate_profile(rule_profile)
        elif rule_profile is not None:
            raise TabularInputError("No se configuró un catálogo para seleccionar perfiles de reglas.")

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

        rule_composition = None
        prompt_rules = known_rules
        if self._rule_service is not None:
            rule_composition = self._rule_service.compose(
                summary, rule_profile, known_rules, source_sha256=sha256(content).hexdigest(),
            )
            result.applied_rules = rule_composition.metadata
            prompt_rules = json.dumps(
                rule_composition.prompt_rules, ensure_ascii=False, separators=(",", ":")
            )

        stage_started = perf_counter()
        try:
            result.llm_response = await self._llm_provider.generate_recipe(
                summary, selected_model, prompt_rules
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
            if self._rule_service is not None and rule_composition is not None:
                recipe_findings = self._rule_service.validate_recipe(summary, result.recipe, rule_composition)
                rule_composition.findings.extend(recipe_findings)
                rule_composition.review_reasons.extend(
                    finding["message"] for finding in recipe_findings if finding["requires_review"]
                )
            if result.recipe.status != "failed":
                stage_started = perf_counter()
                execution_options = {}
                if rule_composition is not None and rule_composition.preparation_policies is not None:
                    execution_options["policies"] = rule_composition.preparation_policies
                result.execution = await asyncio.to_thread(
                    self._processor.execute, file_name, content, result.recipe, summary, **execution_options,
                )
                metrics.execution_duration_ms = self._elapsed_ms(stage_started)
            stage_started = perf_counter()
            validation_options = {}
            if rule_composition is not None:
                if rule_composition.identity_clarifications:
                    validation_options["identity_clarifications"] = rule_composition.identity_clarifications
                if rule_composition.preparation_policies is not None:
                    validation_options["preparation_policies"] = rule_composition.preparation_policies
            result.validation = self._validator.validate(
                summary, result.recipe, result.execution, **validation_options,
            )
            metrics.validation_duration_ms = self._elapsed_ms(stage_started)
            result.response_format.recipe_executable = self._is_executable(result)

        if rule_composition is not None:
            result.validation.findings.extend(
                TabularFinding.model_validate(finding) for finding in rule_composition.findings
            )
            result.validation.warnings = list(dict.fromkeys(
                result.validation.warnings + rule_composition.review_reasons
            ))
            if (
                rule_composition.review_reasons
                and result.validation.status != EvaluationStatus.FAILED
            ):
                result.validation.status = EvaluationStatus.REVIEW_REQUIRED

        result.validation.review_reasons = list({
            finding.model_dump_json(): finding for finding in result.validation.findings if finding.requires_review
        }.values())

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
        planned_blocks = Counter((sheet.sheet_name, sheet.block_id) for sheet in recipe.sheets)
        executed_blocks = Counter((sheet.sheet_name, sheet.block_id) for sheet in result.execution.sheets)
        treated_blocks = Counter((treatment.sheet_name, block_id) for treatment in result.execution.empty_sheets
                                 for block_id in treatment.skipped_block_ids)
        return (
            planned_blocks == executed_blocks + treated_blocks
            and all(
                sheet.rows_after > 0 and bool(sheet.columns_after) and not sheet.errors
                for sheet in result.execution.sheets
            )
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 3)
