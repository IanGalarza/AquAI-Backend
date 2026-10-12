import asyncio
import json
from hashlib import sha256
from collections.abc import Awaitable, Callable
from copy import deepcopy

from aquai.application.contracts.tabular.results import (
    TabularEvaluation,
    TabularFinding,
    ValidationResult,
)
from aquai.application.contracts.tabular.rules import RuleComposition
from aquai.application.errors import FileTooLargeError, LLMProviderError, TabularInputError
from aquai.application.interfaces.tabular import LLMProvider, TabularProcessor
from aquai.application.services.tabularRecipe import (
    RecipeParseError,
    parse_preparation_recipe,
)
from aquai.application.services.tabularValidation import TabularValidationService
from aquai.application.services.tabularRules import TabularRuleService
from aquai.domain.enums.tabular import EvaluationStatus


class TabularEvaluationService:
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

        return await self._run(
            file_name, content, selected_model, known_rules=known_rules, rule_profile=rule_profile,
        )

    async def process_source(
        self,
        file_name: str,
        content: bytes,
        *,
        checkpoint: Callable[[str, TabularEvaluation | None, dict | None], Awaitable[None]],
        saved_evaluation: TabularEvaluation | None = None,
        saved_rules: dict | None = None,
        resume_stage: str = "execution",
    ) -> TabularEvaluation:
        """Run the same pipeline using a persisted original and durable phase boundaries."""
        if resume_stage not in {"execution", "validation"}:
            raise TabularInputError("Etapa de reanudación desconocida.")
        return await self._run(
            self._validate_file(file_name, content), content, self._default_model,
            checkpoint=checkpoint, saved_evaluation=saved_evaluation, saved_rules=saved_rules,
            resume_stage=resume_stage, automatic_profile=True,
        )

    def validate_upload(self, file_name: str | None, content: bytes) -> str:
        return self._validate_file(file_name, content)

    async def _run(
        self,
        file_name: str,
        content: bytes,
        selected_model: str,
        *,
        known_rules: str | None = None,
        rule_profile: str | None = None,
        checkpoint: Callable[[str, TabularEvaluation | None, dict | None], Awaitable[None]] | None = None,
        saved_evaluation: TabularEvaluation | None = None,
        saved_rules: dict | None = None,
        resume_stage: str = "execution",
        automatic_profile: bool = False,
    ) -> TabularEvaluation:

        async def persist(stage, result=None, composition=None):
            if checkpoint is not None:
                await checkpoint(stage, result, composition.snapshot() if composition else None)

        original_proposal = saved_evaluation.validation.original_proposal if saved_evaluation else None
        if saved_evaluation is None:
            await persist("inspecting")
            summary = await self._run_processor(self._processor.inspect, file_name, content)
            result = TabularEvaluation(
                model=selected_model,
                structural_summary=summary,
                validation=ValidationResult(status=EvaluationStatus.FAILED),
            )
            await persist("inspected", result)
        else:
            result = saved_evaluation.model_copy(deep=True)
            summary = result.structural_summary
            result.validation = ValidationResult(status=EvaluationStatus.FAILED)

        rule_composition = RuleComposition.restore(saved_rules) if saved_rules is not None else None
        prompt_rules = known_rules
        if rule_composition is None and self._rule_service is not None:
            digest = sha256(content).hexdigest()
            if automatic_profile:
                rule_profile = self._rule_service.select_compatible_profile(summary, digest)
            rule_composition = self._rule_service.compose(
                summary, rule_profile, known_rules, source_sha256=digest,
            )
        if rule_composition is not None:
            result.applied_rules = deepcopy(rule_composition.metadata)
            prompt_rules = json.dumps(
                rule_composition.prompt_rules, ensure_ascii=False, separators=(",", ":")
            )

        if result.recipe is None:
            await persist("generating_recipe", result, rule_composition)
            try:
                result.llm_response = await self._llm_provider.generate_recipe(
                    summary, selected_model, prompt_rules
                )
            except LLMProviderError as exc:
                result.validation.errors.append(str(exc))
            if result.llm_response is not None:
                result.model = result.llm_response.model
                try:
                    result.recipe = parse_preparation_recipe(result.llm_response.content)
                except RecipeParseError as exc:
                    result.validation.errors.append(str(exc))
            await persist("recipe_generated", result, rule_composition)

        recipe_findings = []
        if result.recipe is not None:
            if self._rule_service is not None and rule_composition is not None:
                recipe_findings = self._rule_service.validate_recipe(summary, result.recipe, rule_composition)
            if result.recipe.status != "failed" and (saved_evaluation is None or resume_stage == "execution"):
                await persist("executing", result, rule_composition)
                execution_options = {}
                if rule_composition is not None and rule_composition.preparation_policies is not None:
                    execution_options["policies"] = rule_composition.preparation_policies
                result.execution = await self._run_processor(
                    self._processor.execute, file_name, content, result.recipe, summary, **execution_options,
                )
                await persist("executed", result, rule_composition)
            await persist("validating", result, rule_composition)
            validation_options = {}
            if rule_composition is not None:
                if rule_composition.identity_clarifications:
                    validation_options["identity_clarifications"] = rule_composition.identity_clarifications
                if rule_composition.preparation_policies is not None:
                    validation_options["preparation_policies"] = rule_composition.preparation_policies
            result.validation = self._validator.validate(
                summary, result.recipe, result.execution, **validation_options,
            )
            if original_proposal is not None:
                result.validation.original_proposal = original_proposal

        if rule_composition is not None:
            all_rule_findings = rule_composition.findings + recipe_findings
            rule_review_reasons = rule_composition.review_reasons + [
                finding["message"] for finding in recipe_findings if finding["requires_review"]
            ]
            result.applied_rules["findings"] = deepcopy(all_rule_findings)
            result.validation.findings.extend(
                TabularFinding.model_validate(finding) for finding in all_rule_findings
            )
            result.validation.warnings = list(dict.fromkeys(
                result.validation.warnings + rule_review_reasons
            ))
            if (
                rule_review_reasons
                and result.validation.status != EvaluationStatus.FAILED
            ):
                result.validation.status = EvaluationStatus.REVIEW_REQUIRED

        result.validation.review_reasons = list({
            finding.model_dump_json(): finding for finding in result.validation.findings if finding.requires_review
        }.values())

        await persist("completed", result, rule_composition)
        return result

    @staticmethod
    async def _run_processor(function, *args, **kwargs):
        """Keep the worker occupied until a cancelled CPU phase has actually stopped."""
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # Consume a concurrent processor error while retaining the shutdown cancellation.
            if not task.cancelled():
                try:
                    task.result()
                except Exception:
                    pass
            raise

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
