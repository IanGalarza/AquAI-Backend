from typing import Protocol

from aquai.application.dtos.tabular import (
    ExecutionResult,
    LLMResponse,
    PreparationRecipe,
    StructuralSummary,
)
from aquai.application.dtos.tabular_policies import PreparationPolicies


class TabularProcessor(Protocol):
    def inspect(self, file_name: str, content: bytes) -> StructuralSummary: ...

    def execute(
        self,
        file_name: str,
        content: bytes,
        recipe: PreparationRecipe,
        summary: StructuralSummary,
        *,
        policies: PreparationPolicies | None = None,
    ) -> ExecutionResult: ...


class LLMProvider(Protocol):
    async def generate_recipe(
        self,
        summary: StructuralSummary,
        model: str,
        known_rules: str | None = None,
    ) -> LLMResponse: ...
