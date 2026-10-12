from copy import deepcopy
from dataclasses import dataclass

from pydantic import JsonValue

from aquai.application.contracts.tabular.policies import (
    PreparationPolicies,
    RecordIdentityClarification,
)


@dataclass(frozen=True)
class RuleComposition:
    metadata: dict[str, JsonValue]
    prompt_rules: dict[str, JsonValue]
    review_reasons: list[str]
    findings: list[dict[str, JsonValue]]
    identity_clarifications: tuple[RecordIdentityClarification, ...] = ()
    preparation_policies: PreparationPolicies | None = None
    source_decisions: tuple[dict, ...] = ()

    def snapshot(self) -> dict:
        """Persist the effective scope, including policies outside the LLM recipe."""
        return {
            "metadata": deepcopy(self.metadata),
            "prompt_rules": deepcopy(self.prompt_rules),
            "review_reasons": list(self.review_reasons),
            "findings": deepcopy(self.findings),
            "identity_clarifications": [item.model_dump(mode="json") for item in self.identity_clarifications],
            "preparation_policies": (
                self.preparation_policies.model_dump(mode="json") if self.preparation_policies else None
            ),
            "source_decisions": deepcopy(list(self.source_decisions)),
        }

    @classmethod
    def restore(cls, snapshot: dict) -> "RuleComposition":
        policies = snapshot.get("preparation_policies")
        return cls(
            metadata=deepcopy(snapshot.get("metadata", {})),
            prompt_rules=deepcopy(snapshot.get("prompt_rules", {})),
            review_reasons=list(snapshot.get("review_reasons", [])),
            findings=deepcopy(snapshot.get("findings", [])),
            identity_clarifications=tuple(
                RecordIdentityClarification.model_validate(item)
                for item in snapshot.get("identity_clarifications", [])
            ),
            preparation_policies=PreparationPolicies.model_validate(policies) if policies else None,
            source_decisions=tuple(deepcopy(snapshot.get("source_decisions", []))),
        )
