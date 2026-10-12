import re

from pydantic import JsonValue

from aquai.application.utils.tabularText import normalized_name
from aquai.application.validators.tabular.ruleValidation import rule_finding


def detect_additional_conflicts(
    additional: str | None, rules: list[dict[str, JsonValue]]
) -> list[dict[str, JsonValue]]:
    if not additional or not additional.strip():
        return []
    normalized = normalized_name(additional)
    change = r"(?:convert\w*|conviert\w*|reemplaz\w*|cambi\w*|transform\w*|replace|set|turn)"
    patterns = [
        (
            "preserve_original_values",
            change + r"[^;\n]{0,80}(?:\bnd\b|\bn[.]d[.]?\b|guion\w*|categor\w*|<\s*\d)"
            + r"[^;\n]{0,60}(?:\ba\b|\ben\b|\bpor\b|\bto\b|\bwith\b)\s*(?:0\b|cero\b|zero\b|null\b|nulo\b)",
        ),
        (
            "preserve_original_values",
            r"(?:elimin\w*|borr\w*|quit\w*|remove|drop|discard)[^;\n]{0,45}\b(?:ceros|zeros)\b",
        ),
        (
            "preserve_identifiers_and_periods",
            r"(?:corrig\w*|correg\w*|renombr\w*|correct|rename)[^;\n]{0,45}\b(?:taxon\w*|especies|species)\b",
        ),
        (
            "preserve_identifiers_and_periods",
            r"(?:deduplic\w*|deduplicate)[^;\n]{0,70}\b(?:parecid\w*|simil\w*|semejan\w*)\b",
        ),
        (
            "preserve_formula_provenance",
            r"(?:recalcul\w*|recalculate)[^;\n]{0,45}\b(?:formula\w*)\b",
        ),
        (
            "separate_structural_roles",
            r"(?:elimin\w*|borr\w*|descart\w*|remove|drop|discard)[^;\n]{0,45}"
            r"\b(?:totales|leyendas|derivados|hoja3|totals|legends)\b",
        ),
        (
            "belen_abundance_modalities",
            change + r"[^;\n]{0,65}(?:relativ\w*|rar\w*|categor\w*|colonias/ml|individuos/ml)"
            + r"[^;\n]{0,65}(?:celulas/ml|cells/ml|absolut\w*)",
        ),
        (
            "alert_values_and_labels",
            r"(?:asign\w*|clasific\w*|interpret\w*|replace|label)[^;\n]{0,45}alert_class"
            r"[^;\n]{0,45}\b(?:seguro|peligroso|safe|dangerous)\b",
        ),
    ]
    findings: list[dict[str, JsonValue]] = []
    conflicts: set[str] = set()
    available = {rule["id"] for rule in rules}
    for rule_id, pattern in patterns:
        if rule_id not in available or rule_id in conflicts:
            continue
        for match in re.finditer(pattern, normalized):
            prefix = normalized[max(0, match.start() - 40):match.start()]
            if re.search(r"\b(?:no|nunca|not|never|evitar|avoid|sin|without)(?:\s+\w+){0,2}\s*$", prefix):
                continue
            conflicts.add(rule_id)
            findings.append(rule_finding(
                "rule_additional_conflict",
                f"Las reglas adicionales contradicen {rule_id}. Se conservaron ambas "
                "instrucciones para auditoría y la evaluación requiere revisión.",
            ))
            break
    return findings
