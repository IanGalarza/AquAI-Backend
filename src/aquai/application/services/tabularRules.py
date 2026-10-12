from copy import deepcopy
from pathlib import PurePath

from pydantic import JsonValue

from aquai.application.contracts.tabular.recipe import PreparationRecipe
from aquai.application.contracts.tabular.rules import RuleComposition
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.errors import TabularInputError
from aquai.application.contracts.tabular.policies import (
    AssociationPolicy,
    ConfirmedCellPolicy,
    DateColumnPolicy,
    NumberColumnPolicy,
    ObservationRowPolicy,
    PreparationPolicies,
    RecordIdentityClarification,
)
from aquai.application.interfaces.tabularRules import TabularRuleCatalog
from aquai.application.utils.tabularText import normalized_name
from aquai.application.validators.tabular.additionalRuleConflicts import detect_additional_conflicts
from aquai.application.validators.tabular.ruleValidation import rule_finding, validate_recipe


class TabularRuleService:
    def __init__(self, catalog: TabularRuleCatalog) -> None:
        self._catalog = catalog.load()

    def validate_profile(self, profile: str | None) -> str | None:
        if profile is None or not profile.strip():
            return None
        selected = profile.strip()
        if selected not in self._catalog["profiles"]:
            available = ", ".join(self._catalog["profiles"])
            raise TabularInputError(
                f"Perfil de reglas desconocido: {selected}. Perfiles disponibles: {available}."
            )
        return selected

    def select_compatible_profile(self, summary: StructuralSummary, source_sha256: str) -> str | None:
        """Normal uploads only inherit profiles corroborated for the exact catalogued original."""
        file_name = PurePath(summary.file_name.replace("\\", "/")).name
        candidates = {
            rule["profile"] for rule in self._catalog.get("file_specific_rules", [])
            if rule["file_name"].casefold() == file_name.casefold() and rule["sha256"] == source_sha256
        }
        verified = []
        for profile in candidates:
            evidence, _ = self._profile_evidence(summary, profile)
            if evidence.get("file_name_matches") and not evidence.get("missing_expected_sheets"):
                verified.append(profile)
        return verified[0] if len(verified) == 1 else None

    def compose(
        self,
        summary: StructuralSummary,
        profile: str | None = None,
        additional: str | None = None,
        *,
        source_sha256: str | None = None,
    ) -> RuleComposition:
        selected = self.validate_profile(profile)
        version = self._catalog["version"]
        common = self._versioned_rules(self._catalog["common"], version, "common")
        profiles = self._catalog["profiles"]
        profile_rules = (
            self._versioned_rules(profiles[selected]["rules"], version, selected)
            if selected else []
        )
        file_specific_rules = self._file_specific_rules(summary, selected, source_sha256)
        identity_clarifications = tuple(
            RecordIdentityClarification.model_validate(clarification)
            for rule in file_specific_rules for clarification in rule.get("identity_clarifications", [])
        )
        evidence, findings = self._profile_evidence(summary, selected)
        preparation_policies = self._preparation_policies(evidence, profile_rules, file_specific_rules)
        findings.extend(detect_additional_conflicts(additional, common + profile_rules))
        review_reasons = [finding["message"] for finding in findings if finding["requires_review"]]
        used_sources = {rule["origin"] for rule in common + profile_rules + file_specific_rules}
        origins = {
            key: deepcopy(value) for key, value in self._catalog["sources"].items()
            if key in used_sources
        }
        metadata: dict[str, JsonValue] = {
            "version": version,
            "profile": selected,
            "origins": origins,
            "common_rules": common,
            "profile_rules": profile_rules,
            "additional_rules": additional,
            "evidence": evidence,
            "findings": findings,
        }
        prompt_rules: dict[str, JsonValue] = {
            "version": version,
            "profile": selected,
            "common": [{"id": rule["id"], "text": rule["text"]} for rule in common],
            "selected_profile": [
                {"id": rule["id"], "text": rule["text"]} for rule in profile_rules
            ],
            "additional": additional,
            "file_evidence": evidence,
            "conflicts": review_reasons,
        }
        if file_specific_rules:
            metadata["file_specific_rules"] = file_specific_rules
            prompt_rules["file_specific_rules"] = deepcopy(file_specific_rules)
        return RuleComposition(
            metadata, prompt_rules, review_reasons, findings, identity_clarifications, preparation_policies,
        )

    def _preparation_policies(
        self, evidence: dict[str, JsonValue], profile_rules: list[dict[str, JsonValue]],
        file_specific_rules: list[dict[str, JsonValue]],
    ) -> PreparationPolicies | None:
        """Sólo el catálogo corroborado autoriza tratamientos del sistema, nunca texto adicional."""
        if not evidence.get("file_name_matches") or evidence.get("missing_expected_sheets"):
            return None
        kinds = {
            "date_columns": DateColumnPolicy, "number_columns": NumberColumnPolicy,
            "confirmed_cells": ConfirmedCellPolicy, "observation_rows": ObservationRowPolicy,
            "associations": AssociationPolicy,
        }
        collected: dict[str, list] = {name: [] for name in kinds}
        for rules, file_specific in ((profile_rules, False), (file_specific_rules, True)):
            for rule in rules:
                for name, definitions in rule.get("preparation_policies", {}).items():
                    if name not in kinds:
                        raise TabularInputError(f"Política de preparación desconocida en el catálogo: {name}.")
                    for definition in definitions:
                        values = {
                            **deepcopy(definition), "rule_id": rule["id"],
                            "version": rule["version"], "origin": rule["origin"],
                        }
                        if name == "confirmed_cells":
                            values["source_sha256"] = self._confirmed_source_hash(rule, file_specific)
                        collected[name].append(kinds[name].model_validate(values))
        policies = PreparationPolicies.model_validate(collected)
        return policies if policies.enabled else None

    def _confirmed_source_hash(self, rule: dict[str, JsonValue], file_specific: bool) -> str:
        origin = self._catalog["sources"].get(rule["origin"], {})
        if not file_specific or origin.get("verification") != "user_confirmation":
            raise TabularInputError(
                "Una corrección confirmada requiere una regla de archivo con hash y origen humano."
            )
        return rule["sha256"]

    def _file_specific_rules(
        self, summary: StructuralSummary, selected: str | None, source_sha256: str | None,
    ) -> list[dict[str, JsonValue]]:
        if selected is None or source_sha256 is None:
            return []
        file_name = PurePath(summary.file_name.replace("\\", "/")).name
        return [
            {**deepcopy(rule), "scope": "file_specific"}
            for rule in self._catalog.get("file_specific_rules", [])
            if rule["profile"] == selected and rule["file_name"].casefold() == file_name.casefold()
            and rule["sha256"] == source_sha256
        ]

    def validate_recipe(
        self,
        summary: StructuralSummary,
        recipe: PreparationRecipe,
        composition: RuleComposition,
    ) -> list[dict[str, JsonValue]]:
        return validate_recipe(summary, recipe, composition)

    @staticmethod
    def _versioned_rules(
        rules: list[dict[str, JsonValue]], version: str, scope: str
    ) -> list[dict[str, JsonValue]]:
        return [{**deepcopy(rule), "version": rule.get("version", version), "scope": scope} for rule in rules]

    def _profile_evidence(
        self, summary: StructuralSummary, selected: str | None
    ) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]]]:
        if selected is None:
            return {"profile_selected": False}, []
        definition = self._catalog["profiles"][selected]
        file_name = PurePath(summary.file_name.replace("\\", "/")).name
        matches = normalized_name(file_name) in {
            normalized_name(name) for name in definition["file_names"]
        }
        sheet_names = {normalized_name(sheet.name): sheet for sheet in summary.sheets}
        missing = [
            name for name in definition["required_sheets"] if normalized_name(name) not in sheet_names
        ]
        evidence: dict[str, JsonValue] = {
            "profile_selected": True,
            "file_name_matches": matches,
            "missing_expected_sheets": missing,
            "observed_sheet_names": [sheet.name for sheet in summary.sheets],
            "observations_are_conditional": True,
        }
        findings: list[dict[str, JsonValue]] = []
        if not matches or missing:
            message = (
                f"El perfil {selected} no quedó corroborado por el nombre y las hojas del archivo. "
                "Verificar su pertinencia antes de aplicar observaciones específicas."
            )
            findings.append(rule_finding("rule_profile_unverified", message))
        if selected == "biological_counts" and matches and not missing:
            sheet = sheet_names[normalized_name("Hoja3")]
            evidence["hoja3_merged_ranges"] = list(sheet.merged_cells)
            evidence["hoja3_merged_observation"] = "observed" if sheet.merged_cells else "not_observed"
            if not sheet.merged_cells:
                findings.append(rule_finding(
                    "rule_observation_corrected",
                    "La inspección actual no encontró celdas combinadas en Hoja3; "
                    "se descartó esa observación previa y no se aplicó relleno por combinaciones.",
                    requires_review=False,
                    sheet_name=sheet.name,
                ))
        return evidence, findings
