from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePath
import re
import unicodedata

from pydantic import JsonValue

from aquai.application.contracts.tabular.recipe import (
    ConvertType,
    PreparationRecipe,
    RenameColumns,
    SheetRecipe,
)
from aquai.application.contracts.tabular.structure import SheetSummary, StructuralSummary
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


def _normalized(value: str) -> str:
    return "".join(
        character for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).strip()


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
        findings.extend(self._additional_conflicts(additional, common + profile_rules))
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
        profile = composition.metadata["profile"]
        sources = {sheet.name: sheet for sheet in summary.sheets}
        findings: list[dict[str, JsonValue]] = []
        for block in recipe.sheets:
            source = sources.get(block.sheet_name)
            if source is None:
                continue
            findings.extend(self._validate_configured_sheet_role(source, block, composition))
            findings.extend(self._validate_preparation_policy_block(source, block, composition))
            match profile:
                case "cyanobacterial_alerts":
                    findings.extend(self._validate_alert_block(source, block))
                case "biological_counts":
                    findings.extend(self._validate_presence_summary(source, block, composition))
        return findings

    def _validate_presence_summary(
        self, source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
    ) -> list[dict[str, JsonValue]]:
        evidence = composition.metadata["evidence"]
        if not evidence["file_name_matches"] or evidence["missing_expected_sheets"]:
            return []
        if _normalized(source.name) != "hoja1" or block.role != "observations":
            return []
        if not any("presencia" in _normalized(name) for name in source.initial_columns):
            return []
        return [self._block_finding(
            "rule_profile_role_conflict",
            "Hoja1 contiene un resumen de presencias verificado por su encabezado; "
            "clasificarlo como observaciones puede mezclar presencia y abundancia.",
            source, block,
        )]

    def _validate_preparation_policy_block(
        self, source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
    ) -> list[dict[str, JsonValue]]:
        policies = composition.preparation_policies
        if policies is None:
            return []
        findings: list[dict[str, JsonValue]] = []
        protected_number_names = {
            policy.expected_headers[column] for policy in policies.number_columns
            if policy.sheet_name == source.name for column in policy.column_indices
        }
        for transformation in block.transformations:
            match transformation:
                case RenameColumns(mapping=mapping):
                    protected_number_names = {mapping.get(name, name) for name in protected_number_names}
                case ConvertType(target_type="datetime", columns=columns):
                    incompatible = protected_number_names.intersection(columns)
                    if incompatible:
                        findings.append(self._block_finding(
                            "rule_policy_type_conflict",
                            "La política numérica no permite interpretar como fechas las columnas de abundancia: "
                            + ", ".join(sorted(incompatible)) + ". Conservar sus valores y revisar la receta.",
                            source, block,
                        ))
        for policy in policies.observation_rows:
            if policy.sheet_name != source.name or block.role != "observations":
                continue
            represented = set(range(block.data_start_column, block.data_end_column + 1))
            identity = set(policy.auxiliary_column_indices)
            if represented - identity and not identity.issubset(represented):
                findings.append(self._block_finding(
                    "rule_policy_identifier_separation",
                    f"La regla {policy.rule_id} conserva los identificadores calculados junto a sus registros; "
                    "el bloque observacional deja fuera columnas de identificación documentadas.",
                    source, block,
                ))
        return findings

    def _validate_configured_sheet_role(
        self, source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
    ) -> list[dict[str, JsonValue]]:
        """Contrasta roles documentados del perfil corroborado sin interpretar nombres en código."""
        evidence = composition.metadata["evidence"]
        if not evidence.get("file_name_matches") or evidence.get("missing_expected_sheets"):
            return []
        content = source.content_evidence
        if (
            content is not None and content.sheet_name == source.name and content.is_empty
            and content.verification == "full_sheet_values_and_formulas"
            and content.inspected_cells == source.rows * source.columns
            and source.formula_count == 0 and source.missing_formula_cache_count == 0
            and source.error_cell_count == 0
        ):
            return []
        findings: list[dict[str, JsonValue]] = []
        for rule in composition.metadata["profile_rules"]:
            for requirement in rule.get("sheet_role_requirements", []):
                if _normalized(requirement["sheet_name"]) != _normalized(source.name):
                    continue
                allowed_roles = requirement["allowed_roles"]
                if block.role not in allowed_roles:
                    findings.append(self._block_finding(
                        "rule_profile_role_conflict",
                        f"La regla {rule['id']} del perfil {composition.metadata['profile']} "
                        f"documenta los roles {', '.join(allowed_roles)} para la hoja '{source.name}'; "
                        f"el bloque declara '{block.role}'. Conservar sus celdas y procedencia, "
                        "y revisar la función del bloque sin convertirlo automáticamente en observaciones.",
                        source,
                        block,
                    ))
        return findings

    def _validate_alert_block(
        self, source: SheetSummary, block: SheetRecipe
    ) -> list[dict[str, JsonValue]]:
        protected = {"sample_id": "sample_id", "alert_class": "alert_class"}
        for name in source.initial_columns:
            if _normalized(name) in protected:
                protected[_normalized(name)] = name
        findings: list[dict[str, JsonValue]] = []
        for transformation in block.transformations:
            match transformation:
                case RenameColumns():
                    changed = [old for old, new in transformation.mapping.items() if old != new]
                    if changed:
                        findings.append(self._block_finding(
                            "rule_profile_columns_changed",
                            "El perfil cyanobacterial_alerts conserva los nombres de columnas "
                            f"originales; la receta propone renombrar: {', '.join(changed)}.",
                            source,
                            block,
                        ))
                    renamed = {
                        _normalized(new): protected[_normalized(old)]
                        for old, new in transformation.mapping.items() if _normalized(old) in protected
                    }
                    for old in transformation.mapping:
                        protected.pop(_normalized(old), None)
                    protected.update(renamed)
                case ConvertType():
                    for name in transformation.columns:
                        original = protected.get(_normalized(name))
                        if original is None:
                            continue
                        findings.append(self._block_finding(
                            "rule_profile_categorical_conversion",
                            f"El perfil cyanobacterial_alerts conserva {original} como código o "
                            f"categoría original. Convertirlo a {transformation.target_type} requiere "
                            "documentación del significado y revisión, aunque la conversión ejecute.",
                            source,
                            block,
                        ))
        return findings

    @classmethod
    def _block_finding(
        cls, code: str, message: str, source: SheetSummary, block: SheetRecipe
    ) -> dict[str, JsonValue]:
        finding = cls._finding(code, message, sheet_name=source.name)
        finding["block_id"] = block.block_id
        first = cls._coordinate(block.data_start_row, block.data_start_column)
        last = cls._coordinate(
            block.data_end_row, block.data_end_column
        )
        finding["range"] = f"{first}:{last}"
        return finding

    @staticmethod
    def _coordinate(row: int, column: int) -> str:
        letters = ""
        while column:
            column, remainder = divmod(column - 1, 26)
            letters = chr(65 + remainder) + letters
        return f"{letters}{row}"

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
        matches = _normalized(file_name) in {
            _normalized(name) for name in definition["file_names"]
        }
        sheet_names = {_normalized(sheet.name): sheet for sheet in summary.sheets}
        missing = [
            name for name in definition["required_sheets"] if _normalized(name) not in sheet_names
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
            findings.append(self._finding("rule_profile_unverified", message))
        if selected == "biological_counts" and matches and not missing:
            sheet = sheet_names[_normalized("Hoja3")]
            evidence["hoja3_merged_ranges"] = list(sheet.merged_cells)
            evidence["hoja3_merged_observation"] = "observed" if sheet.merged_cells else "not_observed"
            if not sheet.merged_cells:
                findings.append(self._finding(
                    "rule_observation_corrected",
                    "La inspección actual no encontró celdas combinadas en Hoja3; "
                    "se descartó esa observación previa y no se aplicó relleno por combinaciones.",
                    requires_review=False,
                    sheet_name=sheet.name,
                ))
        return evidence, findings

    def _additional_conflicts(
        self, additional: str | None, rules: list[dict[str, JsonValue]]
    ) -> list[dict[str, JsonValue]]:
        if not additional or not additional.strip():
            return []
        normalized = _normalized(additional)
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
                findings.append(self._finding(
                    "rule_additional_conflict",
                    f"Las reglas adicionales contradicen {rule_id}. Se conservaron ambas "
                    "instrucciones para auditoría y la evaluación requiere revisión.",
                ))
                break
        return findings

    @staticmethod
    def _finding(
        code: str,
        message: str,
        *,
        requires_review: bool = True,
        sheet_name: str | None = None,
    ) -> dict[str, JsonValue]:
        return {
            "code": code,
            "message": message,
            "sheet_name": sheet_name,
            "requires_review": requires_review,
        }
