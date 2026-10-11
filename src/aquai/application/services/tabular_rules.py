from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePath
import re
import unicodedata

from pydantic import JsonValue

from aquai.application.dtos.tabular import (
    ConvertType,
    PreparationRecipe,
    RecordIdentityClarification,
    RenameColumns,
    SheetRecipe,
    SheetSummary,
    StructuralSummary,
)
from aquai.application.errors import TabularInputError
from aquai.application.interfaces.tabular_rules import TabularRuleCatalog


@dataclass(frozen=True)
class RuleComposition:
    metadata: dict[str, JsonValue]
    prompt_rules: dict[str, JsonValue]
    review_reasons: list[str]
    findings: list[dict[str, JsonValue]]
    identity_clarifications: tuple[RecordIdentityClarification, ...] = ()


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
        return RuleComposition(metadata, prompt_rules, review_reasons, findings, identity_clarifications)

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
            if profile == "cyanobacterial_alerts":
                findings.extend(self._validate_alert_block(source, block))
            elif profile == "biological_counts":
                evidence = composition.metadata["evidence"]
                verified = evidence["file_name_matches"] and not evidence["missing_expected_sheets"]
                presence_header = any(
                    "presencia" in _normalized(name) for name in source.initial_columns
                )
                if verified and _normalized(source.name) == "hoja1" and presence_header:
                    if block.role == "observations":
                        findings.append(self._block_finding(
                            "rule_profile_role_conflict",
                            "Hoja1 contiene un resumen de presencias verificado por su encabezado; "
                            "clasificarlo como observaciones puede mezclar presencia y abundancia.",
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
            if isinstance(transformation, RenameColumns):
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
            elif isinstance(transformation, ConvertType):
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
        return [{**deepcopy(rule), "version": version, "scope": scope} for rule in rules]

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
