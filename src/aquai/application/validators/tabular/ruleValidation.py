from pydantic import JsonValue

from aquai.application.contracts.tabular.recipe import (
    ConvertType,
    PreparationRecipe,
    RenameColumns,
    SheetRecipe,
)
from aquai.application.contracts.tabular.rules import RuleComposition
from aquai.application.contracts.tabular.structure import SheetSummary, StructuralSummary
from aquai.application.utils.tabularCoordinates import cell_coordinate
from aquai.application.utils.tabularText import normalized_name


def validate_recipe(
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
        findings.extend(_validate_configured_sheet_role(source, block, composition))
        findings.extend(_validate_preparation_policy_block(source, block, composition))
        match profile:
            case "cyanobacterial_alerts":
                findings.extend(_validate_alert_block(source, block))
            case "biological_counts":
                findings.extend(_validate_presence_summary(source, block, composition))
    return findings


def _validate_presence_summary(
    source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
) -> list[dict[str, JsonValue]]:
    evidence = composition.metadata["evidence"]
    if not evidence["file_name_matches"] or evidence["missing_expected_sheets"]:
        return []
    if normalized_name(source.name) != "hoja1" or block.role != "observations":
        return []
    if not any("presencia" in normalized_name(name) for name in source.initial_columns):
        return []
    return [_block_finding(
        "rule_profile_role_conflict",
        "Hoja1 contiene un resumen de presencias verificado por su encabezado; "
        "clasificarlo como observaciones puede mezclar presencia y abundancia.",
        source, block,
    )]


def _validate_preparation_policy_block(
    source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
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
                    findings.append(_block_finding(
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
            findings.append(_block_finding(
                "rule_policy_identifier_separation",
                f"La regla {policy.rule_id} conserva los identificadores calculados junto a sus registros; "
                "el bloque observacional deja fuera columnas de identificación documentadas.",
                source, block,
            ))
    return findings


def _validate_configured_sheet_role(
    source: SheetSummary, block: SheetRecipe, composition: RuleComposition,
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
            if normalized_name(requirement["sheet_name"]) != normalized_name(source.name):
                continue
            allowed_roles = requirement["allowed_roles"]
            if block.role in allowed_roles:
                continue
            findings.append(_block_finding(
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
    source: SheetSummary, block: SheetRecipe
) -> list[dict[str, JsonValue]]:
    protected = {"sample_id": "sample_id", "alert_class": "alert_class"}
    for name in source.initial_columns:
        if normalized_name(name) in protected:
            protected[normalized_name(name)] = name
    findings: list[dict[str, JsonValue]] = []
    for transformation in block.transformations:
        match transformation:
            case RenameColumns():
                changed = [old for old, new in transformation.mapping.items() if old != new]
                if changed:
                    findings.append(_block_finding(
                        "rule_profile_columns_changed",
                        "El perfil cyanobacterial_alerts conserva los nombres de columnas "
                        f"originales; la receta propone renombrar: {', '.join(changed)}.",
                        source,
                        block,
                    ))
                renamed = {
                    normalized_name(new): protected[normalized_name(old)]
                    for old, new in transformation.mapping.items() if normalized_name(old) in protected
                }
                for old in transformation.mapping:
                    protected.pop(normalized_name(old), None)
                protected.update(renamed)
            case ConvertType():
                for name in transformation.columns:
                    original = protected.get(normalized_name(name))
                    if original is None:
                        continue
                    findings.append(_block_finding(
                        "rule_profile_categorical_conversion",
                        f"El perfil cyanobacterial_alerts conserva {original} como código o "
                        f"categoría original. Convertirlo a {transformation.target_type} requiere "
                        "documentación del significado y revisión, aunque la conversión ejecute.",
                        source,
                        block,
                    ))
    return findings


def _block_finding(
    code: str, message: str, source: SheetSummary, block: SheetRecipe
) -> dict[str, JsonValue]:
    finding = rule_finding(code, message, sheet_name=source.name)
    finding["block_id"] = block.block_id
    first = cell_coordinate(block.data_start_row, block.data_start_column)
    last = cell_coordinate(
        block.data_end_row, block.data_end_column
    )
    finding["range"] = f"{first}:{last}"
    return finding


def rule_finding(
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
