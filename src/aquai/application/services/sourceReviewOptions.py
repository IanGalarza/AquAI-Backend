"""Classify review findings and expose the permitted correction contracts."""

from typing import Annotated

from pydantic import Field, TypeAdapter

from aquai.application.contracts.tabular.recipe import SheetRecipe
from aquai.application.contracts.tabular.results import TabularEvaluation, TabularFinding
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.utils.tabularCoordinates import cell_position


_ACTION_BY_FINDING = {
    **dict.fromkeys({
        "block_out_of_bounds", "unresolved_block_role", "structural_role_conflict",
        "unresolved_structural_region", "observations_in_nonobservational_block",
        "rule_profile_role_conflict", "rule_policy_identifier_separation", "suspicious_header_rows",
    }, "replace_block"),
    **dict.fromkeys({
        "unit_label_changed", "rule_profile_columns_changed", "rule_profile_categorical_conversion",
        "rule_policy_type_conflict", "conversion_failures", "duplicate_column_names", "unnamed_columns",
        "nonempty_value_loss",
    }, "correct_transformations"),
    **dict.fromkeys({"uncovered_source_cells", "unhandled_source_sheet"}, "add_block"),
}
_TECHNICAL_FINDINGS = {
    "source_missing_formula_cache", "unresolved_formula_cache", "source_excel_error", "excel_error_cells",
    "execution_range_mismatch", "row_count_mismatch", "column_count_mismatch", "unexplained_row_loss",
    "source_header_evidence_mismatch", "auxiliary_row_evidence_mismatch",
    "empty_sheet_evidence_mismatch", "temporal_date_evidence_mismatch", "association_evidence_mismatch",
}


def is_technical_finding(finding: TabularFinding | dict) -> bool:
    finding = finding.model_dump() if isinstance(finding, TabularFinding) else finding
    code = finding.get("code")
    return code in _TECHNICAL_FINDINGS or (
        finding.get("severity") == "error" and code not in _ACTION_BY_FINDING
    )


def build_review_options(finding: TabularFinding, evaluation: TabularEvaluation) -> list[dict]:
    """Use the same eligibility checks for offered and executable actions."""
    action = _review_action(finding, evaluation)
    match action:
        case "replace_block":
            return [{
                "action": action,
                "description": "Corregir el rol, los encabezados o los límites del bloque afectado.",
                "values": _block_values_schema(),
            }]
        case "correct_transformations":
            return [{
                "action": action,
                "description": "Corregir las operaciones del bloque y volver a verificarlas.",
                "values": _transformation_values_schema(),
            }]
        case "add_block":
            return [{
                "action": action,
                "description": "Declarar un bloque de la hoja afectada para conservar las celdas sin tratamiento.",
                "values": _block_values_schema(),
            }]
        case "clarify_record_identity":
            return [{
                "action": action,
                "description": "Confirmar que los registros cuestionados en esta nota son distintos; conservar los datos originales.",
                "values": {
                    "type": "object", "required": ["identity", "statement"], "additionalProperties": False,
                    "properties": {
                        "identity": {"const": "distinct"},
                        "statement": {"type": "string", "minLength": 1, "maxLength": 4000},
                    },
                },
            }]
        case _:
            return []


def _review_action(finding: TabularFinding, evaluation: TabularEvaluation) -> str | None:
    if evaluation.recipe is None:
        return None
    action = _ACTION_BY_FINDING.get(finding.code)
    match action:
        case "replace_block" | "correct_transformations":
            if any(block.sheet_name == finding.sheet_name and block.block_id == finding.block_id
                   for block in evaluation.recipe.sheets):
                return action
        case "add_block":
            if finding.sheet_name is not None:
                return action
    if finding.code == "source_record_identity_uncertainty" and original_note(evaluation.structural_summary, finding):
        return "clarify_record_identity"
    return None


def _transformation_values_schema() -> dict:
    transformations = TypeAdapter(
        Annotated[
            SheetRecipe.model_fields["transformations"].rebuild_annotation(),
            Field(title="Transformations"),
        ]
    ).json_schema()
    definitions = transformations.pop("$defs", {})
    return {
        "type": "object", "required": ["transformations"], "additionalProperties": False,
        "properties": {"transformations": transformations}, "$defs": definitions,
    }


def _block_values_schema() -> dict:
    block = SheetRecipe.model_json_schema()
    definitions = block.pop("$defs", {})
    return {
        "type": "object", "required": ["block"], "additionalProperties": False,
        "properties": {"block": block}, "$defs": definitions,
    }


def original_note(summary: StructuralSummary, finding: TabularFinding) -> str | None:
    if finding.sheet_name is None or finding.cell is None:
        return None
    position = cell_position(finding.cell)
    if position is None:
        return None
    row_number, column = position
    for sheet in summary.sheets:
        if sheet.name != finding.sheet_name:
            continue
        samples = [(sheet.sample, sheet.column_indices)] + [
            (region.sample, region.column_indices) for region in sheet.regions
        ]
        for rows, columns in samples:
            if column not in columns:
                continue
            index = columns.index(column)
            for row in rows:
                if row.row_number != row_number or index >= len(row.values):
                    continue
                value = row.values[index]
                return value if isinstance(value, str) and value.strip() else None
    return None
