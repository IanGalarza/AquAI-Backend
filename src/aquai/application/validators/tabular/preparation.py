"""Verify the original values and provenance required by preparation policies."""

import re
from collections import Counter
from datetime import date, datetime
from numbers import Number
from typing import Literal

from pydantic import JsonValue

from aquai.application.contracts.tabular.policies import DateColumnPolicy, PreparationPolicies
from aquai.application.contracts.tabular.recipe import PreparationRecipe
from aquai.application.contracts.tabular.results import AuxiliaryRowTreatment, ExecutionResult, TabularFinding, TemporalDateTreatment
from aquai.application.contracts.tabular.structure import SheetSummary, SourceHeaderEvidence, StructuralSummary


SourceHeaders = dict[tuple[str, int], SourceHeaderEvidence]
AuxiliaryRows = dict[tuple[str, str | None], list[AuxiliaryRowTreatment]]


def validate_temporal_dates(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
    policies: PreparationPolicies | None, source_headers: SourceHeaders,
) -> list[TabularFinding]:
    """Una advertencia temporal sólo queda resuelta con selección y procedencia corroboradas."""
    sources = {sheet.name: sheet for sheet in summary.sheets}
    plans = {(block.sheet_name, block.block_id): block for block in recipe.sheets}
    temporal_policies = [policy for policy in policies.date_columns if policy.temporal_priority] if policies else []
    reported = [*execution.findings, *(finding for sheet in execution.sheets for finding in sheet.findings)]
    findings, verified = [], Counter()
    seen = set()
    for treatment in execution.temporal_dates:
        source = sources.get(treatment.sheet_name)
        plan = plans.get((treatment.sheet_name, treatment.block_id))
        position = cell_position(treatment.date_cell)
        policy = next((policy for policy in temporal_policies
                       if (policy.sheet_name, policy.rule_id, policy.version, policy.origin)
                       == (treatment.sheet_name, treatment.rule_id, treatment.version, treatment.origin)
                       and position is not None and position[1] in policy.column_indices), None)
        fields = {"date": position[1], **policy.related_date_columns} if position and policy else {}
        key = (treatment.sheet_name, treatment.block_id, treatment.date_cell)
        valid = (
            source is not None and plan is not None and policy is not None
            and plan.role == "observations"
            and position[0] == treatment.source_row
            and plan.data_start_row <= treatment.source_row <= plan.data_end_row
            and plan.data_start_column <= position[1] <= plan.data_end_column
            and policy_headers_match(source, policy.expected_headers, source_headers, policy.header_row)
            and key not in seen and treatment.original_values.keys() == fields.keys()
            and all(cell_position(treatment.original_evidence[field].cell)
                    == (treatment.source_row, column) for field, column in fields.items())
            and _temporal_selection_matches_originals(treatment, policy)
        )
        if valid:
            sampled = next((row for row in source.sample if row.row_number == treatment.source_row), None)
            if sampled is not None:
                columns = source.column_indices or list(range(1, len(source.initial_columns) + 1))
                sampled_values = dict(zip(columns, sampled.values))
                valid = all(column not in sampled_values or sampled_values[column] == treatment.original_values[field]
                            for field, column in fields.items())
        seen.add(key)
        if not valid:
            findings.append(TabularFinding(
                code="temporal_date_evidence_mismatch", sheet_name=treatment.sheet_name,
                block_id=treatment.block_id, cell=treatment.date_cell, severity="error",
                message="La selección temporal no coincide con la política corroborada, sus valores o su procedencia original.",
            ))
            continue
        verified[(treatment.sheet_name, treatment.block_id, position[1])] += 1
        findings.extend(_temporal_resolution_findings(treatment, reported))
    for policy in temporal_policies:
        for sheet in execution.sheets:
            if sheet.sheet_name != policy.sheet_name or sheet.role != "observations":
                continue
            plan = plans.get((sheet.sheet_name, sheet.block_id))
            if plan is None:
                continue
            for column in policy.column_indices:
                if not plan.data_start_column <= column <= plan.data_end_column:
                    continue
                count = verified[(sheet.sheet_name, sheet.block_id, column)]
                if count == sheet.rows_after:
                    continue
                findings.append(TabularFinding(
                    code="temporal_date_evidence_missing", sheet_name=sheet.sheet_name,
                    block_id=sheet.block_id, severity="error",
                    message=(f"La selección temporal conserva {count} registros con evidencia válida, "
                             f"pero el bloque tiene {sheet.rows_after} observaciones. No se resuelve la revisión sin esa procedencia."),
                ))
    return findings


def _temporal_resolution_findings(
    treatment: TemporalDateTreatment, reported: list[TabularFinding],
) -> list[TabularFinding]:
    located = [finding for finding in reported
               if (finding.sheet_name, finding.block_id, finding.cell)
               == (treatment.sheet_name, treatment.block_id, treatment.date_cell)]
    match treatment.resolution:
        case "review_required":
            if any(finding.code == "temporal_date_unresolved" and finding.requires_review for finding in located):
                return []
            return [TabularFinding(
                code="temporal_date_unresolved", sheet_name=treatment.sheet_name,
                block_id=treatment.block_id, cell=treatment.date_cell,
                message=(f"Fila {treatment.source_row}: ni Código Fecha ni Fecha permiten determinar una fecha válida; "
                         "Mes y Año se conservan únicamente como contraste. Revisar los valores originales."),
            )]
        case "user_priority" if treatment.discrepancies:
            if any(finding.code == "temporal_priority_discrepancies_resolved" and not finding.requires_review
                   for finding in located):
                return []
            return [TabularFinding(
                code="temporal_priority_discrepancies_resolved", sheet_name=treatment.sheet_name,
                block_id=treatment.block_id, cell=treatment.date_cell, requires_review=False,
                message=(f"Fecha {treatment.normalized_date} seleccionada desde {treatment.source_cell} por "
                         f"{treatment.rule_id} v{treatment.version}: "
                         + "; ".join(discrepancy.message for discrepancy in treatment.discrepancies)
                         + " Se conservan los originales; la política no constituye una confirmación científica."),
            )]
        case _:
            return []


def _temporal_selection_matches_originals(treatment: TemporalDateTreatment, policy: DateColumnPolicy) -> bool:
    chosen_code, interpretation = _original_date_code(treatment)
    chosen_date = _original_date_text(treatment.original_values.get("date"), policy.text_formats)
    chosen = chosen_code or chosen_date
    match treatment.resolution:
        case "review_required":
            return chosen is None
        case "user_priority":
            return (
                chosen is not None
                and treatment.normalized_date == chosen
                and treatment.source_field == ("date_code" if chosen_code else "date")
                and (not chosen_code or treatment.code_interpretation == interpretation)
            )
        case _:
            return False


def _original_date_code(
    treatment: TemporalDateTreatment,
) -> tuple[str | None, Literal["numeric_leading_zero", "exact_8_digits"]]:
    value = treatment.original_values.get("date_code")
    interpretation = "exact_8_digits"
    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            text = str(int(value)) if value == int(value) and value >= 0 else ""
        except (ValueError, OverflowError, TypeError):
            text = ""
        evidence = treatment.original_evidence.get("date_code")
        if (len(text) == 7 and evidence is not None
                and evidence.source_kind == "native_number" and evidence.number_format == "00000000"):
            text = "0" + text
            interpretation = "numeric_leading_zero"
    else:
        text = value.strip() if isinstance(value, str) else ""
    if not re.fullmatch(r"[0-9]{8}", text):
        return None, interpretation
    try:
        return date(int(text[4:]), int(text[2:4]), int(text[:2])).isoformat(), interpretation
    except ValueError:
        return None, interpretation


def _original_date_text(value: JsonValue, text_formats: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?", text):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
    for explicit_format in text_formats:
        try:
            return datetime.strptime(text, explicit_format).date().isoformat()
        except ValueError:
            continue
    return None


def verified_source_headers(
    summary: StructuralSummary, execution: ExecutionResult, policies: PreparationPolicies | None,
) -> tuple[SourceHeaders, list[TabularFinding]]:
    sources = {sheet.name: sheet for sheet in summary.sheets}
    expected = set()
    if policies:
        for group in (policies.date_columns, policies.number_columns, policies.confirmed_cells, policies.observation_rows):
            expected.update((policy.sheet_name, policy.header_row) for policy in group)
        for policy in policies.associations:
            expected.update(((policy.source_sheet, policy.header_row), (policy.target_sheet, policy.header_row)))
    verified, findings = {}, []
    seen = set()
    for proof in execution.source_headers:
        key = (proof.sheet_name, proof.row_number)
        source = sources.get(proof.sheet_name)
        compact = dict(zip(source.column_indices or range(1, len(source.initial_columns) + 1),
                           source.initial_columns)) if source else {}
        sampled_formula_columns = {
            position[1] for cell in source.formula_sample
            if (position := cell_position(cell.cell)) is not None and position[0] == proof.row_number
        } if source else set()
        valid = (
            source is not None and key in expected and key not in seen
            and proof.row_number <= source.rows and proof.inspected_columns == source.columns
            and set(proof.values) == set(range(1, source.columns + 1))
            and set(proof.formula_columns).issubset(proof.values)
            and len(set(proof.formula_columns)) == len(proof.formula_columns)
            and len(proof.formula_columns) <= source.formula_count
            and sampled_formula_columns.issubset(proof.formula_columns)
            and (proof.row_number != 1 or all(
                _summary_header_value(proof.values.get(column), column) == name
                for column, name in compact.items()
            ))
        )
        seen.add(key)
        if not valid:
            verified.pop(key, None)
            findings.append(TabularFinding(
                code="source_header_evidence_mismatch", sheet_name=proof.sheet_name,
                severity="error", message="La prueba de encabezados originales es incompleta o contradice la fuente inspeccionada.",
            ))
        else:
            verified[key] = proof
    for sheet_name, row in sorted(expected - verified.keys()):
        findings.append(TabularFinding(
            code="source_header_evidence_missing", sheet_name=sheet_name, cell=f"A{row}",
            severity="error", message="Falta la fila completa de encabezados originales requerida por las políticas corroboradas.",
        ))
    return verified, findings


def _summary_header_value(value: JsonValue, column: int) -> str:
    if value is None or isinstance(value, str) and not value.strip():
        return f"column_{column}"
    return value[:200] if isinstance(value, str) else str(value)


def verified_auxiliary_rows(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
    policies: PreparationPolicies | None, source_headers: SourceHeaders,
) -> tuple[AuxiliaryRows, list[TabularFinding]]:
    sources = {sheet.name: sheet for sheet in summary.sheets}
    plans = {(block.sheet_name, block.block_id): block for block in recipe.sheets}
    verified = {}
    findings = []
    seen = set()
    for treatment in execution.auxiliary_rows:
        source = sources.get(treatment.sheet_name)
        block = plans.get((treatment.sheet_name, treatment.block_id))
        policy = next((policy for policy in policies.observation_rows
                       if (policy.sheet_name, policy.rule_id, policy.version, policy.origin)
                       == (treatment.sheet_name, treatment.rule_id, treatment.version, treatment.origin)), None) if policies else None
        coordinates = [cell_position(cell.cell) for cell in treatment.auxiliary_cells]
        sampled_formulas = {cell.cell: cell for cell in source.formula_sample} if source else {}
        key = (treatment.sheet_name, treatment.row_number)
        valid = (
            source is not None and block is not None and policy is not None
            and block.role in {"observations", "derived", "legend", "unknown"}
            and treatment.removed_from_observations == (block.role == "observations")
            and policy_headers_match(source, policy.expected_headers, source_headers, policy.header_row)
            and treatment.inspected_columns == source.columns
            and block.data_start_row <= treatment.row_number <= block.data_end_row
            and key not in seen
            and len(set(coordinates)) == len(coordinates)
            and all(position is not None and position[0] == treatment.row_number
                    and position[1] in policy.auxiliary_column_indices
                    and block.data_start_column <= position[1] <= block.data_end_column
                    for position in coordinates)
            and all(cell.cached_value is None and not cell.cache_available and cell.formula.startswith("=")
                    for cell in treatment.auxiliary_cells)
            and all(cell.cell not in sampled_formulas or (
                sampled_formulas[cell.cell].formula == cell.formula
                and not sampled_formulas[cell.cell].cache_available
                and sampled_formulas[cell.cell].cached_value is None
            ) for cell in treatment.auxiliary_cells)
        )
        if not valid:
            findings.append(TabularFinding(
                code="auxiliary_row_evidence_mismatch", sheet_name=treatment.sheet_name,
                block_id=treatment.block_id, message="El tratamiento de fila auxiliar no tiene evidencia completa y política corroborada.",
                severity="error",
            ))
            continue
        seen.add(key)
        verified.setdefault((treatment.sheet_name, treatment.block_id), []).append(treatment)
        findings.append(TabularFinding(
            code="auxiliary_row_preserved", sheet_name=treatment.sheet_name, block_id=treatment.block_id,
            cell=treatment.auxiliary_cells[0].cell, requires_review=False,
            message=(f"Fila {treatment.row_number} sin valores observacionales: se conservan "
                     "las fórmulas auxiliares como evidencia. " + (
                         "Se omite la fila analítica." if treatment.removed_from_observations
                         else "Se conserva el bloque auxiliar sin convertirlo en observaciones."
                     )),
        ))
    return verified, findings


def validate_associations(
    summary: StructuralSummary, execution: ExecutionResult,
    policies: PreparationPolicies | None, source_headers: SourceHeaders,
) -> list[TabularFinding]:
    """Los vínculos son diagnósticos: no incorporan, filtran ni multiplican filas."""
    sources = {sheet.name: sheet for sheet in summary.sheets}
    findings = []
    seen = set()
    for diagnostic in execution.associations:
        source = sources.get(diagnostic.source_sheet)
        target = sources.get(diagnostic.target_sheet)
        policy = next((policy for policy in policies.associations
                       if (policy.source_sheet, policy.target_sheet, policy.rule_id, policy.version, policy.origin)
                       == (diagnostic.source_sheet, diagnostic.target_sheet, diagnostic.rule_id,
                           diagnostic.version, diagnostic.origin)), None) if policies else None
        position = cell_position(diagnostic.source_cell)
        key = (diagnostic.rule_id, diagnostic.source_sheet, diagnostic.source_row, diagnostic.source_cell)
        expected_status = _association_status(len(diagnostic.target_rows))
        valid = (
            source is not None and target is not None and policy is not None
            and position == (diagnostic.source_row, policy.source_column)
            and policy.source_start_row <= diagnostic.source_row <= source.rows
            and policy_headers_match(source, {policy.source_column: policy.expected_source_header}, source_headers, policy.header_row)
            and policy_headers_match(target, {policy.target_column: policy.expected_target_header}, source_headers, policy.header_row)
            and all(policy.target_start_row <= row <= target.rows for row in diagnostic.target_rows)
            and len(set(diagnostic.target_rows)) == len(diagnostic.target_rows)
            and diagnostic.status == expected_status and key not in seen
        )
        if not valid:
            findings.append(TabularFinding(
                code="association_evidence_mismatch", sheet_name=diagnostic.source_sheet,
                cell=diagnostic.source_cell, severity="error",
                message="El diagnóstico de asociación no coincide con la política y la procedencia corroboradas.",
            ))
            continue
        seen.add(key)
        if diagnostic.status == "ambiguous":
            findings.append(TabularFinding(
                code="ambiguous_optional_association", sheet_name=diagnostic.source_sheet,
                cell=diagnostic.source_cell,
                message=(f"La asociación opcional {diagnostic.rule_id} tiene varios destinos en "
                         f"'{diagnostic.target_sheet}', filas {', '.join(map(str, diagnostic.target_rows))}. "
                         "Se conservan los registros originales sin realizar la asociación."),
            ))
    return findings


def _association_status(target_count: int) -> Literal["missing", "matched", "ambiguous"]:
    match target_count:
        case 0:
            return "missing"
        case 1:
            return "matched"
        case _:
            return "ambiguous"


def cell_position(cell: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell)
    if match is None:
        return None
    column = 0
    for character in match[1]:
        column = column * 26 + ord(character) - ord("A") + 1
    return int(match[2]), column


def policy_headers_match(
    source: SheetSummary, expected_headers: dict[int, str],
    source_headers: SourceHeaders | None = None, header_row: int = 1,
) -> bool:
    if source_headers is not None:
        proof = source_headers.get((source.name, header_row))
        return proof is not None and all(
            column not in proof.formula_columns and proof.values.get(column) == expected
            for column, expected in expected_headers.items()
        )
    positions = source.column_indices or list(range(1, len(source.initial_columns) + 1))
    actual = dict(zip(positions, source.initial_columns))
    return all(actual.get(column) == expected for column, expected in expected_headers.items())
