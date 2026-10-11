"""Ejecución de políticas corroboradas, sin modificar la receta ni las fuentes."""

from collections import defaultdict
from datetime import date, datetime
from numbers import Number

import pandas as pd
from openpyxl.utils.cell import coordinate_to_tuple, get_column_letter

from aquai.application.dtos.tabular import SheetRecipe, TabularFinding
from aquai.application.dtos.tabular_policies import (
    AssociationDiagnostic,
    AuxiliaryFormulaEvidence,
    AuxiliaryRowTreatment,
    CellNormalization,
    PreparationPolicies,
)
from aquai.infrastructure.processing.tabular.normalization import (
    parse_explicit_date,
    parse_mixed_decimal,
    trace_value,
)
from aquai.infrastructure.processing.tabular.reader import CellMetadata, SheetData


def strict_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return not isinstance(value, str) and bool(pd.isna(value))


def headers_match(sheet: SheetData, row: int, expected: dict[int, str]) -> bool:
    rows, columns = sheet.frame.shape
    return 1 <= row <= rows and all(
        1 <= column <= columns and (row, column) not in sheet.formula_cells
        and sheet.frame.iat[row - 1, column - 1] == label
        for column, label in expected.items()
    )


def cell_metadata(sheet: SheetData, row: int, column: int) -> CellMetadata:
    metadata = sheet.cell_metadata.get((row, column))
    if metadata is not None:
        return metadata
    value = sheet.frame.iat[row - 1, column - 1]
    kind = ("formula" if (row, column) in sheet.formula_cells else
            "empty" if strict_missing(value) else
            "native_datetime" if isinstance(value, (date, datetime)) else
            "native_boolean" if isinstance(value, bool) else
            "native_number" if isinstance(value, Number) else "text")
    return CellMetadata(source_kind=kind)


def _finding(sheet: SheetData, recipe: SheetRecipe, code: str, message: str, cell: str | None = None):
    return TabularFinding(code=code, sheet_name=sheet.name, block_id=recipe.block_id,
                          cell=cell, message=message, requires_review=True)


def prepare_frame(
    sheet: SheetData,
    recipe: SheetRecipe,
    frame: pd.DataFrame,
    policies: PreparationPolicies,
    source_sha256: str,
) -> tuple[pd.DataFrame, list[CellNormalization], list[AuxiliaryRowTreatment], list[TabularFinding]]:
    normalizations, auxiliary_rows, findings = [], [], []
    if recipe.role == "header":
        return frame, normalizations, auxiliary_rows, findings
    is_observations = recipe.role == "observations"
    columns = {column: name for column, name in enumerate(frame.columns, recipe.data_start_column)}

    def trace(row, column, original, prepared, policy, operation):
        metadata = cell_metadata(sheet, row, column)
        normalizations.append(CellNormalization(
            sheet_name=sheet.name, block_id=recipe.block_id,
            cell=f"{get_column_letter(column)}{row}",
            original_value=trace_value(original), prepared_value=trace_value(prepared),
            rule_id=policy.rule_id, version=policy.version, origin=policy.origin,
            operation=operation, number_format=metadata.number_format,
            source_kind=metadata.source_kind, raw_value=metadata.raw_value,
            excel_epoch=metadata.excel_epoch,
        ))

    for policy in policies.confirmed_cells if is_observations else ():
        if policy.sheet_name != sheet.name:
            continue
        row, column = coordinate_to_tuple(policy.cell)
        if row - 1 not in frame.index or column not in columns:
            continue
        original = sheet.frame.iat[row - 1, column - 1]
        metadata = cell_metadata(sheet, row, column)
        verified = (
            source_sha256 == policy.source_sha256
            and headers_match(sheet, policy.header_row, {column: policy.expected_header})
            and (row, column) not in sheet.formula_cells
            and trace_value(original) == policy.expected_source_value
            and metadata.number_format == policy.expected_number_format
            and metadata.source_kind == policy.expected_source_kind
            and (policy.expected_raw_value is None or metadata.raw_value == policy.expected_raw_value)
        )
        if not verified:
            findings.append(_finding(sheet, recipe, "confirmed_cell_evidence_mismatch",
                                     "La celda no coincide con toda la evidencia corroborada; se conserva el original.", policy.cell))
            continue
        frame.at[row - 1, columns[column]] = policy.prepared_value
        trace(row, column, original, policy.prepared_value, policy, "confirmed_cell")

    for operation, column_policies in (("date_text", policies.date_columns if is_observations else ()),
                                       ("number_text", policies.number_columns if is_observations else ())):
        for policy in column_policies:
            selected_columns = [column for column in policy.column_indices if column in columns]
            if policy.sheet_name != sheet.name or not selected_columns:
                continue
            if not headers_match(sheet, policy.header_row, policy.expected_headers):
                findings.append(_finding(sheet, recipe, "preparation_policy_header_mismatch",
                                         f"Los encabezados originales no coinciden con la política {policy.rule_id}; no se aplica la conversión."))
                continue
            for column in selected_columns:
                name = columns[column]
                for index in frame.index:
                    row = int(index) + 1
                    if (row, column) in sheet.formula_cells:
                        continue
                    value = frame.at[index, name]
                    if strict_missing(value):
                        continue
                    if operation == "date_text":
                        parsed = parse_explicit_date(value, policy.text_formats)
                    elif isinstance(value, (date, datetime)):
                        findings.append(_finding(sheet, recipe, "abundance_native_date_unresolved",
                                                 "La abundancia contiene una fecha nativa sin aclaración inequívoca para esta celda; se conserva el original.",
                                                 f"{get_column_letter(column)}{row}"))
                        continue
                    else:
                        parsed = parse_mixed_decimal(value, policy.decimal_separators)
                    if parsed.conflict:
                        if operation == "date_text" and policy.temporal_priority is not None:
                            # The temporal selector contrasts this preserved
                            # value with the primary code and reports either a
                            # resolved warning or an unresolved date. A failed
                            # fallback candidate alone must not block a valid
                            # primary candidate.
                            continue
                        findings.append(_finding(sheet, recipe, "date_text_unresolved" if operation == "date_text" else "number_text_unresolved",
                                                 parsed.conflict, f"{get_column_letter(column)}{row}"))
                    elif parsed.changed:
                        frame.at[index, name] = parsed.value
                        trace(row, column, value, parsed.value, policy, operation)

    removed = set()
    classified = set()
    for policy in policies.observation_rows if recipe.role in {"observations", "derived", "legend", "unknown"} else ():
        if policy.sheet_name != sheet.name:
            continue
        if not headers_match(sheet, policy.header_row, policy.expected_headers):
            findings.append(_finding(sheet, recipe, "preparation_policy_header_mismatch",
                                     f"Los encabezados originales no coinciden con la política {policy.rule_id}; no se omiten filas."))
            continue
        auxiliary_columns = set(policy.auxiliary_column_indices)
        for index in frame.index:
            if index in classified:
                continue
            row = int(index) + 1
            formulas = {column: formula for (formula_row, column), formula in sheet.formula_cells.items()
                        if formula_row == row}
            if not formulas or not set(formulas).issubset(auxiliary_columns):
                continue
            if any(not strict_missing(formula.cached_value) for formula in formulas.values()):
                continue
            if any(not strict_missing(value) for value in sheet.frame.iloc[index]):
                continue
            classified.add(index)
            if is_observations:
                removed.add(index)
            auxiliary_rows.append(AuxiliaryRowTreatment(
                sheet_name=sheet.name, block_id=recipe.block_id, row_number=row,
                rule_id=policy.rule_id, version=policy.version, origin=policy.origin,
                inspected_columns=len(sheet.frame.columns),
                auxiliary_cells=tuple(AuxiliaryFormulaEvidence(
                    cell=f"{get_column_letter(column)}{row}", formula=formula.expression,
                ) for column, formula in sorted(formulas.items())),
                removed_from_observations=is_observations,
            ))
    if removed:
        frame = frame.drop(index=sorted(removed)).copy()
    return frame, normalizations, auxiliary_rows, findings


def association_diagnostics(sheets: list[SheetData], policies: PreparationPolicies) -> tuple[list[AssociationDiagnostic], list[TabularFinding]]:
    sources = {sheet.name: sheet for sheet in sheets}
    diagnostics, findings = [], []

    def key(value):
        if hasattr(value, "item"):
            value = value.item()
        return type(value).__qualname__, value

    for policy in policies.associations:
        source, target = sources.get(policy.source_sheet), sources.get(policy.target_sheet)
        if (source is None or target is None
                or not headers_match(source, policy.header_row, {policy.source_column: policy.expected_source_header})
                or not headers_match(target, policy.header_row, {policy.target_column: policy.expected_target_header})):
            findings.append(TabularFinding(code="association_policy_header_mismatch", sheet_name=policy.source_sheet,
                                           message=f"La asociación opcional {policy.rule_id} no se aplica porque falta la evidencia de sus encabezados.",
                                           requires_review=True))
            continue
        destinations = defaultdict(list)
        for row in range(policy.target_start_row, len(target.frame) + 1):
            value = target.frame.iat[row - 1, policy.target_column - 1]
            if not strict_missing(value):
                destinations[key(value)].append(row)
        for row in range(policy.source_start_row, len(source.frame) + 1):
            value = source.frame.iat[row - 1, policy.source_column - 1]
            if strict_missing(value):
                continue
            target_rows = tuple(destinations.get(key(value), ()))
            status = "missing" if not target_rows else "matched" if len(target_rows) == 1 else "ambiguous"
            diagnostics.append(AssociationDiagnostic(
                source_sheet=source.name, source_row=row,
                source_cell=f"{get_column_letter(policy.source_column)}{row}", source_value=trace_value(value),
                target_sheet=target.name, target_rows=target_rows, status=status,
                requires_review=status == "ambiguous", rule_id=policy.rule_id,
                version=policy.version, origin=policy.origin,
            ))
    return diagnostics, findings
