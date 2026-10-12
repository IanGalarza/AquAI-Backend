"""Validate execution coverage and conservation of source data."""

from collections import Counter

from aquai.application.contracts.tabular.recipe import PreparationRecipe, SheetRecipe
from aquai.application.contracts.tabular.results import ExecutionResult, SheetExecution, TabularFinding
from aquai.application.contracts.tabular.structure import SheetSummary, StructuralSummary
from aquai.application.validators.tabular.preparation import AuxiliaryRows
from aquai.application.validators.tabular.structure import block_range


def verified_empty_sheets(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
) -> tuple[set[str], list[TabularFinding]]:
    sources = {sheet.name: sheet for sheet in summary.sheets}
    evidence = {item.sheet_name: item for item in execution.sheet_content_evidence}
    verified: set[str] = set()
    findings: list[TabularFinding] = []
    if len(evidence) != len(execution.sheet_content_evidence):
        findings.append(TabularFinding(
            code="empty_sheet_evidence_mismatch", message="La ejecución repite evidencia de contenido de una hoja.",
            severity="error",
        ))
    for treatment in execution.empty_sheets:
        source = sources.get(treatment.sheet_name)
        expected_ids = [block.block_id for block in recipe.sheets if block.sheet_name == treatment.sheet_name]
        proof = treatment.evidence
        coherent = (
            source is not None and source.content_evidence is not None
            and proof.sheet_name == treatment.sheet_name and proof.is_empty
            and proof.inspected_cells == source.rows * source.columns
            and proof == source.content_evidence == evidence.get(treatment.sheet_name)
            and source.formula_count == 0 and source.error_cell_count == 0
            and Counter(treatment.skipped_block_ids) == Counter(expected_ids)
            and treatment.sheet_name not in verified
            and not any(sheet.sheet_name == treatment.sheet_name for sheet in execution.sheets)
            and all(coverage.nonempty_cells == 0 for coverage in execution.coverage
                    if coverage.sheet_name == treatment.sheet_name)
        )
        if not coherent:
            findings.append(TabularFinding(
                code="empty_sheet_evidence_mismatch",
                message="El tratamiento de hoja vacía no coincide con la inspección completa y la ejecución.",
                sheet_name=treatment.sheet_name, severity="error",
            ))
            continue
        verified.add(treatment.sheet_name)
        findings.append(TabularFinding(
            code="verified_empty_sheet",
            message=(f"Hoja vacía verificada: {proof.inspected_cells} celdas examinadas, "
                     "0 valores y 0 fórmulas. No se genera tabla; la receta original se conserva."),
            sheet_name=treatment.sheet_name, requires_review=False,
        ))
    return verified, findings


def _block_label(sheet_name: str, block_id: str | None) -> str:
    if block_id is None:
        return f"la hoja '{sheet_name}'"
    return f"el bloque '{block_id}' de la hoja '{sheet_name}'"


def _has_source_data(sheet: SheetSummary) -> bool:
    if sheet.content_evidence is not None:
        return not sheet.content_evidence.is_empty or sheet.formula_count > 0 or sheet.error_cell_count > 0
    empty_rows = max(sheet.empty_row_count, len(set(sheet.empty_rows)))
    empty_columns = max(sheet.empty_column_count, len(set(sheet.empty_columns)))
    return sheet.formula_count > 0 or sheet.error_cell_count > 0 or (
        sheet.rows > empty_rows and sheet.columns > empty_columns
    )


def validate_executed_sheet(
    sheet: SheetExecution, *, coverage_available: bool = False,
    auxiliary_rows: int = 0, auxiliary_formula_cells: int = 0,
) -> tuple[list[str], list[str], list[TabularFinding]]:
    prefix = f"Hoja '{sheet.sheet_name}': "
    warnings = [prefix + warning for warning in sheet.warnings]
    errors = [prefix + error for error in sheet.errors]
    findings: list[TabularFinding] = []

    def review(code: str, message: str) -> None:
        findings.append(TabularFinding(
            code=code, message=message, sheet_name=sheet.sheet_name, block_id=sheet.block_id,
        ))

    counts = (
        sheet.source_rows,
        sheet.source_columns,
        sheet.rows_before,
        sheet.rows_after,
        sheet.excluded_nonempty_rows,
        sheet.dropped_empty_rows,
        sheet.auxiliary_rows_removed,
        sheet.formula_count,
        sheet.unresolved_formula_count,
        sheet.error_cell_count,
        *(value for value in (sheet.source_nonempty_cells, sheet.preserved_nonempty_cells) if value is not None),
        *sheet.conversion_failures.values(),
    )
    if any(count < 0 for count in counts):
        errors.append(prefix + "La ejecución informó cantidades negativas.")
    if sheet.rows_after == 0:
        errors.append(prefix + "La tabla final no tiene filas de datos.")
    if not sheet.columns_after:
        errors.append(prefix + "La tabla final no tiene columnas.")

    if sheet.excluded_nonempty_rows and not coverage_available:
        review("excluded_source_rows", f"Se excluyeron {sheet.excluded_nonempty_rows} filas no vacías del rango de datos.")
    if sheet.suspicious_header_rows:
        review("suspicious_header_rows", "Los encabezados seleccionados podrían contener registros de datos.")

    # rows_before ya excluye los encabezados declarados en la receta.
    if sheet.auxiliary_rows_removed != auxiliary_rows:
        errors.append(prefix + "Las filas auxiliares omitidas no coinciden con su evidencia verificada.")
    row_difference = sheet.rows_before - sheet.dropped_empty_rows - auxiliary_rows - sheet.rows_after
    if row_difference > 0:
        review("unexplained_row_loss", f"Se perdieron {row_difference} filas sin una eliminación de vacíos que lo explique.")
    elif row_difference < 0:
        review("row_count_mismatch", "Las cantidades de filas antes y después no concuerdan con la limpieza informada.")

    column_difference = (
        len(sheet.columns_before) - len(sheet.dropped_empty_columns) - len(sheet.columns_after)
    )
    if column_difference:
        review("column_count_mismatch", "Las cantidades de columnas antes y después no concuerdan con la limpieza informada.")
    if (
        len(set(sheet.columns_before)) != len(sheet.columns_before)
        or len(set(sheet.columns_after)) != len(sheet.columns_after)
    ):
        review("duplicate_column_names", "Se detectaron nombres de columnas duplicados.")
    if any(not column.strip() for column in sheet.columns_after):
        review("unnamed_columns", "La tabla final contiene columnas sin nombre.")

    for column, failures in sheet.conversion_failures.items():
        if failures > 0:
            review("conversion_failures", f"Fallaron {failures} conversiones en la columna '{column}'.")
    if (
        sheet.source_nonempty_cells is not None
        and sheet.preserved_nonempty_cells is not None
        and sheet.source_nonempty_cells != sheet.preserved_nonempty_cells + auxiliary_formula_cells
    ):
        review("nonempty_value_loss", "La cantidad de celdas no vacías no se conservó durante las transformaciones.")
    if sheet.unresolved_formula_cells or sheet.unresolved_formula_count:
        review("unresolved_formula_cache", "Hay fórmulas cuyo resultado no está disponible.")
    if sheet.error_cells or sheet.error_cell_count:
        review("excel_error_cells", "La hoja contiene celdas con errores de Excel.")

    if sheet.empty_columns:
        warnings.append(prefix + "La tabla final contiene columnas completamente vacías.")
    if sheet.dropped_empty_rows:
        warnings.append(prefix + f"Se eliminaron {sheet.dropped_empty_rows} filas vacías.")
    if sheet.dropped_empty_columns:
        warnings.append(prefix + f"Se eliminaron {len(sheet.dropped_empty_columns)} columnas vacías.")

    return warnings, errors, findings


def validate_source_cells(
    summary: StructuralSummary, auxiliary_rows: AuxiliaryRows,
) -> tuple[list[str], list[TabularFinding]]:
    warnings: list[str] = []
    findings: list[TabularFinding] = []
    auxiliary_cells_by_sheet = Counter(
        treatment.sheet_name for treatments in auxiliary_rows.values()
        for treatment in treatments for _ in treatment.auxiliary_cells
    )
    for sheet in summary.sheets:
        warnings.extend(f"Hoja '{sheet.name}': {warning}" for warning in sheet.warnings)
        remaining_missing = sheet.missing_formula_cache_count - auxiliary_cells_by_sheet[sheet.name]
        if remaining_missing < 0:
            findings.append(TabularFinding(
                code="auxiliary_row_evidence_mismatch", sheet_name=sheet.name,
                message="El tratamiento auxiliar excede las fórmulas sin caché de la fuente.", severity="error",
            ))
        if remaining_missing > 0:
            auxiliary_coordinates = {
                formula.cell for treatments in auxiliary_rows.values() for treatment in treatments
                if treatment.sheet_name == sheet.name for formula in treatment.auxiliary_cells
            }
            missing_sample = next(
                (formula.cell for formula in sheet.formula_sample
                 if not formula.cache_available and formula.cell not in auxiliary_coordinates), None,
            )
            findings.append(TabularFinding(
                code="source_missing_formula_cache",
                message=(
                    f"Fórmulas sin resultado almacenado en la fuente: {remaining_missing}. "
                    "El control incluye cabeceras y regiones separadas."
                ),
                sheet_name=sheet.name,
                cell=missing_sample,
            ))
        if sheet.error_cell_count:
            findings.append(TabularFinding(
                code="source_excel_error",
                message=(
                    f"Errores de Excel en la fuente: {sheet.error_cell_count}. "
                    "El control incluye cabeceras y regiones separadas."
                ),
                sheet_name=sheet.name,
                cell=sheet.error_cells[0] if sheet.error_cells else None,
            ))
    return warnings, findings


def validate_planned_blocks(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
    empty_names: set[str],
) -> list[str]:
    errors: list[str] = []
    source_sheets = {sheet.name: sheet for sheet in summary.sheets}
    planned_names = {sheet.sheet_name for sheet in recipe.sheets}
    executed_names = {sheet.sheet_name for sheet in execution.sheets}
    planned_blocks = {(sheet.sheet_name, sheet.block_id) for sheet in recipe.sheets}
    executed_blocks = {(sheet.sheet_name, sheet.block_id) for sheet in execution.sheets}
    skipped_names = set(execution.skipped_sheets)
    for name in sorted(planned_names - source_sheets.keys()):
        errors.append(f"La receta referencia una hoja inexistente: '{name}'.")
    treated_blocks = {(block.sheet_name, block.block_id) for block in recipe.sheets
                      if block.sheet_name in empty_names}
    for name, block_id in sorted(planned_blocks - executed_blocks - treated_blocks, key=str):
        errors.append(f"No se ejecutó {_block_label(name, block_id)} incluido en la receta.")
    for name, block_id in sorted(executed_blocks - planned_blocks, key=str):
        errors.append(f"Se ejecutó {_block_label(name, block_id)} sin estar incluido en la receta.")
    for name in sorted(executed_names & skipped_names):
        errors.append(f"La hoja '{name}' figura como ejecutada y omitida.")
    if len(executed_blocks) != len(execution.sheets):
        errors.append("La ejecución repite un bloque de una hoja.")
    return errors


def validate_coverage(
    summary: StructuralSummary, execution: ExecutionResult,
) -> tuple[list[str], list[TabularFinding]]:
    errors: list[str] = []
    findings: list[TabularFinding] = []
    source_sheets = {sheet.name: sheet for sheet in summary.sheets}
    coverage_names = {coverage.sheet_name for coverage in execution.coverage}
    if len(coverage_names) != len(execution.coverage):
        errors.append("La ejecución repite la cobertura de una hoja.")
    for coverage in execution.coverage:
        if coverage.sheet_name not in source_sheets:
            errors.append(f"La cobertura referencia una hoja inexistente: '{coverage.sheet_name}'.")
        if (
            coverage.covered_nonempty_cells + coverage.uncovered_nonempty_cells != coverage.nonempty_cells
            or coverage.overlapping_nonempty_cells > coverage.covered_nonempty_cells
        ):
            errors.append(f"Hoja '{coverage.sheet_name}': Las cantidades de cobertura no concuerdan.")
        if coverage.uncovered_nonempty_cells:
            findings.append(TabularFinding(
                code="uncovered_source_cells",
                message=f"Quedaron {coverage.uncovered_nonempty_cells} celdas no vacías sin tratamiento declarado.",
                sheet_name=coverage.sheet_name,
                cell=coverage.uncovered_sample[0] if coverage.uncovered_sample else None,
            ))
        if coverage.overlapping_nonempty_cells:
            findings.append(TabularFinding(
                code="overlapping_source_cells",
                message=f"Se asignaron {coverage.overlapping_nonempty_cells} celdas no vacías a varios bloques de datos.",
                sheet_name=coverage.sheet_name,
                severity="error",
            ))
    return errors, findings


def validate_omitted_sheets(
    summary: StructuralSummary, execution: ExecutionResult, empty_names: set[str],
) -> list[TabularFinding]:
    findings: list[TabularFinding] = []
    source_sheets = {sheet.name: sheet for sheet in summary.sheets}
    executed_names = {sheet.sheet_name for sheet in execution.sheets}
    skipped_names = set(execution.skipped_sheets)
    omitted_names = skipped_names | (source_sheets.keys() - executed_names)
    for name in sorted(omitted_names - empty_names):
        sheet = source_sheets.get(name)
        if sheet is None:
            message = f"Se informó una hoja omitida desconocida: '{name}'."
        else:
            message = (f"Se omitió la hoja '{name}', que contiene datos."
                       if _has_source_data(sheet) else
                       f"Se omitió la hoja '{name}' sin un tratamiento de vacío verificado.")
        findings.append(TabularFinding(code="unhandled_source_sheet", message=message, sheet_name=name))
    return findings


def validate_executed_blocks(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
    auxiliary_rows: AuxiliaryRows,
) -> tuple[list[str], list[str], list[TabularFinding]]:
    warnings: list[str] = []
    errors: list[str] = []
    findings: list[TabularFinding] = []
    source_sheets = {sheet.name: sheet for sheet in summary.sheets}
    plans = {(block.sheet_name, block.block_id): block for block in recipe.sheets}
    coverage_names = {coverage.sheet_name for coverage in execution.coverage}
    for sheet in execution.sheets:
        plan = plans.get((sheet.sheet_name, sheet.block_id))
        if plan is not None and sheet.role != plan.role:
            errors.append(f"Hoja '{sheet.sheet_name}': El rol ejecutado difiere del declarado en la receta.")
        source = source_sheets.get(sheet.sheet_name)
        findings.extend(_validate_execution_range(sheet, plan, source))
        findings.extend(sheet.findings)
        treatments = auxiliary_rows.get((sheet.sheet_name, sheet.block_id), [])
        removed_treatments = [treatment for treatment in treatments if treatment.removed_from_observations]
        sheet_warnings, sheet_errors, sheet_findings = validate_executed_sheet(
            sheet, coverage_available=sheet.sheet_name in coverage_names,
            auxiliary_rows=len(removed_treatments),
            auxiliary_formula_cells=sum(len(treatment.auxiliary_cells) for treatment in removed_treatments),
        )
        warnings.extend(sheet_warnings)
        errors.extend(sheet_errors)
        findings.extend(sheet_findings)
    return warnings, errors, findings


def _validate_execution_range(
    sheet: SheetExecution, plan: SheetRecipe | None, source: SheetSummary | None,
) -> list[TabularFinding]:
    if plan is None or source is None:
        return []
    expected_range = (plan.data_start_row, plan.data_end_row, plan.data_start_column, plan.data_end_column)
    actual_range = (sheet.data_start_row, sheet.data_end_row, sheet.data_start_column, sheet.data_end_column)
    if actual_range == expected_range:
        return []
    return [TabularFinding(
        code="execution_range_mismatch",
        message="Las coordenadas ejecutadas difieren del rango declarado en la receta.",
        sheet_name=sheet.sheet_name,
        block_id=sheet.block_id,
        range=block_range(plan, source),
        severity="error",
    )]


def validate_output(
    summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
    empty_names: set[str],
) -> tuple[list[str], list[TabularFinding]]:
    errors: list[str] = []
    findings: list[TabularFinding] = []
    if len(empty_names) == len({sheet.name for sheet in summary.sheets}):
        return errors, findings
    if recipe.status == "review_required" and not recipe.sheets:
        findings.append(TabularFinding(
            code="unplanned_recipe", message="La receta no propone ninguna hoja para ejecutar; requiere revisión.",
        ))
        return errors, findings
    if not any(sheet.rows_after > 0 and sheet.columns_after for sheet in execution.sheets):
        errors.append("La ejecución no produjo ninguna tabla con filas y columnas de datos.")
    return errors, findings
