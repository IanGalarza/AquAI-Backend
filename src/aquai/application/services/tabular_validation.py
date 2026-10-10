from aquai.application.dtos.tabular import (
    ExecutionResult,
    PreparationRecipe,
    SheetExecution,
    SheetSummary,
    StructuralSummary,
    ValidationResult,
)
from aquai.domain.enums.tabular import EvaluationStatus


class TabularValidationService:
    def validate(
        self,
        summary: StructuralSummary,
        recipe: PreparationRecipe,
        execution: ExecutionResult,
    ) -> ValidationResult:
        warnings = list(summary.warnings) + list(recipe.warnings)
        errors: list[str] = []
        requires_review = recipe.requires_human_review or recipe.status == "review_required"

        if recipe.status == "failed":
            errors.append("La receta indica que el archivo no pudo prepararse.")
        if requires_review:
            warnings.append("La receta solicita revisión humana.")

        source_sheets = {sheet.name: sheet for sheet in summary.sheets}
        planned_names = {sheet.sheet_name for sheet in recipe.sheets}
        executed_names = {sheet.sheet_name for sheet in execution.sheets}
        skipped_names = set(execution.skipped_sheets)

        for sheet in summary.sheets:
            warnings.extend(f"Hoja '{sheet.name}': {warning}" for warning in sheet.warnings)

        for name in sorted(planned_names - source_sheets.keys()):
            errors.append(f"La receta referencia una hoja inexistente: '{name}'.")
        for name in sorted(planned_names - executed_names):
            errors.append(f"No se ejecutó la hoja '{name}' incluida en la receta.")
        for name in sorted(executed_names - planned_names):
            errors.append(f"Se ejecutó la hoja '{name}' sin estar incluida en la receta.")
        for name in sorted(executed_names & skipped_names):
            errors.append(f"La hoja '{name}' figura como ejecutada y omitida.")
        if len(executed_names) != len(execution.sheets):
            errors.append("La ejecución repite una hoja.")

        omitted_names = skipped_names | (source_sheets.keys() - executed_names)
        for name in sorted(omitted_names):
            sheet = source_sheets.get(name)
            if sheet is None:
                requires_review = True
                warnings.append(f"Se informó una hoja omitida desconocida: '{name}'.")
            elif self._has_source_data(sheet):
                requires_review = True
                warnings.append(f"Se omitió la hoja '{name}', que contiene datos.")
            else:
                warnings.append(f"Se omitió la hoja vacía '{name}'.")

        for sheet in execution.sheets:
            sheet_warnings, sheet_errors, sheet_requires_review = self._validate_sheet(sheet)
            warnings.extend(sheet_warnings)
            errors.extend(sheet_errors)
            requires_review = requires_review or sheet_requires_review

        if recipe.status == "review_required" and not recipe.sheets:
            warnings.append("La receta no propone ninguna hoja para ejecutar; requiere revisión.")
        elif not any(sheet.rows_after > 0 and sheet.columns_after for sheet in execution.sheets):
            errors.append("La ejecución no produjo ninguna tabla con filas y columnas de datos.")

        if errors:
            status = EvaluationStatus.FAILED
        elif requires_review:
            status = EvaluationStatus.REVIEW_REQUIRED
        elif warnings:
            status = EvaluationStatus.READY_WITH_WARNINGS
        else:
            status = EvaluationStatus.READY

        return ValidationResult(
            status=status,
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
        )

    def _has_source_data(self, sheet: SheetSummary) -> bool:
        empty_rows = max(sheet.empty_row_count, len(set(sheet.empty_rows)))
        empty_columns = max(sheet.empty_column_count, len(set(sheet.empty_columns)))
        return sheet.formula_count > 0 or sheet.error_cell_count > 0 or (
            sheet.rows > empty_rows and sheet.columns > empty_columns
        )

    def _validate_sheet(self, sheet: SheetExecution) -> tuple[list[str], list[str], bool]:
        prefix = f"Hoja '{sheet.sheet_name}': "
        warnings = [prefix + warning for warning in sheet.warnings]
        errors = [prefix + error for error in sheet.errors]
        requires_review = False

        counts = (
            sheet.source_rows,
            sheet.source_columns,
            sheet.rows_before,
            sheet.rows_after,
            sheet.excluded_nonempty_rows,
            sheet.dropped_empty_rows,
            *sheet.conversion_failures.values(),
        )
        if any(count < 0 for count in counts):
            errors.append(prefix + "La ejecución informó cantidades negativas.")
        if sheet.rows_after == 0:
            errors.append(prefix + "La tabla final no tiene filas de datos.")
        if not sheet.columns_after:
            errors.append(prefix + "La tabla final no tiene columnas.")

        if sheet.excluded_nonempty_rows:
            requires_review = True
            warnings.append(
                prefix + f"Se excluyeron {sheet.excluded_nonempty_rows} filas no vacías del rango de datos."
            )
        if sheet.suspicious_header_rows:
            requires_review = True
            warnings.append(prefix + "Los encabezados seleccionados podrían contener registros de datos.")

        # rows_before ya excluye los encabezados declarados en la receta.
        row_difference = sheet.rows_before - sheet.dropped_empty_rows - sheet.rows_after
        if row_difference > 0:
            requires_review = True
            warnings.append(
                prefix + f"Se perdieron {row_difference} filas sin una eliminación de vacíos que lo explique."
            )
        elif row_difference < 0:
            requires_review = True
            warnings.append(prefix + "Las cantidades de filas antes y después no concuerdan con la limpieza informada.")

        column_difference = (
            len(sheet.columns_before) - len(sheet.dropped_empty_columns) - len(sheet.columns_after)
        )
        if column_difference:
            requires_review = True
            warnings.append(
                prefix + "Las cantidades de columnas antes y después no concuerdan con la limpieza informada."
            )
        if (
            len(set(sheet.columns_before)) != len(sheet.columns_before)
            or len(set(sheet.columns_after)) != len(sheet.columns_after)
        ):
            requires_review = True
            warnings.append(prefix + "Se detectaron nombres de columnas duplicados.")
        if any(not column.strip() for column in sheet.columns_after):
            requires_review = True
            warnings.append(prefix + "La tabla final contiene columnas sin nombre.")

        for column, failures in sheet.conversion_failures.items():
            if failures > 0:
                requires_review = True
                warnings.append(prefix + f"Fallaron {failures} conversiones en la columna '{column}'.")
        if sheet.unresolved_formula_cells:
            requires_review = True
            warnings.append(prefix + "Hay fórmulas cuyo resultado no está disponible.")
        if sheet.error_cells:
            requires_review = True
            warnings.append(prefix + "La hoja contiene celdas con errores de Excel.")

        if sheet.empty_columns:
            warnings.append(prefix + "La tabla final contiene columnas completamente vacías.")
        if sheet.dropped_empty_rows:
            warnings.append(prefix + f"Se eliminaron {sheet.dropped_empty_rows} filas vacías.")
        if sheet.dropped_empty_columns:
            warnings.append(prefix + f"Se eliminaron {len(sheet.dropped_empty_columns)} columnas vacías.")

        return warnings, errors, requires_review
