import re
import unicodedata
from numbers import Number

from aquai.application.dtos.tabular import (
    ExecutionResult,
    PreparationRecipe,
    RenameColumns,
    SheetExecution,
    SheetRecipe,
    SheetSummary,
    StructuralRegion,
    StructuralSummary,
    TabularFinding,
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
        findings = list(execution.findings)
        requires_review = recipe.requires_human_review or recipe.status == "review_required"

        if recipe.status == "failed":
            errors.append("La receta indica que el archivo no pudo prepararse.")
        if requires_review:
            warnings.append("La receta solicita revisión humana.")

        source_sheets = {sheet.name: sheet for sheet in summary.sheets}
        planned_names = {sheet.sheet_name for sheet in recipe.sheets}
        executed_names = {sheet.sheet_name for sheet in execution.sheets}
        planned_blocks = {(sheet.sheet_name, sheet.block_id) for sheet in recipe.sheets}
        executed_blocks = {(sheet.sheet_name, sheet.block_id) for sheet in execution.sheets}
        plans = {(block.sheet_name, block.block_id): block for block in recipe.sheets}
        skipped_names = set(execution.skipped_sheets)

        for sheet in summary.sheets:
            warnings.extend(f"Hoja '{sheet.name}': {warning}" for warning in sheet.warnings)
            if sheet.missing_formula_cache_count:
                missing_sample = next(
                    (formula.cell for formula in sheet.formula_sample if not formula.cache_available), None,
                )
                findings.append(TabularFinding(
                    code="source_missing_formula_cache",
                    message=(
                        f"Fórmulas sin resultado almacenado en la fuente: {sheet.missing_formula_cache_count}. "
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

        for name in sorted(planned_names - source_sheets.keys()):
            errors.append(f"La receta referencia una hoja inexistente: '{name}'.")
        for name, block_id in sorted(planned_blocks - executed_blocks, key=str):
            errors.append(f"No se ejecutó {self._block_label(name, block_id)} incluido en la receta.")
        for name, block_id in sorted(executed_blocks - planned_blocks, key=str):
            errors.append(f"Se ejecutó {self._block_label(name, block_id)} sin estar incluido en la receta.")
        for name in sorted(executed_names & skipped_names):
            errors.append(f"La hoja '{name}' figura como ejecutada y omitida.")
        if len(executed_blocks) != len(execution.sheets):
            errors.append("La ejecución repite un bloque de una hoja.")

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
            plan = plans.get((sheet.sheet_name, sheet.block_id))
            if plan is not None and sheet.role != plan.role:
                errors.append(f"Hoja '{sheet.sheet_name}': El rol ejecutado difiere del declarado en la receta.")
            source = source_sheets.get(sheet.sheet_name)
            if plan is not None and source is not None:
                expected_range = (
                    plan.data_start_row, plan.data_end_row,
                    plan.data_start_column, plan.data_end_column,
                )
                actual_range = (
                    sheet.data_start_row, sheet.data_end_row,
                    sheet.data_start_column, sheet.data_end_column,
                )
                range_mismatch = any(
                    actual != expected
                    for actual, expected in zip(actual_range, expected_range, strict=True)
                )
                if range_mismatch:
                    findings.append(TabularFinding(
                        code="execution_range_mismatch",
                        message="Las coordenadas ejecutadas difieren del rango declarado en la receta.",
                        sheet_name=sheet.sheet_name,
                        block_id=sheet.block_id,
                        range=self._block_range(plan, source),
                        severity="error",
                    ))
            findings.extend(sheet.findings)
            sheet_warnings, sheet_errors, sheet_requires_review = self._validate_sheet(
                sheet, coverage_available=sheet.sheet_name in coverage_names,
            )
            warnings.extend(sheet_warnings)
            errors.extend(sheet_errors)
            requires_review = requires_review or sheet_requires_review

        findings.extend(self._validate_regions(summary, recipe))
        findings.extend(self._validate_unit_labels(summary, recipe))
        for finding in findings:
            prefix = f"Hoja '{finding.sheet_name}': " if finding.sheet_name else ""
            if finding.severity == "error":
                errors.append(prefix + finding.message)
            else:
                warnings.append(prefix + finding.message)
            requires_review = requires_review or finding.requires_review

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
            findings=list({finding.model_dump_json(): finding for finding in findings}.values()),
        )

    def _block_label(self, sheet_name: str, block_id: str | None) -> str:
        if block_id is None:
            return f"la hoja '{sheet_name}'"
        return f"el bloque '{block_id}' de la hoja '{sheet_name}'"

    def _validate_regions(
        self, summary: StructuralSummary, recipe: PreparationRecipe,
    ) -> list[TabularFinding]:
        findings: list[TabularFinding] = []
        sources = {sheet.name: sheet for sheet in summary.sheets}
        expected_roles = {
            "total": "total", "legend": "legend", "derived": "derived",
            "summary": "summary", "repeated_header": "header", "header": "header",
        }
        for block in recipe.sheets:
            source = sources.get(block.sheet_name)
            if source is None:
                continue
            if (
                block.data_start_row > source.rows
                or block.data_end_row > source.rows
                or block.data_start_column > source.columns
                or block.data_end_column > source.columns
                or any(row > source.rows for row in block.header_rows)
            ):
                findings.append(TabularFinding(
                    code="block_out_of_bounds",
                    message="El bloque declara coordenadas fuera de la hoja de origen.",
                    sheet_name=block.sheet_name,
                    block_id=block.block_id,
                    range=self._block_range(block, source),
                    severity="error",
                ))
            if block.role == "unknown":
                findings.append(TabularFinding(
                    code="unresolved_block_role",
                    message="El bloque conserva un rol sin resolver.",
                    sheet_name=block.sheet_name,
                    block_id=block.block_id,
                    range=self._block_range(block, source),
                ))
            for region in source.regions:
                if not self._intersects(block, source, region):
                    continue
                expected_role = expected_roles.get(region.kind)
                if expected_role is not None and block.role != expected_role:
                    if self._candidate_contradicted(source, region, block):
                        continue
                    findings.append(TabularFinding(
                        code="structural_role_conflict",
                        message=(
                            f"El bloque de rol '{block.role}' incluye evidencia candidata de '{region.kind}': "
                            f"{region.reason} Requiere contrastar sus muestras y separar funciones distintas."
                        ),
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        region_id=region.region_id,
                        range=self._block_range(block, source),
                    ))
                elif region.kind == "unknown" and block.role in {"unknown", "observations"}:
                    findings.append(TabularFinding(
                        code="unresolved_structural_region",
                        message="Una región de estructura incierta requiere confirmar su tratamiento antes de usarla como observaciones.",
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        region_id=region.region_id,
                        range=self._block_range(block, source),
                    ))
                elif region.kind == "observations" and block.role in {"derived", "summary"}:
                    findings.append(TabularFinding(
                        code="observations_in_nonobservational_block",
                        message=(
                            f"El bloque de rol '{block.role}' incluye una región compatible con "
                            "observaciones; requiere verificar la separación de magnitudes."
                        ),
                        sheet_name=block.sheet_name,
                        block_id=block.block_id,
                        region_id=region.region_id,
                        range=self._block_range(block, source),
                    ))
        return findings

    def _candidate_contradicted(
        self, source: SheetSummary, region: StructuralRegion, block: SheetRecipe,
    ) -> bool:
        """Contrasta candidatos con muestras; una etiqueta heurística no impone una partición."""
        positions = region.column_indices or list(range(region.start_column, region.end_column + 1))
        values = [
            value for row in region.sample
            if block.data_start_row <= row.row_number <= block.data_end_row
            for column, value in zip(positions, row.values)
            if block.data_start_column <= column <= block.data_end_column and value is not None
        ]
        labels = [value.strip().casefold() for value in values if isinstance(value, str)]
        if region.kind == "total" and values:
            return not any(re.fullmatch(r"(?:grand )?totals?|totales|total general|suma", label) for label in labels)
        if region.kind in {"header", "repeated_header"} and values:
            # Una cabecera candidata numérica no desplaza registros de una tabla sin cabecera.
            return not any(not self._numeric(value) for value in values)
        if region.kind == "legend" and values:
            # La baja densidad sola no convierte mediciones numéricas dispersas en leyenda.
            return all(self._numeric(value) for value in values)
        if region.kind == "derived" and "fórmulas" in region.reason.casefold():
            return source.formula_count == 0
        return False

    @staticmethod
    def _numeric(value: object) -> bool:
        if isinstance(value, Number) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                float(value.strip())
                return True
            except ValueError:
                pass
        return False

    def _intersects(self, block: SheetRecipe, source: SheetSummary, region: StructuralRegion) -> bool:
        return (
            max(block.data_start_row, region.start_row) <= min(block.data_end_row, region.end_row)
            and max(block.data_start_column, region.start_column)
            <= min(block.data_end_column, region.end_column)
        )

    def _block_range(self, block: SheetRecipe, source: SheetSummary) -> str:
        return (
            f"{self._column_label(block.data_start_column)}{block.data_start_row}:"
            f"{self._column_label(block.data_end_column)}{block.data_end_row}"
        )

    def _column_label(self, column: int) -> str:
        label = ""
        while column:
            column, remainder = divmod(column - 1, 26)
            label = chr(65 + remainder) + label
        return label

    def _validate_unit_labels(
        self, summary: StructuralSummary, recipe: PreparationRecipe,
    ) -> list[TabularFinding]:
        findings: list[TabularFinding] = []
        sources = {sheet.name: sheet for sheet in summary.sheets}
        for block in recipe.sheets:
            source = sources.get(block.sheet_name)
            if source is None:
                continue
            for transformation in block.transformations:
                if not isinstance(transformation, RenameColumns):
                    continue
                for original, renamed in transformation.mapping.items():
                    original_units = self._unit_labels(original)
                    renamed_units = self._unit_labels(renamed)
                    if original_units != renamed_units:
                        findings.append(TabularFinding(
                            code="unit_label_changed",
                            message=(
                                f"El renombrado de '{original}' a '{renamed}' cambia una etiqueta "
                                "explícita de unidad o magnitud; requiere verificar su significado."
                            ),
                            sheet_name=block.sheet_name,
                            block_id=block.block_id,
                            range=self._block_range(block, source),
                        ))
        return findings

    def _unit_labels(self, label: str) -> set[str]:
        normalized = "".join(
            character for character in unicodedata.normalize("NFKD", label)
            if not unicodedata.combining(character)
        )
        count_labels = {"celulas", "cells", "colonias", "colonies", "individuos", "individuals"}
        labels = set()
        for match in re.finditer(r"[^\W\d_]+(?:[23²³])?\s*/\s*[^\W\d_]+(?:[23²³])?", normalized):
            numerator, denominator = re.sub(r"\s+", "", match.group()).split("/")
            if sum(character.isalpha() for character in denominator) > 3:
                continue
            if numerator.casefold() in count_labels:
                numerator = numerator.casefold()
            labels.add(f"{numerator}/{denominator}")
        if "%" in normalized or re.search(r"\b(?:percent|percentage|porcentaje)\b", normalized, re.IGNORECASE):
            labels.add("%")
        return labels

    def _has_source_data(self, sheet: SheetSummary) -> bool:
        empty_rows = max(sheet.empty_row_count, len(set(sheet.empty_rows)))
        empty_columns = max(sheet.empty_column_count, len(set(sheet.empty_columns)))
        return sheet.formula_count > 0 or sheet.error_cell_count > 0 or (
            sheet.rows > empty_rows and sheet.columns > empty_columns
        )

    def _validate_sheet(
        self, sheet: SheetExecution, *, coverage_available: bool = False,
    ) -> tuple[list[str], list[str], bool]:
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
        if (
            sheet.source_nonempty_cells is not None
            and sheet.preserved_nonempty_cells is not None
            and sheet.source_nonempty_cells != sheet.preserved_nonempty_cells
        ):
            requires_review = True
            warnings.append(prefix + "La cantidad de celdas no vacías no se conservó durante las transformaciones.")
        if sheet.unresolved_formula_cells or sheet.unresolved_formula_count:
            requires_review = True
            warnings.append(prefix + "Hay fórmulas cuyo resultado no está disponible.")
        if sheet.error_cells or sheet.error_cell_count:
            requires_review = True
            warnings.append(prefix + "La hoja contiene celdas con errores de Excel.")

        if sheet.empty_columns:
            warnings.append(prefix + "La tabla final contiene columnas completamente vacías.")
        if sheet.dropped_empty_rows:
            warnings.append(prefix + f"Se eliminaron {sheet.dropped_empty_rows} filas vacías.")
        if sheet.dropped_empty_columns:
            warnings.append(prefix + f"Se eliminaron {len(sheet.dropped_empty_columns)} columnas vacías.")

        return warnings, errors, requires_review
