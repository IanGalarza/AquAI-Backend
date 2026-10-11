import re
import unicodedata
from collections import Counter
from numbers import Number

from aquai.application.dtos.tabular import (
    ExecutionResult,
    OriginalProposalState,
    PreparationRecipe,
    RecordIdentityClarification,
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
        *,
        identity_clarifications: tuple[RecordIdentityClarification, ...] = (),
    ) -> ValidationResult:
        # La propuesta archivada es antecedente; el estado vigente depende de los controles actuales.
        warnings = list(summary.warnings)
        errors: list[str] = []
        findings = list(execution.findings)
        requires_review = False
        empty_names, empty_findings = self._verified_empty_sheets(summary, recipe, execution)
        findings.extend(empty_findings)

        if recipe.status == "failed":
            errors.append("La receta indica que el archivo no pudo prepararse.")

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
        treated_blocks = {(block.sheet_name, block.block_id) for block in recipe.sheets
                          if block.sheet_name in empty_names}
        for name, block_id in sorted(planned_blocks - executed_blocks - treated_blocks, key=str):
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
        for name in sorted(omitted_names - empty_names):
            sheet = source_sheets.get(name)
            if sheet is None:
                message = f"Se informó una hoja omitida desconocida: '{name}'."
            else:
                message = (f"Se omitió la hoja '{name}', que contiene datos."
                           if self._has_source_data(sheet) else
                           f"Se omitió la hoja '{name}' sin un tratamiento de vacío verificado.")
            findings.append(TabularFinding(code="unhandled_source_sheet", message=message, sheet_name=name))

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
            sheet_warnings, sheet_errors, sheet_findings = self._validate_sheet(
                sheet, coverage_available=sheet.sheet_name in coverage_names,
            )
            warnings.extend(sheet_warnings)
            errors.extend(sheet_errors)
            findings.extend(sheet_findings)

        findings.extend(self._validate_regions(summary, recipe, empty_names))
        findings.extend(self._validate_unit_labels(summary, recipe))
        findings.extend(self._validate_record_overlaps(summary, execution))
        findings.extend(self._validate_record_identity_notes(summary, identity_clarifications))
        if recipe.status == "review_required" and not recipe.sheets and len(empty_names) != len(source_sheets):
            findings.append(TabularFinding(
                code="unplanned_recipe", message="La receta no propone ninguna hoja para ejecutar; requiere revisión.",
            ))
        elif (not any(sheet.rows_after > 0 and sheet.columns_after for sheet in execution.sheets)
              and len(empty_names) != len(source_sheets)):
            errors.append("La ejecución no produjo ninguna tabla con filas y columnas de datos.")

        for finding in findings:
            prefix = f"Hoja '{finding.sheet_name}': " if finding.sheet_name else ""
            if finding.severity == "error":
                errors.append(prefix + finding.message)
            else:
                warnings.append(prefix + finding.message)
            requires_review = requires_review or finding.requires_review

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
            original_proposal=OriginalProposalState(
                status=recipe.status, requires_human_review=recipe.requires_human_review,
                warnings=list(recipe.warnings),
            ),
            review_reasons=list({finding.model_dump_json(): finding for finding in findings
                                 if finding.requires_review}.values()),
        )

    def _verified_empty_sheets(
        self, summary: StructuralSummary, recipe: PreparationRecipe, execution: ExecutionResult,
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

    def _block_label(self, sheet_name: str, block_id: str | None) -> str:
        if block_id is None:
            return f"la hoja '{sheet_name}'"
        return f"el bloque '{block_id}' de la hoja '{sheet_name}'"

    def _validate_record_overlaps(
        self, summary: StructuralSummary, execution: ExecutionResult,
    ) -> list[TabularFinding]:
        """Expone relaciones de contenido sin anexar ni deduplicar observaciones."""
        findings = []
        source_relations = Counter(relation.model_dump_json() for relation in summary.record_overlaps)
        executed_relations = Counter(relation.model_dump_json() for relation in execution.record_overlaps)
        if source_relations != executed_relations:
            findings.append(TabularFinding(
                code="record_overlap_evidence_mismatch",
                message="La ejecución no conserva las relaciones de registros observadas en la fuente.",
                severity="error",
            ))
        for relation in summary.record_overlaps:
            if not relation.matches:
                if relation.ambiguous_repeated_rows:
                    findings.append(TabularFinding(
                        code="source_repeated_record_content",
                        message=(
                            f"No hay registros compartidos entre '{relation.left_sheet}' y "
                            f"'{relation.right_sheet}', pero se informa contenido repetido dentro de "
                            "al menos uno de los orígenes. Conservar registros y procedencias; "
                            "confirmar identidad antes de anexar o deduplicar."
                        ),
                    ))
                continue
            rows = [match.left_row for match in relation.matches] + relation.left_only_rows
            findings.append(TabularFinding(
                code="source_record_overlap",
                message=(
                    f"Hay {len(relation.matches)} registros de contenido exacto compartido con "
                    f"'{relation.right_sheet}'; {len(relation.left_only_rows)} registros exclusivos "
                    f"de esta hoja y {len(relation.right_only_rows)} de la otra. Conservar ambas "
                    "procedencias sin sumar coincidencias como observaciones independientes."
                    + (" La fuente repite contenido dentro de una hoja: confirmar identidad antes de deduplicar."
                       if relation.ambiguous_repeated_rows else "")
                ),
                sheet_name=relation.left_sheet,
                range=(
                    f"{self._column_label(relation.left_start_column)}{min(rows)}:"
                    f"{self._column_label(relation.left_end_column)}{max(rows)}"
                ),
                requires_review=relation.ambiguous_repeated_rows,
            ))
        return findings

    def _validate_record_identity_notes(
        self, summary: StructuralSummary,
        clarifications: tuple[RecordIdentityClarification, ...] = (),
    ) -> list[TabularFinding]:
        """Las dudas explícitas de la fuente se conservan como evidencia para revisión."""
        findings = []
        seen = set()
        clarified_notes = {
            self._normalized_note(clarification.original_note): clarification
            for clarification in clarifications
        }
        uncertainty = re.compile(
            r"\b(?:no (?:se(?: sabe)?|sabemos) si|posibles?|possible|potential)\b.{0,100}"
            r"\b(?:repetid\w*|duplicad\w*|duplicates?|repeated)\b", re.IGNORECASE,
        )
        for sheet in summary.sheets:
            samples = [(sheet.sample, sheet.column_indices)] + [
                (region.sample, region.column_indices) for region in sheet.regions
            ]
            for sample, positions in samples:
                if not positions:
                    continue
                for row in sample:
                    for column, value in zip(positions, row.values):
                        if not isinstance(value, str):
                            continue
                        normalized = self._normalized_note(value)
                        key = (sheet.name, row.row_number, column)
                        if key in seen or not uncertainty.search(normalized):
                            continue
                        seen.add(key)
                        clarification = clarified_notes.get(normalized)
                        if clarification is not None:
                            findings.append(TabularFinding(
                                code="source_record_identity_clarified",
                                message=(
                                    f"La nota original {value!r} se conserva como antecedente. "
                                    f"Aclarada por {clarification.confirmed_by}: {clarification.statement} "
                                    f"Regla {clarification.rule_id}; fuente: {clarification.source_reference}."
                                ),
                                sheet_name=sheet.name,
                                cell=f"{self._column_label(column)}{row.row_number}",
                                requires_review=False,
                            ))
                            continue
                        findings.append(TabularFinding(
                            code="source_record_identity_uncertainty",
                            message=(
                                "La nota original declara posibles repeticiones. Se conservan los registros "
                                "y sus notas; confirmar su identidad antes de anexar o deduplicar."
                            ),
                            sheet_name=sheet.name,
                            cell=f"{self._column_label(column)}{row.row_number}",
                        ))
        return findings

    @staticmethod
    def _normalized_note(value: str) -> str:
        normalized = "".join(
            character for character in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(character)
        )
        return " ".join(normalized.split())

    def _validate_regions(
        self, summary: StructuralSummary, recipe: PreparationRecipe, empty_names: set[str] | None = None,
    ) -> list[TabularFinding]:
        findings: list[TabularFinding] = []
        sources = {sheet.name: sheet for sheet in summary.sheets}
        expected_roles = {
            "total": "total", "legend": "legend", "derived": "derived",
            "summary": "summary", "repeated_header": "header", "header": "header",
        }
        for block in recipe.sheets:
            if block.sheet_name in (empty_names or set()):
                continue
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
        if sheet.content_evidence is not None:
            return not sheet.content_evidence.is_empty or sheet.formula_count > 0 or sheet.error_cell_count > 0
        empty_rows = max(sheet.empty_row_count, len(set(sheet.empty_rows)))
        empty_columns = max(sheet.empty_column_count, len(set(sheet.empty_columns)))
        return sheet.formula_count > 0 or sheet.error_cell_count > 0 or (
            sheet.rows > empty_rows and sheet.columns > empty_columns
        )

    def _validate_sheet(
        self, sheet: SheetExecution, *, coverage_available: bool = False,
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
        row_difference = sheet.rows_before - sheet.dropped_empty_rows - sheet.rows_after
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
            and sheet.source_nonempty_cells != sheet.preserved_nonempty_cells
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
