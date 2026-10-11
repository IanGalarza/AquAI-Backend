import re
import unicodedata
from collections import Counter
from datetime import date, datetime
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
from aquai.application.dtos.tabular_policies import PreparationPolicies


class TabularValidationService:
    def validate(
        self,
        summary: StructuralSummary,
        recipe: PreparationRecipe,
        execution: ExecutionResult,
        *,
        identity_clarifications: tuple[RecordIdentityClarification, ...] = (),
        preparation_policies: PreparationPolicies | None = None,
    ) -> ValidationResult:
        # La propuesta archivada es antecedente; el estado vigente depende de los controles actuales.
        warnings = list(summary.warnings)
        errors: list[str] = []
        findings = list(execution.findings)
        requires_review = False
        empty_names, empty_findings = self._verified_empty_sheets(summary, recipe, execution)
        findings.extend(empty_findings)
        source_headers, header_findings = self._verified_source_headers(summary, execution, preparation_policies)
        findings.extend(header_findings)
        auxiliary_rows, auxiliary_findings = self._verified_auxiliary_rows(
            summary, recipe, execution, preparation_policies, source_headers,
        )
        findings.extend(auxiliary_findings)
        findings.extend(self._validate_associations(summary, execution, preparation_policies, source_headers))
        findings.extend(self._validate_temporal_dates(summary, recipe, execution, preparation_policies, source_headers))
        auxiliary_cells_by_sheet = Counter(
            treatment.sheet_name for treatments in auxiliary_rows.values()
            for treatment in treatments for _ in treatment.auxiliary_cells
        )

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
            treatments = auxiliary_rows.get((sheet.sheet_name, sheet.block_id), [])
            removed_treatments = [treatment for treatment in treatments if treatment.removed_from_observations]
            sheet_warnings, sheet_errors, sheet_findings = self._validate_sheet(
                sheet, coverage_available=sheet.sheet_name in coverage_names,
                auxiliary_rows=len(removed_treatments),
                auxiliary_formula_cells=sum(len(treatment.auxiliary_cells) for treatment in removed_treatments),
            )
            warnings.extend(sheet_warnings)
            errors.extend(sheet_errors)
            findings.extend(sheet_findings)

        findings.extend(self._validate_regions(summary, recipe, empty_names, preparation_policies, auxiliary_rows, source_headers))
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

    def _validate_temporal_dates(self, summary, recipe, execution, policies, source_headers):
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
            position = self._cell_position(treatment.date_cell)
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
                and self._policy_headers_match(source, policy.expected_headers, source_headers, policy.header_row)
                and key not in seen and treatment.original_values.keys() == fields.keys()
                and all(self._cell_position(treatment.original_evidence[field].cell)
                        == (treatment.source_row, column) for field, column in fields.items())
                and self._temporal_selection_matches_originals(treatment, policy)
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
            unresolved_reported = any(
                finding.code == "temporal_date_unresolved" and finding.requires_review
                and (finding.sheet_name, finding.block_id, finding.cell)
                == (treatment.sheet_name, treatment.block_id, treatment.date_cell)
                for finding in reported
            )
            if treatment.resolution == "review_required" and not unresolved_reported:
                findings.append(TabularFinding(
                    code="temporal_date_unresolved", sheet_name=treatment.sheet_name,
                    block_id=treatment.block_id, cell=treatment.date_cell,
                    message=(f"Fila {treatment.source_row}: ni Código Fecha ni Fecha permiten determinar una fecha válida; "
                             "Mes y Año se conservan únicamente como contraste. Revisar los valores originales."),
                ))
            elif treatment.resolution == "user_priority" and treatment.discrepancies and not any(
                finding.code == "temporal_priority_discrepancies_resolved" and not finding.requires_review
                and (finding.sheet_name, finding.block_id, finding.cell)
                == (treatment.sheet_name, treatment.block_id, treatment.date_cell)
                for finding in reported
            ):
                findings.append(TabularFinding(
                    code="temporal_priority_discrepancies_resolved", sheet_name=treatment.sheet_name,
                    block_id=treatment.block_id, cell=treatment.date_cell, requires_review=False,
                    message=(f"Fecha {treatment.normalized_date} seleccionada desde {treatment.source_cell} por "
                             f"{treatment.rule_id} v{treatment.version}: "
                             + "; ".join(discrepancy.message for discrepancy in treatment.discrepancies)
                             + " Se conservan los originales; la política no constituye una confirmación científica."),
                ))
        for policy in temporal_policies:
            for sheet in execution.sheets:
                if sheet.sheet_name != policy.sheet_name or sheet.role != "observations":
                    continue
                plan = plans.get((sheet.sheet_name, sheet.block_id))
                if plan is None:
                    continue
                for column in policy.column_indices:
                    if plan.data_start_column <= column <= plan.data_end_column:
                        count = verified[(sheet.sheet_name, sheet.block_id, column)]
                        if count != sheet.rows_after:
                            findings.append(TabularFinding(
                                code="temporal_date_evidence_missing", sheet_name=sheet.sheet_name,
                                block_id=sheet.block_id, severity="error",
                                message=(f"La selección temporal conserva {count} registros con evidencia válida, "
                                         f"pero el bloque tiene {sheet.rows_after} observaciones. No se resuelve la revisión sin esa procedencia."),
                            ))
        return findings

    @staticmethod
    def _temporal_selection_matches_originals(treatment, policy):
        value = treatment.original_values.get("date_code")
        numeric = isinstance(value, Number) and not isinstance(value, bool)
        if numeric:
            try:
                text = str(int(value)) if value == int(value) and value >= 0 else ""
            except (ValueError, OverflowError, TypeError):
                text = ""
            code_evidence = treatment.original_evidence.get("date_code")
            if (len(text) == 7 and code_evidence is not None
                    and code_evidence.source_kind == "native_number"
                    and code_evidence.number_format == "00000000"):
                text = "0" + text
        else:
            text = value.strip() if isinstance(value, str) else ""
        chosen_code = None
        if re.fullmatch(r"[0-9]{8}", text):
            try:
                chosen_code = date(int(text[4:]), int(text[2:4]), int(text[:2])).isoformat()
            except ValueError:
                pass
        chosen_date = None
        original_date = treatment.original_values.get("date")
        if isinstance(original_date, str):
            text = original_date.strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)?", text):
                try:
                    chosen_date = datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    pass
            if chosen_date is None:
                for explicit_format in policy.text_formats:
                    try:
                        chosen_date = datetime.strptime(text, explicit_format).date().isoformat()
                    except ValueError:
                        continue
                    break
        chosen = chosen_code or chosen_date
        return (
            treatment.resolution == "review_required" and chosen is None
            or treatment.resolution == "user_priority" and chosen is not None
            and treatment.normalized_date == chosen
            and treatment.source_field == ("date_code" if chosen_code else "date")
            and (not chosen_code or treatment.code_interpretation == (
                "numeric_leading_zero" if numeric and len(str(int(value))) == 7 else "exact_8_digits"
            ))
        )

    def _verified_source_headers(self, summary, execution, policies):
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
                if (position := self._cell_position(cell.cell)) is not None and position[0] == proof.row_number
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
                    self._summary_header_value(proof.values.get(column), column) == name
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

    @staticmethod
    def _summary_header_value(value, column):
        if value is None or isinstance(value, str) and not value.strip():
            return f"column_{column}"
        return value[:200] if isinstance(value, str) else str(value)

    def _verified_auxiliary_rows(self, summary, recipe, execution, policies, source_headers):
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
            coordinates = [self._cell_position(cell.cell) for cell in treatment.auxiliary_cells]
            sampled_formulas = {cell.cell: cell for cell in source.formula_sample} if source else {}
            key = (treatment.sheet_name, treatment.row_number)
            valid = (
                source is not None and block is not None and policy is not None
                and block.role in {"observations", "derived", "legend", "unknown"}
                and treatment.removed_from_observations == (block.role == "observations")
                and self._policy_headers_match(source, policy.expected_headers, source_headers, policy.header_row)
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

    def _validate_associations(self, summary, execution, policies, source_headers):
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
            position = self._cell_position(diagnostic.source_cell)
            key = (diagnostic.rule_id, diagnostic.source_sheet, diagnostic.source_row, diagnostic.source_cell)
            expected_status = ("missing" if not diagnostic.target_rows else
                               "matched" if len(diagnostic.target_rows) == 1 else "ambiguous")
            valid = (
                source is not None and target is not None and policy is not None
                and position == (diagnostic.source_row, policy.source_column)
                and policy.source_start_row <= diagnostic.source_row <= source.rows
                and self._policy_headers_match(source, {policy.source_column: policy.expected_source_header}, source_headers, policy.header_row)
                and self._policy_headers_match(target, {policy.target_column: policy.expected_target_header}, source_headers, policy.header_row)
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

    @staticmethod
    def _cell_position(cell):
        match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell)
        if match is None:
            return None
        column = 0
        for character in match[1]:
            column = column * 26 + ord(character) - ord("A") + 1
        return int(match[2]), column

    @staticmethod
    def _policy_headers_match(source, expected_headers, source_headers=None, header_row=1):
        if source_headers is not None:
            proof = source_headers.get((source.name, header_row))
            return proof is not None and all(
                column not in proof.formula_columns and proof.values.get(column) == expected
                for column, expected in expected_headers.items()
            )
        positions = source.column_indices or list(range(1, len(source.initial_columns) + 1))
        actual = dict(zip(positions, source.initial_columns))
        return all(actual.get(column) == expected for column, expected in expected_headers.items())

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
        preparation_policies: PreparationPolicies | None = None,
        auxiliary_rows: dict | None = None,
        source_headers: dict | None = None,
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
                    if (region.kind == "derived" and block.role == "legend"
                        and self._derived_auxiliary_legend(region, block, (auxiliary_rows or {}).get(
                            (block.sheet_name, block.block_id), [],
                        ))):
                        continue
                    if region.kind == "derived" and block.role == "observations" and preparation_policies:
                        identity_columns = {
                            column for policy in preparation_policies.observation_rows
                            if policy.sheet_name == source.name and self._policy_headers_match(
                                source, policy.expected_headers, source_headers, policy.header_row,
                            )
                            for column in policy.auxiliary_column_indices
                        }
                        if (set(range(region.start_column, region.end_column + 1)).issubset(identity_columns)
                            and identity_columns.issubset(range(block.data_start_column, block.data_end_column + 1))
                            and set(range(block.data_start_column, block.data_end_column + 1)) - identity_columns):
                            continue
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

    def _derived_auxiliary_legend(self, region, block, treatments):
        """Sólo una intersección enteramente probada como fórmula auxiliar puede ser leyenda."""
        first_row, last_row = max(block.data_start_row, region.start_row), min(block.data_end_row, region.end_row)
        first_column = max(block.data_start_column, region.start_column)
        last_column = min(block.data_end_column, region.end_column)
        proofs = {treatment.row_number: treatment for treatment in treatments
                  if not treatment.removed_from_observations}
        for row in range(first_row, last_row + 1):
            treatment = proofs.get(row)
            if treatment is None:
                return False
            positions = {self._cell_position(cell.cell) for cell in treatment.auxiliary_cells}
            if not all((row, column) in positions for column in range(first_column, last_column + 1)):
                return False
        return first_row <= last_row and first_column <= last_column

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
