from aquai.application.contracts.tabular.policies import PreparationPolicies, RecordIdentityClarification
from aquai.application.contracts.tabular.recipe import PreparationRecipe
from aquai.application.contracts.tabular.results import ExecutionResult, OriginalProposalState, TabularFinding, ValidationResult
from aquai.application.contracts.tabular.structure import StructuralSummary
from aquai.application.validators.tabular import execution as executionChecks, preparation, structure
from aquai.domain.enums.tabular import EvaluationStatus


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
        empty_names, empty_findings = executionChecks.verified_empty_sheets(summary, recipe, execution)
        findings.extend(empty_findings)
        source_headers, header_findings = preparation.verified_source_headers(summary, execution, preparation_policies)
        findings.extend(header_findings)
        auxiliary_rows, auxiliary_findings = preparation.verified_auxiliary_rows(
            summary, recipe, execution, preparation_policies, source_headers,
        )
        findings.extend(auxiliary_findings)
        findings.extend(preparation.validate_associations(summary, execution, preparation_policies, source_headers))
        findings.extend(preparation.validate_temporal_dates(summary, recipe, execution, preparation_policies, source_headers))
        if recipe.status == "failed":
            errors.append("La receta indica que el archivo no pudo prepararse.")

        source_warnings, source_findings = executionChecks.validate_source_cells(summary, auxiliary_rows)
        warnings.extend(source_warnings)
        findings.extend(source_findings)
        errors.extend(executionChecks.validate_planned_blocks(summary, recipe, execution, empty_names))
        coverage_errors, coverage_findings = executionChecks.validate_coverage(summary, execution)
        errors.extend(coverage_errors)
        findings.extend(coverage_findings)
        findings.extend(executionChecks.validate_omitted_sheets(summary, execution, empty_names))
        sheet_warnings, sheet_errors, sheet_findings = executionChecks.validate_executed_blocks(
            summary, recipe, execution, auxiliary_rows,
        )
        warnings.extend(sheet_warnings)
        errors.extend(sheet_errors)
        findings.extend(sheet_findings)

        findings.extend(structure.validate_regions(
            summary, recipe, empty_names, preparation_policies, auxiliary_rows, source_headers,
        ))
        findings.extend(structure.validate_unit_labels(summary, recipe))
        findings.extend(structure.validate_record_overlaps(summary, execution))
        findings.extend(structure.validate_record_identity_notes(summary, identity_clarifications))
        output_errors, output_findings = executionChecks.validate_output(summary, recipe, execution, empty_names)
        errors.extend(output_errors)
        findings.extend(output_findings)

        for finding in findings:
            prefix = f"Hoja '{finding.sheet_name}': " if finding.sheet_name else ""
            if finding.severity == "error":
                errors.append(prefix + finding.message)
            else:
                warnings.append(prefix + finding.message)
        unique_findings = list({finding.model_dump_json(): finding for finding in findings}.values())
        review_reasons = [finding for finding in unique_findings if finding.requires_review]

        return ValidationResult(
            status=_current_status(errors, review_reasons, warnings),
            warnings=list(dict.fromkeys(warnings)),
            errors=list(dict.fromkeys(errors)),
            findings=unique_findings,
            original_proposal=OriginalProposalState(
                status=recipe.status, requires_human_review=recipe.requires_human_review,
                warnings=list(recipe.warnings),
            ),
            review_reasons=review_reasons,
        )


def _current_status(
    errors: list[str], review_reasons: list[TabularFinding], warnings: list[str],
) -> EvaluationStatus:
    if errors:
        return EvaluationStatus.FAILED
    if review_reasons:
        return EvaluationStatus.REVIEW_REQUIRED
    if warnings:
        return EvaluationStatus.READY_WITH_WARNINGS
    return EvaluationStatus.READY
