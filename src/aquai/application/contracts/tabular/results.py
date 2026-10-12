from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_serializer, model_validator

from aquai.application.contracts.tabular.recipe import PreparationRecipe
from aquai.application.contracts.tabular.structure import (
    BlockRole, FormulaSample, RecordOverlap, SampleRow, SheetContentEvidence,
    SourceHeaderEvidence, StructuralSummary,
)
from aquai.application.validators.tabular.policyValidation import validate_temporal_selection
from aquai.domain.enums.tabular import EvaluationStatus


class TabularFinding(BaseModel):
    code: str
    message: str
    sheet_name: str | None = None
    block_id: str | None = None
    cell: str | None = None
    region_id: str | None = None
    range: str | None = None
    requires_review: bool = True
    severity: Literal["warning", "error"] = "warning"


class SheetCoverage(BaseModel):
    sheet_name: str
    nonempty_cells: int = Field(ge=0)
    covered_nonempty_cells: int = Field(ge=0)
    uncovered_nonempty_cells: int = Field(ge=0)
    overlapping_nonempty_cells: int = Field(ge=0)
    uncovered_sample: list[str] = Field(default_factory=list)


class SheetExecution(BaseModel):
    sheet_name: str
    block_id: str | None = None
    role: BlockRole = "observations"
    data_start_row: int | None = None
    data_end_row: int | None = None
    data_start_column: int | None = None
    data_end_column: int | None = None
    source_rows: int
    source_columns: int
    rows_before: int
    rows_after: int
    columns_before: list[str]
    columns_after: list[str]
    excluded_nonempty_rows: int = 0
    suspicious_header_rows: list[int] = Field(default_factory=list)
    dropped_empty_rows: int = 0
    auxiliary_rows_removed: int = 0
    dropped_empty_columns: list[str] = Field(default_factory=list)
    conversion_failures: dict[str, int] = Field(default_factory=dict)
    empty_columns: list[str] = Field(default_factory=list)
    unresolved_formula_cells: list[str] = Field(default_factory=list)
    unresolved_formula_count: int = 0
    formula_count: int = 0
    formula_sample: list[FormulaSample] = Field(default_factory=list)
    error_cell_count: int = 0
    error_cells: list[str] = Field(default_factory=list)
    source_nonempty_cells: int | None = None
    preserved_nonempty_cells: int | None = None
    inferred_types: dict[str, str] = Field(default_factory=dict)
    sample: list[SampleRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    findings: list[TabularFinding] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_system_rows(self, handler):
        value = handler(self)
        if not self.auxiliary_rows_removed:
            value.pop("auxiliary_rows_removed", None)
        return value


class EmptySheetTreatment(BaseModel):
    """Tratamiento determinístico registrado sin editar la propuesta original."""

    sheet_name: str
    evidence: SheetContentEvidence
    skipped_block_ids: list[str] = Field(default_factory=list)
    action: Literal["verified_empty_no_table"] = "verified_empty_no_table"


class CellNormalization(BaseModel):
    sheet_name: str
    block_id: str | None = None
    cell: str
    original_value: JsonValue
    prepared_value: JsonValue
    rule_id: str
    version: str
    origin: str
    operation: Literal["date_text", "number_text", "confirmed_cell"]
    number_format: str | None = None
    source_kind: str | None = None
    raw_value: str | None = None
    excel_epoch: str | None = None


class AuxiliaryFormulaEvidence(BaseModel):
    cell: str
    formula: str
    cached_value: JsonValue = None
    cache_available: Literal[False] = False


class AuxiliaryRowTreatment(BaseModel):
    sheet_name: str
    block_id: str | None = None
    row_number: int = Field(ge=1)
    rule_id: str
    version: str
    origin: str
    inspected_columns: int = Field(ge=1)
    observational_nonempty_cells: Literal[0] = 0
    auxiliary_cells: tuple[AuxiliaryFormulaEvidence, ...] = Field(min_length=1)
    removed_from_observations: bool = True
    action: Literal["auxiliary_formula_only_no_observation"] = "auxiliary_formula_only_no_observation"


class AssociationDiagnostic(BaseModel):
    source_sheet: str
    source_row: int = Field(ge=1)
    source_cell: str
    source_value: JsonValue
    target_sheet: str
    target_rows: tuple[int, ...] = ()
    rule_id: str
    version: str
    origin: str
    status: Literal["matched", "missing", "ambiguous"]
    requires_review: bool = False


class TemporalSourceEvidence(BaseModel):
    """Valor y representación físicos conservados para contrastar la fecha elegida."""

    cell: str = Field(pattern=r"^[A-Z]+[1-9][0-9]*$")
    original_value: JsonValue
    source_kind: str | None = None
    number_format: str | None = None
    raw_value: str | None = None
    excel_epoch: str | None = None


class TemporalDiscrepancy(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    fields: tuple[Literal["date", "date_code", "month", "year", "day"], ...] = Field(min_length=1)
    resolved_by_policy: bool


class TemporalDateTreatment(BaseModel):
    """Selección temporal del sistema, separada de los datos y la receta originales."""

    sheet_name: str = Field(min_length=1)
    block_id: str | None = None
    source_row: int = Field(ge=1)
    date_cell: str = Field(pattern=r"^[A-Z]+[1-9][0-9]*$")
    normalized_date: str | None
    source_field: Literal["date_code", "date"] | None
    source_cell: str | None
    rule_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    original_values: dict[Literal["date", "date_code", "month", "year", "day"], JsonValue]
    original_evidence: dict[Literal["date", "date_code", "month", "year", "day"], TemporalSourceEvidence]
    discrepancies: list[TemporalDiscrepancy] = Field(default_factory=list)
    resolution: Literal["user_priority", "review_required"]
    code_interpretation: Literal["exact_8_digits", "numeric_leading_zero", "partial", "invalid", "missing"]

    @model_validator(mode="after")
    def coherent_temporal_selection(self):
        validate_temporal_selection(self)
        return self


class ExecutionResult(BaseModel):
    sheets: list[SheetExecution] = Field(default_factory=list)
    skipped_sheets: list[str] = Field(default_factory=list)
    coverage: list[SheetCoverage] = Field(default_factory=list)
    findings: list[TabularFinding] = Field(default_factory=list)
    record_overlaps: list[RecordOverlap] = Field(default_factory=list)
    sheet_content_evidence: list[SheetContentEvidence] = Field(default_factory=list)
    empty_sheets: list[EmptySheetTreatment] = Field(default_factory=list)
    normalizations: list[CellNormalization] = Field(default_factory=list)
    auxiliary_rows: list[AuxiliaryRowTreatment] = Field(default_factory=list)
    associations: list[AssociationDiagnostic] = Field(default_factory=list)
    source_headers: list[SourceHeaderEvidence] = Field(default_factory=list)
    temporal_dates: list[TemporalDateTreatment] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_system_treatments(self, handler):
        value = handler(self)
        optional_results = {
            "normalizations": self.normalizations,
            "auxiliary_rows": self.auxiliary_rows,
            "associations": self.associations,
            "source_headers": self.source_headers,
            "temporal_dates": self.temporal_dates,
        }
        for name, results in optional_results.items():
            if not results:
                value.pop(name, None)
        return value


class OriginalProposalState(BaseModel):
    status: Literal["ready", "review_required", "failed"]
    requires_human_review: bool
    warnings: list[str]


class ValidationResult(BaseModel):
    status: EvaluationStatus
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    findings: list[TabularFinding] = Field(default_factory=list)
    original_proposal: OriginalProposalState | None = None
    review_reasons: list[TabularFinding] = Field(default_factory=list)


class LLMResponse(BaseModel):
    model: str
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TabularEvaluation(BaseModel):
    model: str
    structural_summary: StructuralSummary
    recipe: PreparationRecipe | None = None
    llm_response: LLMResponse | None = None
    execution: ExecutionResult = Field(default_factory=ExecutionResult)
    validation: ValidationResult
    applied_rules: dict[str, JsonValue] = Field(default_factory=dict)
