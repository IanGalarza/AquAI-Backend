from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aquai.domain.enums.tabular import EvaluationStatus


class SampleRow(BaseModel):
    row_number: int
    values: list[JsonValue]


class FormulaSample(BaseModel):
    cell: str
    formula: str


class SheetSummary(BaseModel):
    name: str
    rows: int
    columns: int
    initial_columns: list[str]
    inferred_types: list[str]
    null_counts: list[int]
    sample: list[SampleRow]
    merged_cells: list[str] = Field(default_factory=list)
    formula_count: int = 0
    formula_sample: list[FormulaSample] = Field(default_factory=list)
    error_cell_count: int = 0
    error_cells: list[str] = Field(default_factory=list)
    empty_rows: list[int] = Field(default_factory=list)
    empty_columns: list[int] = Field(default_factory=list)
    empty_row_count: int = 0
    empty_column_count: int = 0
    possible_header_rows: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StructuralSummary(BaseModel):
    file_name: str
    file_type: Literal["csv", "xlsx"]
    file_size_bytes: int
    sheets: list[SheetSummary]
    warnings: list[str] = Field(default_factory=list)
    csv_encoding: str | None = None
    csv_delimiter: str | None = None


class RecipeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DropEmptyRows(RecipeContract):
    operation: Literal["drop_empty_rows"]


class DropEmptyColumns(RecipeContract):
    operation: Literal["drop_empty_columns"]


class StripStrings(RecipeContract):
    operation: Literal["strip_strings"]
    columns: list[str] | None = None


class RenameColumns(RecipeContract):
    operation: Literal["rename_columns"]
    mapping: dict[str, str] = Field(min_length=1)


class ConvertType(RecipeContract):
    operation: Literal["convert_type"]
    columns: list[str] = Field(min_length=1)
    target_type: Literal["number", "integer", "datetime", "boolean"]


Transformation = Annotated[
    DropEmptyRows | DropEmptyColumns | StripStrings | RenameColumns | ConvertType,
    Field(discriminator="operation"),
]


class SheetRecipe(RecipeContract):
    sheet_name: str = Field(min_length=1, max_length=255)
    header_rows: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    data_start_row: int = Field(ge=1)
    data_end_row: int | None = Field(default=None, ge=1)
    transformations: list[Transformation] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def validate_row_ranges(self) -> "SheetRecipe":
        if self.header_rows != sorted(set(self.header_rows)):
            raise ValueError("header_rows debe contener filas únicas en orden ascendente.")
        if self.header_rows and self.header_rows[-1] >= self.data_start_row:
            raise ValueError("Los encabezados deben estar antes de data_start_row.")
        if self.data_end_row is not None and self.data_end_row < self.data_start_row:
            raise ValueError("data_end_row debe ser mayor o igual a data_start_row.")
        return self


class PreparationRecipe(RecipeContract):
    status: Literal["ready", "review_required", "failed"]
    sheets: list[SheetRecipe] = Field(default_factory=list, max_length=30)
    warnings: list[str] = Field(default_factory=list)
    requires_human_review: bool = False

    @model_validator(mode="after")
    def validate_sheets(self) -> "PreparationRecipe":
        names = [sheet.sheet_name for sheet in self.sheets]
        if len(names) != len(set(names)):
            raise ValueError("La receta no puede repetir una hoja.")
        if self.status == "ready" and not self.sheets:
            raise ValueError("Una receta ready debe incluir al menos una hoja.")
        return self


class SheetExecution(BaseModel):
    sheet_name: str
    source_rows: int
    source_columns: int
    rows_before: int
    rows_after: int
    columns_before: list[str]
    columns_after: list[str]
    excluded_nonempty_rows: int = 0
    suspicious_header_rows: list[int] = Field(default_factory=list)
    dropped_empty_rows: int = 0
    dropped_empty_columns: list[str] = Field(default_factory=list)
    conversion_failures: dict[str, int] = Field(default_factory=dict)
    empty_columns: list[str] = Field(default_factory=list)
    unresolved_formula_cells: list[str] = Field(default_factory=list)
    error_cells: list[str] = Field(default_factory=list)
    inferred_types: dict[str, str] = Field(default_factory=dict)
    sample: list[SampleRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    sheets: list[SheetExecution] = Field(default_factory=list)
    skipped_sheets: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    status: EvaluationStatus
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    model: str
    content: str
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    inspection_duration_ms: float = 0
    llm_duration_ms: float = 0
    execution_duration_ms: float = 0
    validation_duration_ms: float = 0
    total_duration_ms: float = 0


class TabularEvaluation(BaseModel):
    model: str
    structural_summary: StructuralSummary
    recipe: PreparationRecipe | None = None
    llm_response: LLMResponse | None = None
    execution: ExecutionResult = Field(default_factory=ExecutionResult)
    validation: ValidationResult
    metrics: EvaluationMetrics
