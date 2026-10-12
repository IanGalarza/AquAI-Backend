from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

from aquai.application.validators.tabular.structureValidation import validate_record_overlap


BlockRole = Literal["observations", "total", "legend", "derived", "summary", "header", "unknown"]


class SampleRow(BaseModel):
    row_number: int
    values: list[JsonValue]


class SourceRowMatch(BaseModel):
    left_row: int = Field(ge=1, strict=True)
    right_row: int = Field(ge=1, strict=True)


class RecordOverlap(BaseModel):
    left_sheet: str = Field(min_length=1)
    right_sheet: str = Field(min_length=1)
    left_header_rows: list[Annotated[int, Field(ge=1, strict=True)]]
    right_header_rows: list[Annotated[int, Field(ge=1, strict=True)]]
    left_start_column: int = Field(ge=1, strict=True)
    left_end_column: int = Field(ge=1, strict=True)
    right_start_column: int = Field(ge=1, strict=True)
    right_end_column: int = Field(ge=1, strict=True)
    left_record_count: int = Field(ge=0, strict=True)
    right_record_count: int = Field(ge=0, strict=True)
    matches: list[SourceRowMatch]
    left_only_rows: list[Annotated[int, Field(ge=1, strict=True)]]
    right_only_rows: list[Annotated[int, Field(ge=1, strict=True)]]
    comparison: Literal["exact_values_headers_and_formulas"] = "exact_values_headers_and_formulas"
    policy: Literal["preserve_sources_do_not_add_matches"] = "preserve_sources_do_not_add_matches"
    ambiguous_repeated_rows: bool = False

    @model_validator(mode="after")
    def coherent_source_rows(self) -> "RecordOverlap":
        validate_record_overlap(self)
        return self


class FormulaSample(BaseModel):
    cell: str
    formula: str
    cached_value: JsonValue = None
    cache_available: bool = False


class StructuralRegion(BaseModel):
    region_id: str
    start_row: int = Field(ge=1)
    end_row: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_column: int = Field(ge=1)
    kind: Literal["observations", "total", "legend", "derived", "summary", "header", "repeated_header", "unknown"]
    reason: str
    nonempty_cells: int = Field(default=0, ge=0)
    sample: list[SampleRow] = Field(default_factory=list)
    column_indices: list[int] = Field(default_factory=list)


class SheetContentEvidence(BaseModel):
    """Conteo completo de contenido, independiente de muestras y del formato."""

    sheet_name: str
    inspected_cells: int = Field(ge=0)
    value_cells: int = Field(ge=0)
    formula_cells: int = Field(ge=0)
    verification: Literal["full_sheet_values_and_formulas"] = "full_sheet_values_and_formulas"

    @model_validator(mode="after")
    def coherent_content_counts(self) -> "SheetContentEvidence":
        if max(self.value_cells, self.formula_cells) > self.inspected_cells:
            raise ValueError("Los conteos de contenido no pueden superar las celdas examinadas.")
        return self

    @property
    def is_empty(self) -> bool:
        return self.value_cells == 0 and self.formula_cells == 0


class SheetSummary(BaseModel):
    name: str
    rows: int
    columns: int
    initial_columns: list[str]
    inferred_types: list[str]
    null_counts: list[int]
    sample: list[SampleRow]
    column_indices: list[int] = Field(default_factory=list)
    merged_cells: list[str] = Field(default_factory=list)
    formula_count: int = 0
    missing_formula_cache_count: int = Field(default=0, ge=0)
    formula_sample: list[FormulaSample] = Field(default_factory=list)
    error_cell_count: int = 0
    error_cells: list[str] = Field(default_factory=list)
    empty_rows: list[int] = Field(default_factory=list)
    empty_columns: list[int] = Field(default_factory=list)
    empty_row_count: int = 0
    empty_column_count: int = 0
    possible_header_rows: list[int] = Field(default_factory=list)
    regions: list[StructuralRegion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    content_evidence: SheetContentEvidence | None = None


class StructuralSummary(BaseModel):
    file_name: str
    file_type: Literal["csv", "xlsx"]
    file_size_bytes: int
    sheets: list[SheetSummary]
    record_overlaps: list[RecordOverlap] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    csv_encoding: str | None = None
    csv_delimiter: str | None = None


class SourceHeaderEvidence(BaseModel):
    """Fila original completa inspeccionada por el sistema, independiente de las muestras."""

    sheet_name: str
    row_number: int = Field(ge=1)
    inspected_columns: int = Field(ge=1)
    values: dict[int, JsonValue]
    formula_columns: tuple[int, ...] = ()
    verification: Literal["full_source_header_row"] = "full_source_header_row"
