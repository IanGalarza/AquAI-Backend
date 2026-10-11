from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aquai.domain.enums.tabular import EvaluationStatus


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
        if self.left_start_column > self.left_end_column or self.right_start_column > self.right_end_column:
            raise ValueError("Los rangos de columnas de la relación deben estar ordenados.")
        if self.left_end_column - self.left_start_column != self.right_end_column - self.right_start_column:
            raise ValueError("La comparación exacta requiere el mismo ancho de columnas en ambos orígenes.")
        for side in ("left", "right"):
            headers = getattr(self, f"{side}_header_rows")
            if headers != sorted(set(headers)):
                raise ValueError(f"{side}_header_rows debe contener filas únicas y ordenadas.")
            matched_rows = [getattr(match, f"{side}_row") for match in self.matches]
            exclusive_rows = getattr(self, f"{side}_only_rows")
            rows = matched_rows + exclusive_rows
            if len(rows) != len(set(rows)):
                raise ValueError(f"Cada fila de origen {side} debe aparecer una sola vez entre coincidencias y exclusivas.")
            if set(headers).intersection(rows):
                raise ValueError(f"Las filas de encabezado {side} no pueden contarse como registros.")
            if getattr(self, f"{side}_record_count") != len(rows):
                raise ValueError(f"{side}_record_count debe coincidir con las coincidencias más las filas exclusivas.")
        return self


class FormulaSample(BaseModel):
    cell: str
    formula: str
    cached_value: JsonValue = None
    cache_available: bool = False


BlockRole = Literal["observations", "total", "legend", "derived", "summary", "header", "unknown"]


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


def _sheet_recipe_schema(schema: dict) -> None:
    """Proyecta los roles en ramas explícitas para la gramática de generación."""
    properties = schema["properties"]
    roles = properties["role"]["enum"]
    common = {key: schema[key] for key in ("type", "required", "additionalProperties")}
    descriptions = [
        f"{name}: {definition['description']}"
        for name, definition in properties.items() if "description" in definition
    ]
    branches = []
    for role in ("header", "unknown", None):
        branch_properties = {name: dict(definition) for name, definition in properties.items()}
        for definition in branch_properties.values():
            definition.pop("description", None)
        if role is None:
            branch_properties["role"]["enum"] = [value for value in roles if value not in {"header", "unknown"}]
        else:
            branch_properties["role"].pop("enum")
            branch_properties["role"]["const"] = role
            empty_field = "header_rows" if role == "header" else "transformations"
            branch_properties[empty_field]["maxItems"] = 0
        branches.append({**common, "properties": branch_properties})
    for key in ("type", "properties", "required", "additionalProperties"):
        schema.pop(key)
    schema["description"] = " ".join(descriptions)
    schema["oneOf"] = branches


def _preparation_recipe_schema(schema: dict, ref_template: str) -> dict:
    """Comparte las ramas generadas y expresa estados sin condicionales omitidos por Ollama."""
    definitions = schema["$defs"]
    block_schema = definitions["SheetRecipe"]
    block_references = []
    for name, branch in zip(
        ("SheetRecipeHeader", "SheetRecipeUnknown", "SheetRecipeContent"),
        block_schema.pop("oneOf"), strict=True,
    ):
        definitions[name] = branch
        block_references.append({"$ref": ref_template.format(model=name)})
    block_schema["oneOf"] = block_references
    definitions["ResolvedSheetRecipe"] = {"oneOf": [block_references[0], block_references[2]]}
    properties = schema["properties"]
    common = {key: schema[key] for key in ("type", "required", "additionalProperties")}
    branches = []
    for status in properties["status"]["enum"]:
        branch_properties = {name: dict(definition) for name, definition in properties.items()}
        branch_properties["status"].pop("enum")
        branch_properties["status"]["const"] = status
        if status == "ready":
            branch_properties["sheets"]["minItems"] = 1
            branch_properties["warnings"]["maxItems"] = 0
            branch_properties["requires_human_review"]["const"] = False
        else:
            branch_properties["warnings"]["minItems"] = 1
        if status == "review_required":
            branch_properties["requires_human_review"]["const"] = True
        else:
            branch_properties["sheets"]["items"] = {"$ref": ref_template.format(model="ResolvedSheetRecipe")}
        branches.append({**common, "properties": branch_properties})
    for key in ("type", "properties", "required", "additionalProperties"):
        schema.pop(key)
    schema["oneOf"] = branches
    return schema


class SheetRecipe(RecipeContract):
    model_config = ConfigDict(json_schema_extra=_sheet_recipe_schema)

    sheet_name: str = Field(min_length=1, max_length=255, pattern=r"\S")
    block_id: str = Field(
        min_length=1, max_length=100, pattern=r"\S",
        description="Identificador explícito, no vacío y único en toda la receta.",
    )
    role: BlockRole = Field(description=(
        "Función de las celdas dentro del rango de datos. header conserva únicamente "
        "celdas de encabezado como contenido; no representa una copia de los registros."
    ))
    header_rows: list[Annotated[int, Field(ge=1)]] = Field(description=(
        "Filas originales usadas para nombrar columnas del bloque, antes de data_start_row. "
        "No requieren otro bloque header. Para role=header esta lista debe estar vacía."
    ))
    data_start_row: int = Field(ge=1, description="Primera fila de contenido del bloque, original e inclusiva.")
    data_end_row: int = Field(ge=1, description="Última fila de contenido del bloque, original e inclusiva.")
    data_start_column: int = Field(ge=1, description="Primera columna de contenido del bloque, original e inclusiva.")
    data_end_column: int = Field(ge=1, description="Última columna de contenido del bloque, original e inclusiva.")
    transformations: list[Transformation] = Field(max_length=50)

    @model_validator(mode="after")
    def validate_row_ranges(self) -> "SheetRecipe":
        if self.header_rows != sorted(set(self.header_rows)):
            raise ValueError("header_rows debe contener filas únicas en orden ascendente.")
        if self.header_rows and self.header_rows[-1] >= self.data_start_row:
            raise ValueError("Los encabezados deben estar antes de data_start_row.")
        if self.data_end_row < self.data_start_row:
            raise ValueError("data_end_row debe ser mayor o igual a data_start_row.")
        if self.data_end_column < self.data_start_column:
            raise ValueError("data_end_column debe ser mayor o igual a data_start_column.")
        if self.role == "unknown" and self.transformations:
            raise ValueError("Un bloque unknown debe conservarse sin transformaciones.")
        if self.role == "header" and self.header_rows:
            raise ValueError("Un bloque de rol header conserva los encabezados en su rango y debe tener header_rows vacío.")
        return self


class PreparationRecipe(RecipeContract):
    status: Literal["ready", "review_required", "failed"]
    sheets: list[SheetRecipe] = Field(max_length=120)
    warnings: list[Annotated[str, Field(min_length=1, max_length=512, pattern=r"\S")]] = Field(max_length=20)
    requires_human_review: bool

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict:
        schema = super().model_json_schema(*args, **kwargs)
        ref_template = args[1] if len(args) > 1 else kwargs.get("ref_template", "#/$defs/{model}")
        return _preparation_recipe_schema(schema, ref_template)

    @model_validator(mode="after")
    def validate_sheets(self) -> "PreparationRecipe":
        block_ids = [block.block_id for block in self.sheets]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Los block_id deben ser únicos en toda la receta.")
        by_sheet: dict[str, list[SheetRecipe]] = {}
        for block in self.sheets:
            by_sheet.setdefault(block.sheet_name, []).append(block)
        for blocks in by_sheet.values():
            for index, block in enumerate(blocks):
                for other in blocks[index + 1:]:
                    rows_overlap = max(block.data_start_row, other.data_start_row) <= min(
                        block.data_end_row, other.data_end_row,
                    )
                    columns_overlap = max(block.data_start_column, other.data_start_column) <= min(
                        block.data_end_column, other.data_end_column,
                    )
                    if rows_overlap and columns_overlap:
                        raise ValueError(
                            f"Hoja '{block.sheet_name}': Los rangos de datos de los bloques "
                            f"'{block.block_id}' y '{other.block_id}' no pueden solaparse."
                        )
                    if columns_overlap:
                        for referencing, content in ((block, other), (other, block)):
                            conflicting_rows = [
                                row for row in referencing.header_rows
                                if content.data_start_row <= row <= content.data_end_row
                            ]
                            if conflicting_rows and content.role != "header":
                                raise ValueError(
                                    f"Hoja '{block.sheet_name}': El bloque '{referencing.block_id}' "
                                    f"usa como encabezados las filas {conflicting_rows} del rango de "
                                    f"datos no header del bloque '{content.block_id}'."
                                )
        if self.status == "ready" and not self.sheets:
            raise ValueError("Una receta ready debe incluir al menos una hoja.")
        if self.status == "ready" and (self.requires_human_review or self.warnings):
            raise ValueError("Una receta ready debe indicar requires_human_review=false y warnings vacío.")
        if any(block.role == "unknown" for block in self.sheets) and self.status != "review_required":
            raise ValueError("Los bloques unknown requieren status=review_required.")
        if self.status == "review_required" and not self.requires_human_review:
            raise ValueError("Una receta review_required debe indicar requires_human_review=true.")
        if self.status == "review_required" and not self.warnings:
            raise ValueError("Una receta review_required debe explicar su incertidumbre en warnings.")
        if self.status == "failed" and not self.warnings:
            raise ValueError("Una receta failed debe explicar el impedimento identificado en warnings.")
        return self


class RecordIdentityClarification(BaseModel):
    """Aclaración humana verificada por el servicio de reglas para una nota concreta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    original_note: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)


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


class EmptySheetTreatment(BaseModel):
    """Tratamiento determinístico registrado sin editar la propuesta original."""

    sheet_name: str
    evidence: SheetContentEvidence
    skipped_block_ids: list[str] = Field(default_factory=list)
    action: Literal["verified_empty_no_table"] = "verified_empty_no_table"


class ExecutionResult(BaseModel):
    sheets: list[SheetExecution] = Field(default_factory=list)
    skipped_sheets: list[str] = Field(default_factory=list)
    coverage: list[SheetCoverage] = Field(default_factory=list)
    findings: list[TabularFinding] = Field(default_factory=list)
    record_overlaps: list[RecordOverlap] = Field(default_factory=list)
    sheet_content_evidence: list[SheetContentEvidence] = Field(default_factory=list)
    empty_sheets: list[EmptySheetTreatment] = Field(default_factory=list)


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


class EvaluationMetrics(BaseModel):
    inspection_duration_ms: float = 0
    llm_duration_ms: float = 0
    execution_duration_ms: float = 0
    validation_duration_ms: float = 0
    total_duration_ms: float = 0


class ResponseFormatMetrics(BaseModel):
    proposed_status: str | None = None
    json_valid: bool = False
    strict_json: bool = False
    cleanup_required: bool = False
    schema_valid: bool | None = None
    recipe_valid: bool | None = None
    recipe_executable: bool | None = None


class TabularEvaluation(BaseModel):
    model: str
    structural_summary: StructuralSummary
    recipe: PreparationRecipe | None = None
    llm_response: LLMResponse | None = None
    execution: ExecutionResult = Field(default_factory=ExecutionResult)
    validation: ValidationResult
    metrics: EvaluationMetrics
    response_format: ResponseFormatMetrics = Field(default_factory=ResponseFormatMetrics)
    applied_rules: dict[str, JsonValue] = Field(default_factory=dict)
