"""Políticas de preparación corroboradas del catálogo, ajenas a la receta de la LLM."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class PreparationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    origin: str = Field(min_length=1)


class ColumnPreparationPolicy(PreparationPolicy):
    sheet_name: str = Field(min_length=1)
    column_indices: tuple[int, ...] = Field(min_length=1)
    expected_headers: dict[int, str]
    header_row: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def guarded_columns(self):
        if any(column < 1 for column in self.column_indices):
            raise ValueError("Las columnas originales deben ser positivas.")
        if len(set(self.column_indices)) != len(self.column_indices):
            raise ValueError("Las columnas originales no pueden repetirse.")
        if not set(self.column_indices).issubset(self.expected_headers):
            raise ValueError("Cada columna de política debe tener un encabezado esperado.")
        if any(not header.strip() for header in self.expected_headers.values()):
            raise ValueError("Los encabezados esperados no pueden ser vacíos.")
        return self


class DateColumnPolicy(ColumnPreparationPolicy):
    text_formats: tuple[str, ...] = Field(min_length=1)
    related_date_columns: dict[Literal["month", "year", "day", "date_code"], int] = Field(default_factory=dict)
    related_date_code_formats: tuple[str, ...] = ()
    temporal_priority: Literal["date_code_then_date"] | None = None

    @model_validator(mode="after")
    def explicit_date_formats(self):
        allowed = {"%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"}
        if not set(self.text_formats).issubset(allowed):
            raise ValueError("Se requieren formatos explícitos DMY o ISO, sin inferencia de orden de fecha.")
        if any(column < 1 or column not in self.expected_headers
               for column in self.related_date_columns.values()):
            raise ValueError("Las columnas relacionadas necesitan posiciones positivas y encabezados esperados.")
        if not set(self.related_date_code_formats).issubset({"%d%m%Y", "%m%Y"}):
            raise ValueError("Los códigos de fecha requieren formatos explícitos DMY completo o mes/año.")
        if self.related_date_code_formats and "date_code" not in self.related_date_columns:
            raise ValueError("Los formatos de código necesitan una columna de código de fecha corroborada.")
        if self.temporal_priority is not None and (
            "date_code" not in self.related_date_columns or "%d%m%Y" not in self.related_date_code_formats
        ):
            raise ValueError("La prioridad temporal requiere una columna de código y su formato DDMMAAAA explícito.")
        return self


class NumberColumnPolicy(ColumnPreparationPolicy):
    decimal_separators: tuple[Literal[",", "."], ...] = (",", ".")


class ConfirmedCellPolicy(PreparationPolicy):
    sheet_name: str = Field(min_length=1)
    cell: str = Field(pattern=r"^[A-Z]+[1-9][0-9]*$")
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_source_value: JsonValue
    expected_number_format: str
    expected_source_kind: str = Field(min_length=1)
    expected_raw_value: str | None = None
    expected_header: str = Field(min_length=1)
    header_row: int = Field(default=1, ge=1)
    prepared_value: JsonValue


class ObservationRowPolicy(PreparationPolicy):
    sheet_name: str = Field(min_length=1)
    auxiliary_column_indices: tuple[int, ...] = Field(min_length=1)
    expected_headers: dict[int, str]
    header_row: int = Field(default=1, ge=1)
    require_all_other_cells_empty: Literal[True] = True

    @model_validator(mode="after")
    def guarded_auxiliary_columns(self):
        if any(column < 1 for column in self.auxiliary_column_indices):
            raise ValueError("Las columnas auxiliares deben ser positivas.")
        if len(set(self.auxiliary_column_indices)) != len(self.auxiliary_column_indices):
            raise ValueError("Las columnas auxiliares no pueden repetirse.")
        if not set(self.auxiliary_column_indices).issubset(self.expected_headers):
            raise ValueError("Cada columna auxiliar debe tener un encabezado esperado.")
        return self


class AssociationPolicy(PreparationPolicy):
    source_sheet: str = Field(min_length=1)
    source_column: int = Field(ge=1)
    expected_source_header: str = Field(min_length=1)
    target_sheet: str = Field(min_length=1)
    target_column: int = Field(ge=1)
    expected_target_header: str = Field(min_length=1)
    source_start_row: int = Field(default=2, ge=1)
    target_start_row: int = Field(default=2, ge=1)
    header_row: int = Field(default=1, ge=1)
    optional: Literal[True] = True


class PreparationPolicies(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    date_columns: tuple[DateColumnPolicy, ...] = ()
    number_columns: tuple[NumberColumnPolicy, ...] = ()
    confirmed_cells: tuple[ConfirmedCellPolicy, ...] = ()
    observation_rows: tuple[ObservationRowPolicy, ...] = ()
    associations: tuple[AssociationPolicy, ...] = ()

    @property
    def enabled(self) -> bool:
        return any((self.date_columns, self.number_columns, self.confirmed_cells,
                    self.observation_rows, self.associations))


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


class SourceHeaderEvidence(BaseModel):
    """Fila original completa inspeccionada por el sistema, independiente de las muestras."""

    sheet_name: str
    row_number: int = Field(ge=1)
    inspected_columns: int = Field(ge=1)
    values: dict[int, JsonValue]
    formula_columns: tuple[int, ...] = ()
    verification: Literal["full_source_header_row"] = "full_source_header_row"


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
        if self.original_values.keys() != self.original_evidence.keys() or "date" not in self.original_values:
            raise ValueError("La selección temporal debe conservar cada valor original con su evidencia de origen.")
        if any(self.original_values[field] != evidence.original_value
               for field, evidence in self.original_evidence.items()):
            raise ValueError("Los valores originales deben coincidir con su evidencia temporal.")
        if self.original_evidence["date"].cell != self.date_cell:
            raise ValueError("La celda Fecha debe coincidir con su evidencia original.")
        if any(set(discrepancy.fields) - self.original_values.keys() for discrepancy in self.discrepancies):
            raise ValueError("Cada discrepancia temporal debe identificar campos originales conservados.")
        if self.resolution == "review_required":
            if any(value is not None for value in (self.normalized_date, self.source_field, self.source_cell)):
                raise ValueError("Sin una fecha válida no se puede declarar una selección temporal.")
            if any(discrepancy.resolved_by_policy for discrepancy in self.discrepancies):
                raise ValueError("Sin fecha válida las discrepancias no quedan resueltas por prioridad temporal.")
            return self
        if self.normalized_date is None or self.source_field is None or self.source_cell is None:
            raise ValueError("La prioridad temporal debe registrar una fecha válida y su campo y celda de origen.")
        try:
            parsed = date.fromisoformat(self.normalized_date)
        except ValueError as exc:
            raise ValueError("La fecha temporal elegida debe ser una fecha ISO válida.") from exc
        if parsed.isoformat() != self.normalized_date:
            raise ValueError("La fecha temporal elegida debe usar la representación ISO AAAA-MM-DD.")
        if self.source_field not in self.original_evidence or self.original_evidence[self.source_field].cell != self.source_cell:
            raise ValueError("La fecha elegida debe remitir al campo y la celda originales conservados.")
        if self.source_field == "date_code" and self.code_interpretation not in {"exact_8_digits", "numeric_leading_zero"}:
            raise ValueError("Sólo un código de fecha completo y válido puede ser la referencia principal.")
        if any(not discrepancy.resolved_by_policy for discrepancy in self.discrepancies):
            raise ValueError("Las discrepancias de una selección válida deben quedar registradas como resueltas por la política.")
        return self
