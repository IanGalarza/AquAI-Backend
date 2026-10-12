"""Reglas tipadas de preparación corroboradas, independientes de la receta del modelo."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from aquai.application.validators.tabular.policyValidation import (
    validate_column_policy,
    validate_date_policy,
    validate_observation_policy,
)


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
        validate_column_policy(self)
        return self


class DateColumnPolicy(ColumnPreparationPolicy):
    text_formats: tuple[str, ...] = Field(min_length=1)
    related_date_columns: dict[Literal["month", "year", "day", "date_code"], int] = Field(default_factory=dict)
    related_date_code_formats: tuple[str, ...] = ()
    temporal_priority: Literal["date_code_then_date"] | None = None

    @model_validator(mode="after")
    def explicit_date_formats(self):
        validate_date_policy(self)
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
        validate_observation_policy(self)
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


class RecordIdentityClarification(BaseModel):
    """Aclaración humana verificada por el servicio de reglas para una nota concreta."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    original_note: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    sheet_name: str | None = None
    cell: str | None = Field(default=None, pattern=r"^[A-Z]+[1-9][0-9]*$")
