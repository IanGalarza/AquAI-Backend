from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.application.contracts.tabular.policies import (
        ColumnPreparationPolicy, DateColumnPolicy, ObservationRowPolicy,
    )
    from aquai.application.contracts.tabular.results import TemporalDateTreatment


def validate_column_policy(policy: ColumnPreparationPolicy) -> None:
    if any(column < 1 for column in policy.column_indices):
        raise ValueError("Las columnas originales deben ser positivas.")
    if len(set(policy.column_indices)) != len(policy.column_indices):
        raise ValueError("Las columnas originales no pueden repetirse.")
    if not set(policy.column_indices).issubset(policy.expected_headers):
        raise ValueError("Cada columna de política debe tener un encabezado esperado.")
    if any(not header.strip() for header in policy.expected_headers.values()):
        raise ValueError("Los encabezados esperados no pueden ser vacíos.")


def validate_date_formats(policy: DateColumnPolicy) -> None:
    allowed = {"%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d"}
    if not set(policy.text_formats).issubset(allowed):
        raise ValueError("Se requieren formatos explícitos DMY o ISO, sin inferencia de orden de fecha.")


def validate_related_date_columns(policy: DateColumnPolicy) -> None:
    if any(column < 1 or column not in policy.expected_headers for column in policy.related_date_columns.values()):
        raise ValueError("Las columnas relacionadas necesitan posiciones positivas y encabezados esperados.")


def validate_date_code_formats(policy: DateColumnPolicy) -> None:
    if not set(policy.related_date_code_formats).issubset({"%d%m%Y", "%m%Y"}):
        raise ValueError("Los códigos de fecha requieren formatos explícitos DMY completo o mes/año.")
    if policy.related_date_code_formats and "date_code" not in policy.related_date_columns:
        raise ValueError("Los formatos de código necesitan una columna de código de fecha corroborada.")


def validate_temporal_priority(policy: DateColumnPolicy) -> None:
    if policy.temporal_priority is not None and (
        "date_code" not in policy.related_date_columns or "%d%m%Y" not in policy.related_date_code_formats
    ):
        raise ValueError("La prioridad temporal requiere una columna de código y su formato DDMMAAAA explícito.")


def validate_date_policy(policy: DateColumnPolicy) -> None:
    validate_date_formats(policy)
    validate_related_date_columns(policy)
    validate_date_code_formats(policy)
    validate_temporal_priority(policy)


def validate_observation_policy(policy: ObservationRowPolicy) -> None:
    if any(column < 1 for column in policy.auxiliary_column_indices):
        raise ValueError("Las columnas auxiliares deben ser positivas.")
    if len(set(policy.auxiliary_column_indices)) != len(policy.auxiliary_column_indices):
        raise ValueError("Las columnas auxiliares no pueden repetirse.")
    if not set(policy.auxiliary_column_indices).issubset(policy.expected_headers):
        raise ValueError("Cada columna auxiliar debe tener un encabezado esperado.")


def _validate_original_dates(selection: TemporalDateTreatment) -> None:
    if selection.original_values.keys() != selection.original_evidence.keys() or "date" not in selection.original_values:
        raise ValueError("La selección temporal debe conservar cada valor original con su evidencia de origen.")
    if any(selection.original_values[field] != evidence.original_value
           for field, evidence in selection.original_evidence.items()):
        raise ValueError("Los valores originales deben coincidir con su evidencia temporal.")
    if selection.original_evidence["date"].cell != selection.date_cell:
        raise ValueError("La celda Fecha debe coincidir con su evidencia original.")
    if any(set(discrepancy.fields) - selection.original_values.keys() for discrepancy in selection.discrepancies):
        raise ValueError("Cada discrepancia temporal debe identificar campos originales conservados.")


def _validate_unresolved_date(selection: TemporalDateTreatment) -> None:
    if any(value is not None for value in (selection.normalized_date, selection.source_field, selection.source_cell)):
        raise ValueError("Sin una fecha válida no se puede declarar una selección temporal.")
    if any(discrepancy.resolved_by_policy for discrepancy in selection.discrepancies):
        raise ValueError("Sin fecha válida las discrepancias no quedan resueltas por prioridad temporal.")


def _validate_selected_date(selection: TemporalDateTreatment) -> None:
    if selection.normalized_date is None or selection.source_field is None or selection.source_cell is None:
        raise ValueError("La prioridad temporal debe registrar una fecha válida y su campo y celda de origen.")
    try:
        parsed = date.fromisoformat(selection.normalized_date)
    except ValueError as exc:
        raise ValueError("La fecha temporal elegida debe ser una fecha ISO válida.") from exc
    if parsed.isoformat() != selection.normalized_date:
        raise ValueError("La fecha temporal elegida debe usar la representación ISO AAAA-MM-DD.")
    if (selection.source_field not in selection.original_evidence
            or selection.original_evidence[selection.source_field].cell != selection.source_cell):
        raise ValueError("La fecha elegida debe remitir al campo y la celda originales conservados.")
    if selection.source_field == "date_code" and selection.code_interpretation not in {"exact_8_digits", "numeric_leading_zero"}:
        raise ValueError("Sólo un código de fecha completo y válido puede ser la referencia principal.")
    if any(not discrepancy.resolved_by_policy for discrepancy in selection.discrepancies):
        raise ValueError("Las discrepancias de una selección válida deben quedar registradas como resueltas por la política.")


def validate_temporal_selection(selection: TemporalDateTreatment) -> None:
    _validate_original_dates(selection)
    match selection.resolution:
        case "review_required":
            _validate_unresolved_date(selection)
        case "user_priority":
            _validate_selected_date(selection)
        case _:
            raise ValueError("La resolución temporal no está permitida.")
