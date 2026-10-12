from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquai.application.contracts.tabular.structure import RecordOverlap


def _validate_source_rows(
    side: str,
    headers: list[int],
    matched_rows: list[int],
    exclusive_rows: list[int],
    record_count: int,
) -> None:
    if headers != sorted(set(headers)):
        raise ValueError(f"{side}_header_rows debe contener filas únicas y ordenadas.")
    rows = matched_rows + exclusive_rows
    if len(rows) != len(set(rows)):
        raise ValueError(f"Cada fila de origen {side} debe aparecer una sola vez entre coincidencias y exclusivas.")
    if set(headers).intersection(rows):
        raise ValueError(f"Las filas de encabezado {side} no pueden contarse como registros.")
    if record_count != len(rows):
        raise ValueError(f"{side}_record_count debe coincidir con las coincidencias más las filas exclusivas.")


def validate_record_overlap(overlap: RecordOverlap) -> None:
    if overlap.left_start_column > overlap.left_end_column or overlap.right_start_column > overlap.right_end_column:
        raise ValueError("Los rangos de columnas de la relación deben estar ordenados.")
    if overlap.left_end_column - overlap.left_start_column != overlap.right_end_column - overlap.right_start_column:
        raise ValueError("La comparación exacta requiere el mismo ancho de columnas en ambos orígenes.")
    _validate_source_rows(
        "left", overlap.left_header_rows, [match.left_row for match in overlap.matches],
        overlap.left_only_rows, overlap.left_record_count,
    )
    _validate_source_rows(
        "right", overlap.right_header_rows, [match.right_row for match in overlap.matches],
        overlap.right_only_rows, overlap.right_record_count,
    )
