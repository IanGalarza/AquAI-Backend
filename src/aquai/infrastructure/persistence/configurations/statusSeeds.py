"""Datos iniciales de los catálogos para incluir en sus migraciones autorizadas.

Las definiciones son inmutables. Las funciones devuelven filas nuevas para
usar con ``op.bulk_insert``; importar este módulo no ejecuta SQL.
"""

from typing import NamedTuple


class StatusSeed(NamedTuple):
    """Estado con el ID y el nombre acordados para el catálogo."""

    id: int
    name: str


PROCESSING_STATUS_SEEDS: tuple[StatusSeed, ...] = (
    StatusSeed(1, "uploaded"),
    StatusSeed(2, "processing"),
    StatusSeed(3, "ready"),
    StatusSeed(4, "ready_with_warnings"),
    StatusSeed(5, "review_required"),
    StatusSeed(6, "failed"),
)

MESSAGE_STATUS_SEEDS: tuple[StatusSeed, ...] = (
    StatusSeed(1, "pending"),
    StatusSeed(2, "processing"),
    StatusSeed(3, "completed"),
    StatusSeed(4, "failed"),
    StatusSeed(5, "cancelled"),
)


VALIDATION_STATUS_SEEDS: tuple[StatusSeed, ...] = (
    StatusSeed(1, "pending"),
    StatusSeed(2, "ready"),
    StatusSeed(3, "ready_with_warnings"),
    StatusSeed(4, "review_required"),
    StatusSeed(5, "failed"),
)


def processing_status_seed_rows() -> list[dict[str, int | str]]:
    """Devuelve una copia de las filas iniciales de ProcessingStatus."""

    return [{"id": seed.id, "name": seed.name} for seed in PROCESSING_STATUS_SEEDS]


def message_status_seed_rows() -> list[dict[str, int | str]]:
    """Devuelve una copia de las filas iniciales de MessageStatus."""

    return [{"id": seed.id, "name": seed.name} for seed in MESSAGE_STATUS_SEEDS]


def validation_status_seed_rows() -> list[dict[str, int | str]]:
    """Devuelve filas nuevas de ValidationStatus para una futura migración."""

    return [{"id": seed.id, "name": seed.name} for seed in VALIDATION_STATUS_SEEDS]
