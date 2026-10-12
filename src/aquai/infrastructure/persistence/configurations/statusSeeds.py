"""Datos iniciales de los catálogos para incluir en sus migraciones autorizadas.

Las definiciones son inmutables. Las funciones devuelven filas nuevas para
usar con ``op.bulk_insert``; importar este módulo no ejecuta SQL.
"""

from typing import NamedTuple


class StatusSeed(NamedTuple):
    """Estado con el ID y el nombre acordados para el catálogo."""

    id: int
    name: str


MESSAGE_STATUS_SEEDS: tuple[StatusSeed, ...] = (
    StatusSeed(1, "pending"),
    StatusSeed(2, "processing"),
    StatusSeed(3, "completed"),
    StatusSeed(4, "failed"),
    StatusSeed(5, "cancelled"),
)


def message_status_seed_rows() -> list[dict[str, int | str]]:
    """Devuelve una copia de las filas iniciales de MessageStatus."""

    return [{"id": seed.id, "name": seed.name} for seed in MESSAGE_STATUS_SEEDS]
