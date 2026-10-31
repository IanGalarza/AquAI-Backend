from typing import NamedTuple


class StatusSeed(NamedTuple):
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
    return [{"id": seed.id, "name": seed.name} for seed in MESSAGE_STATUS_SEEDS]
