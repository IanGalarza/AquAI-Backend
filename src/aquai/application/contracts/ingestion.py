from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PreparedTable:
    name: str
    source_locator: dict[str, Any]
    columns: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    description: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparationResult:
    tables: list[PreparedTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure_candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class StructureSuggestion:
    region_id: str
    orientation: Literal["rows", "columns"]
    header_start: int
    header_depth: int
    data_start: int


@dataclass
class SemanticDescription:
    description: str
    columns: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
