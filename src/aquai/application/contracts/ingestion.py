"""Tablas preparadas y metadatos; nunca contienen instrucciones ejecutables."""

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class SemanticDescription:
    description: str
    columns: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
