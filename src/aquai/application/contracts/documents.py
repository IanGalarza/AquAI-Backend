from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedSection:
    position: int
    level: int
    title: str
    path: list[str]
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class PreparedElement:
    position: int
    kind: str
    section_position: int | None = None
    label: str | None = None
    caption: str | None = None
    content: str | None = None
    cells: list[list[str]] | None = None
    page_start: int | None = None
    page_end: int | None = None
    locations: list[dict[str, Any]] = field(default_factory=list)
    header_rows: int = 0


@dataclass
class PreparedChunk:
    position: int
    kind: str
    text: str
    token_count: int
    section_position: int | None = None
    element_position: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    locations: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)


@dataclass
class PreparedDocument:
    text_hash: str
    title: str | None = None
    language: str | None = None
    page_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    sections: list[PreparedSection] = field(default_factory=list)
    elements: list[PreparedElement] = field(default_factory=list)
    chunks: list[PreparedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
