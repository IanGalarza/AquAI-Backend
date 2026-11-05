from enum import StrEnum


class DocumentChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    REFERENCES = "references"


class DocumentElementKind(StrEnum):
    TABLE = "table"
    FIGURE = "figure"
