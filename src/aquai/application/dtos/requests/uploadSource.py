from dataclasses import dataclass


@dataclass(frozen=True)
class UploadSourceRequest:
    filename: str | None
    content: bytes
