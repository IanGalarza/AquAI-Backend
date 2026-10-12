"""Archivo original leído por el adaptador que invoca la carga."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadSourceRequest:
    filename: str | None
    content: bytes
