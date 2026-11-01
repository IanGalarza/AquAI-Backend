from dataclasses import dataclass


#
# Solicitud de carga de una fuente: nombre y contenido del archivo.
#
@dataclass(frozen=True)
class UploadSourceRequest:
    filename: str | None
    content: bytes
