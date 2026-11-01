from uuid import UUID

from pydantic import BaseModel


#
# Respuesta de una carga aceptada: identificador de la fuente e indicador de duplicado.
#
class SourceAcceptedResponse(BaseModel):
    id: UUID
    duplicate: bool = False
