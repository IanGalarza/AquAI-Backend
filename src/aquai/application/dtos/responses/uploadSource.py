from uuid import UUID

from pydantic import BaseModel


class SourceAcceptedResponse(BaseModel):
    id: UUID
    duplicate: bool = False
