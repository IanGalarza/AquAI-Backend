"""Resultado de registrar o reenviar una decisión de revisión."""

from pydantic import Field

from aquai.application.dtos.responses.uploadSource import SourceAcceptedResponse


class SourceDecisionResponse(SourceAcceptedResponse):
    review_id: str | None = None
    revision: int = Field(default=0, ge=0)
    duplicate: bool = False
