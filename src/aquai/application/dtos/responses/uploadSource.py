"""Identidad y estados públicos de una fuente aceptada."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel


ProcessingStatus = Literal[
    "uploaded", "processing", "ready", "ready_with_warnings", "review_required", "failed"
]
ValidationStatus = Literal["pending", "ready", "ready_with_warnings", "review_required", "failed"]


class SourceAcceptedResponse(BaseModel):
    id: UUID
    processing_status: ProcessingStatus
    validation_status: ValidationStatus
