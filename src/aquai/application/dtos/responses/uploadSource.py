"""Identidad y estados públicos de una fuente aceptada."""
from uuid import UUID
from pydantic import BaseModel
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus


class SourceAcceptedResponse(BaseModel):
    id: UUID
    processing_status: ProcessingStatus
    semantic_index_status: SemanticIndexStatus
    duplicate: bool = False
