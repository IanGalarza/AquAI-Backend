"""Identidad y estados necesarios para actualizar y sondear las fuentes."""
from uuid import UUID
from pydantic import BaseModel
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus


class SourceStateResponse(BaseModel):
    id: UUID
    processing_status: ProcessingStatus
    semantic_index_status: SemanticIndexStatus
