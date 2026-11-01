from uuid import UUID

from pydantic import BaseModel

from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus


#
# Estado básico de una fuente: identificador, procesamiento e indexación semántica.
#
class SourceStateResponse(BaseModel):
    id: UUID
    processing_status: ProcessingStatus
    semantic_index_status: SemanticIndexStatus
