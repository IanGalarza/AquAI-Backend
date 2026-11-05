from functools import lru_cache

from aquai.infrastructure.config.documents import DocumentSettings


#
# Devuelve la configuración del procesamiento documental, cacheada.
#
@lru_cache
def get_document_settings() -> DocumentSettings:
    return DocumentSettings()
