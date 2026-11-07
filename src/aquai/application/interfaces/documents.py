from typing import Any, Protocol
from uuid import UUID

from aquai.application.contracts.documents import PreparedDocument
from aquai.application.contracts.retrieval import (
    EvidenceUsage,
    ReferenceExtraction,
    SearchTermsDraft,
)


class DocumentProcessor(Protocol):
    #
    # Valida el archivo cargado y devuelve su nombre seguro.
    #
    def validate_upload(self, filename: str | None, content: bytes) -> str: ...

    #
    # Extrae la estructura y los fragmentos del documento.
    #
    async def prepare(self, filename: str, content: bytes) -> PreparedDocument: ...


class DocumentCommands(Protocol):
    #
    # Reemplaza el contenido extraído si la fuente sigue con ese original e indica si lo guardó.
    #
    def save_prepared(
        self, source_id: UUID, document: PreparedDocument, report: dict, content_hash: str
    ) -> bool: ...

    #
    # Reemplaza el archivo original de una fuente e indica si el contenido cambió.
    #
    def replace_file(self, source_id: UUID, filename: str, content: bytes) -> bool: ...

    #
    # Verifica que la fuente se pueda eliminar o reemplazar sin perder referencias.
    #
    def ensure_removable(self, source_id: UUID) -> None: ...

    #
    # Elimina una fuente documental con todo su contenido extraído.
    #
    def delete_source(self, source_id: UUID) -> None: ...


class DocumentQueries(Protocol):
    #
    # Devuelve el resumen del documento extraído de una fuente, si existe.
    #
    def summary(self, source_id: UUID) -> dict[str, Any] | None: ...

    #
    # Devuelve el documento vigente de una fuente con los fragmentos que deben indexarse.
    #
    def indexable(self, source_id: UUID) -> dict[str, Any] | None: ...

    #
    # Devuelve los fragmentos vigentes de fuentes activas con su procedencia completa.
    #
    def verified_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, dict[str, Any]]: ...

    #
    # Devuelve una huella del contenido consultable, que cambia al cargar o desactivar fuentes.
    #
    def searchable_version(self) -> tuple: ...

    #
    # Devuelve el texto buscable de los fragmentos vigentes, con su sección y leyenda.
    #
    def searchable_chunks(self) -> list[dict[str, Any]]: ...


class DocumentIndex(Protocol):
    #
    # Indexa los fragmentos de una fuente documental y elimina los obsoletos.
    #
    async def index_source(self, source: dict) -> None: ...

    #
    # Elimina del índice todos los fragmentos de una fuente.
    #
    async def delete_source(self, source_id: UUID) -> None: ...

    #
    # Busca los fragmentos más cercanos a una consulta y los verifica en PostgreSQL.
    #
    async def search(
        self,
        query: str,
        limit: int = 5,
        *,
        kinds: tuple[str, ...] | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[dict[str, Any]]: ...


class DocumentLexicalIndex(Protocol):
    #
    # Devuelve los identificadores de los fragmentos que mejor coinciden con los términos.
    #
    async def search(
        self, query: str, limit: int, *, document_ids: list[UUID] | None = None
    ) -> list[UUID]: ...


class DocumentEvidenceProvider(Protocol):
    model: str

    #
    # Convierte pedidos en búsquedas autónomas en español y en inglés.
    #
    async def search_terms(self, context: dict) -> tuple[SearchTermsDraft, EvidenceUsage]: ...

    #
    # Ordena los fragmentos por utilidad y descarta los que no sirven para la consulta.
    #
    async def select(self, context: dict) -> tuple[list[int], EvidenceUsage]: ...

    #
    # Propone valores de referencia de los fragmentos para las variables indicadas.
    #
    async def references(
        self, context: dict, documents: list[str]
    ) -> tuple[ReferenceExtraction, EvidenceUsage]: ...
