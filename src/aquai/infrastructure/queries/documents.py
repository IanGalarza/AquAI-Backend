from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from aquai.application.Rules.documents import chunk_doubts
from aquai.domain.enums.documentContent import DocumentChunkKind, DocumentElementKind
from aquai.domain.enums.sourceStatus import ProcessingStatus
from aquai.domain.models import (
    Document,
    DocumentChunk,
    DocumentElement,
    DocumentSection,
    Source,
)
from aquai.infrastructure.persistence.dbContext import DbContext

_READY_STATUSES = (ProcessingStatus.READY, ProcessingStatus.READY_WITH_WARNINGS)
_SEARCHABLE_KINDS = (DocumentChunkKind.TEXT, DocumentChunkKind.TABLE, DocumentChunkKind.FIGURE)


class SqlAlchemyDocumentQueries:
    #
    # Guarda el contexto de base de datos que usa el repositorio.
    #
    def __init__(self, context: DbContext) -> None:
        self._context = context

    #
    # Devuelve el resumen del documento extraído de una fuente, si existe.
    #
    def summary(self, source_id: UUID) -> dict[str, Any] | None:
        with self._context.session() as session:
            document = session.execute(
                select(
                    Document.id,
                    Document.title,
                    Document.language,
                    Document.page_count,
                    Document.duplicate_of_source_id,
                ).where(Document.source_id == source_id)
            ).mappings().first()
            if document is None:
                return None

            return {**document, **self._counts(session, document["id"])}

    #
    # Cuenta las secciones, los fragmentos, las tablas y las figuras de un documento.
    #
    @staticmethod
    def _counts(session, document_id: UUID) -> dict[str, int]:
        elements = dict(
            session.execute(
                select(DocumentElement.kind, func.count())
                .where(DocumentElement.document_id == document_id)
                .group_by(DocumentElement.kind)
            ).all()
        )

        return {
            "section_count": session.scalar(
                select(func.count())
                .select_from(DocumentSection)
                .where(DocumentSection.document_id == document_id)
            ),
            "chunk_count": session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
            ),
            "table_count": elements.get(DocumentElementKind.TABLE, 0),
            "figure_count": elements.get(DocumentElementKind.FIGURE, 0),
        }

    #
    # Devuelve el documento vigente de una fuente con los fragmentos que deben indexarse.
    #
    def indexable(self, source_id: UUID) -> dict[str, Any] | None:
        with self._context.session() as session:
            document = session.execute(
                select(Document.id, Document.title, Document.language).where(
                    Document.source_id == source_id
                )
            ).mappings().first()
            if document is None:
                return None

            rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.position,
                    DocumentChunk.kind,
                    DocumentChunk.text,
                    DocumentChunk.page_start,
                    DocumentChunk.page_end,
                    DocumentSection.path.label("section_path"),
                )
                .outerjoin(DocumentSection, DocumentSection.id == DocumentChunk.section_id)
                .where(DocumentChunk.document_id == document["id"])
                .order_by(DocumentChunk.position)
            ).mappings()

            return {
                "document_id": document["id"],
                "title": document["title"],
                "language": document["language"],
                "chunks": [{**row, "section_path": row["section_path"] or []} for row in rows],
            }

    #
    # Devuelve los fragmentos vigentes de fuentes activas con su procedencia completa.
    #
    def verified_chunks(self, chunk_ids: list[UUID]) -> dict[UUID, dict[str, Any]]:
        if not chunk_ids:
            return {}

        statement = (
            select(
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.document_id,
                DocumentChunk.position,
                DocumentChunk.kind,
                DocumentChunk.token_count,
                DocumentChunk.text,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.locations,
                DocumentChunk.mentions,
                DocumentSection.title.label("section_title"),
                DocumentSection.path.label("section_path"),
                DocumentElement.label.label("element_label"),
                DocumentElement.caption.label("element_caption"),
                DocumentElement.position.label("element_position"),
                Document.title,
                Document.language,
                Document.details,
                Source.id.label("source_id"),
                Source.name.label("source_name"),
                Source.original_filename,
                Source.file_type,
                Source.content_hash,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(Source, Source.id == Document.source_id)
            .outerjoin(DocumentSection, DocumentSection.id == DocumentChunk.section_id)
            .outerjoin(DocumentElement, DocumentElement.id == DocumentChunk.element_id)
            .where(
                DocumentChunk.id.in_(chunk_ids),
                Source.is_active.is_(True),
                Source.processing_status.in_(_READY_STATUSES),
            )
        )

        with self._context.session() as session:
            rows = session.execute(statement).mappings().all()

        return {row["chunk_id"]: _verified_chunk(row) for row in rows}

    #
    # Devuelve una huella del contenido consultable, que cambia al cargar o desactivar fuentes.
    #
    def searchable_version(self) -> tuple:
        statement = (
            select(
                func.count(DocumentChunk.id),
                func.max(DocumentChunk.created_at),
                func.max(Source.updated_at),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(Source, Source.id == Document.source_id)
            .where(*_consultable())
        )

        with self._context.session() as session:
            return tuple(session.execute(statement).one())

    #
    # Devuelve el texto buscable de los fragmentos vigentes, con su sección y leyenda.
    #
    def searchable_chunks(self) -> list[dict[str, Any]]:
        statement = (
            select(
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.document_id,
                DocumentChunk.text,
                DocumentSection.path.label("section_path"),
                DocumentElement.caption.label("element_caption"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(Source, Source.id == Document.source_id)
            .outerjoin(DocumentSection, DocumentSection.id == DocumentChunk.section_id)
            .outerjoin(DocumentElement, DocumentElement.id == DocumentChunk.element_id)
            .where(*_consultable())
        )

        with self._context.session() as session:
            return [dict(row) for row in session.execute(statement).mappings()]


#
# Devuelve las condiciones de un fragmento consultable de una fuente vigente.
#
def _consultable() -> tuple:
    return (
        DocumentChunk.kind.in_(_SEARCHABLE_KINDS),
        Source.is_active.is_(True),
        Source.processing_status.in_(_READY_STATUSES),
    )


#
# Devuelve un fragmento verificado con sus advertencias de extracción, sin el detalle del documento.
#
def _verified_chunk(row) -> dict[str, Any]:
    chunk = dict(row)
    details = chunk.pop("details") or {}
    chunk["doubts"] = chunk_doubts(
        details,
        element_position=chunk["element_position"],
        page_start=chunk["page_start"],
        page_end=chunk["page_end"],
        text=chunk["text"],
    )

    return chunk
