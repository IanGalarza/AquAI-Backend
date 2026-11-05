from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from aquai.application.contracts.documents import PreparedDocument
from aquai.application.errors import SourceConflictError, SourceNotFoundError
from aquai.application.Rules.documents import (
    DOCUMENT_FILE_TYPES,
    DOCUMENT_MIME_TYPES,
    file_suffix,
)
from aquai.domain.enums.documentContent import DocumentChunkKind, DocumentElementKind
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus
from aquai.domain.models import (
    Document,
    DocumentChunk,
    DocumentElement,
    DocumentSection,
    MessageSource,
    Source,
)
from aquai.infrastructure.commands.sources import SourceScope
from aquai.infrastructure.persistence.dbContext import DbContext

_WORKER_KEY = int.from_bytes(sha256(b"aquai-document-ingestion").digest()[:8], "big", signed=True)


#
# Indica si la fuente tiene fragmentos extraídos que pueden conservarse ante un fallo.
#
def _has_document_chunks(session: Session, source_id: UUID) -> bool:
    chunk_id = session.scalar(
        select(DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.source_id == source_id)
        .limit(1)
    )
    return chunk_id is not None


DOCUMENT_SCOPE = SourceScope(DOCUMENT_FILE_TYPES, _WORKER_KEY, _has_document_chunks)


#
# Devuelve la fecha y hora actuales en UTC.
#
def _now() -> datetime:
    return datetime.now(timezone.utc)


#
# Devuelve la fuente bloqueada para escritura o falla si no existe.
#
def _locked_source(session: Session, source_id: UUID) -> Source:
    source = session.scalar(select(Source).where(Source.id == source_id).with_for_update())
    if source is None:
        raise SourceNotFoundError("La fuente no existe.")

    return source


#
# Verifica que la fuente sea un documento sin procesos en curso ni referencias de mensajes.
#
def _ensure_replaceable(session: Session, source: Source) -> None:
    if source.file_type not in DOCUMENT_FILE_TYPES:
        raise SourceConflictError("La operación sólo está disponible para documentos PDF y DOCX.")

    busy = (
        source.processing_status == ProcessingStatus.PROCESSING
        or source.semantic_index_status == SemanticIndexStatus.PROCESSING
    )
    if busy:
        raise SourceConflictError("La fuente está siendo procesada; intentar cuando termine.")

    referenced = session.scalar(
        select(MessageSource.id).where(MessageSource.source_id == source.id).limit(1)
    )
    if referenced:
        raise SourceConflictError("La fuente está referenciada por mensajes; se conserva.")


#
# Devuelve a la cola las copias de una fuente que pierde su contenido, para que se indexen solas.
#
def _requeue_duplicates(session: Session, source_id: UUID) -> None:
    duplicates = session.scalars(
        select(Document.source_id).where(Document.duplicate_of_source_id == source_id)
    ).all()
    if not duplicates:
        return

    session.execute(delete(Document).where(Document.source_id.in_(duplicates)))
    for source in session.scalars(select(Source).where(Source.id.in_(duplicates))):
        source.processing_status = ProcessingStatus.UPLOADED
        source.semantic_index_status = SemanticIndexStatus.PENDING
        source.updated_at = _now()


class SqlAlchemyDocumentCommands:
    #
    # Guarda el contexto de base de datos que usa el repositorio.
    #
    def __init__(self, context: DbContext) -> None:
        self._context = context

    #
    # Reemplaza el contenido extraído; descarta el resultado si el original cambió mientras tanto.
    #
    def save_prepared(
        self, source_id: UUID, document: PreparedDocument, report: dict, content_hash: str
    ) -> bool:
        with self._context.session() as session:
            source = _locked_source(session, source_id)
            if source.content_hash != content_hash:
                return False

            previous = session.scalar(
                select(Document.text_hash).where(Document.source_id == source_id)
            )
            session.execute(delete(Document).where(Document.source_id == source_id))
            if previous is not None and previous != document.text_hash:
                _requeue_duplicates(session, source_id)

            original = self._original_of(session, source_id, document.text_hash)
            record = self._document(source_id, document, original)
            session.add(record)
            session.flush()

            if original is None:
                self._store_content(session, record.id, document)

            source.processing_report = self._final_report(report, original)
            source.processing_status = self._status(source.processing_report)
            source.semantic_index_status = SemanticIndexStatus.PENDING
            source.updated_at = _now()
            session.commit()
            return True

    #
    # Devuelve la fuente activa que ya contiene el mismo texto y la bloquea mientras se decide.
    #
    @staticmethod
    def _original_of(session: Session, source_id: UUID, text_hash: str) -> UUID | None:
        return session.scalar(
            select(Document.source_id)
            .join(Source, Source.id == Document.source_id)
            .where(
                Document.text_hash == text_hash,
                Document.source_id != source_id,
                Document.duplicate_of_source_id.is_(None),
                Source.is_active.is_(True),
            )
            .limit(1)
            .with_for_update(of=Document)
        )

    #
    # Construye el registro del documento con sus metadatos.
    #
    @staticmethod
    def _document(source_id: UUID, document: PreparedDocument, original: UUID | None) -> Document:
        return Document(
            source_id=source_id,
            title=document.title,
            language=document.language,
            page_count=document.page_count,
            text_hash=document.text_hash,
            duplicate_of_source_id=original,
            details=document.details,
        )

    #
    # Guarda las secciones, los elementos y los fragmentos enlazándolos entre sí.
    #
    @staticmethod
    def _store_content(session: Session, document_id: UUID, document: PreparedDocument) -> None:
        sections = {
            section.position: DocumentSection(
                document_id=document_id,
                position=section.position,
                level=section.level,
                title=section.title,
                path=section.path,
                page_start=section.page_start,
                page_end=section.page_end,
            )
            for section in document.sections
        }
        session.add_all(sections.values())
        session.flush()

        elements = {
            element.position: DocumentElement(
                document_id=document_id,
                section_id=_linked_id(sections, element.section_position),
                position=element.position,
                kind=DocumentElementKind(element.kind),
                label=element.label,
                caption=element.caption,
                content=element.content,
                cells=element.cells,
                page_start=element.page_start,
                page_end=element.page_end,
                locations=element.locations,
            )
            for element in document.elements
        }
        session.add_all(elements.values())
        session.flush()

        session.add_all(
            DocumentChunk(
                document_id=document_id,
                section_id=_linked_id(sections, chunk.section_position),
                element_id=_linked_id(elements, chunk.element_position),
                position=chunk.position,
                kind=DocumentChunkKind(chunk.kind),
                text=chunk.text,
                token_count=chunk.token_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                locations=chunk.locations,
                mentions=chunk.mentions,
            )
            for chunk in document.chunks
        )

    #
    # Agrega al informe el aviso de copia cuando el texto ya existe en otra fuente.
    #
    @staticmethod
    def _final_report(report: dict[str, Any], original: UUID | None) -> dict[str, Any]:
        if original is None:
            return report

        warning = f"El texto coincide con la fuente {original}; no se indexa de nuevo."
        warnings = [*report["warnings"], warning]
        return {
            **report,
            "warnings": warnings,
            "warning_count": len(warnings),
            "duplicate_of_source_id": str(original),
        }

    #
    # Devuelve el estado de procesamiento que corresponde al informe.
    #
    @staticmethod
    def _status(report: dict[str, Any]) -> ProcessingStatus:
        if report["warnings"]:
            return ProcessingStatus.READY_WITH_WARNINGS

        return ProcessingStatus.READY

    #
    # Reemplaza el original de una fuente y descarta el contenido extraído del archivo anterior.
    #
    def replace_file(self, source_id: UUID, filename: str, content: bytes) -> bool:
        digest = sha256(content).hexdigest()

        with self._context.session() as session:
            source = _locked_source(session, source_id)
            _ensure_replaceable(session, source)
            if source.content_hash == digest:
                return self._retry_failed(session, source)

            owner = session.scalar(select(Source.id).where(Source.content_hash == digest))
            if owner is not None:
                raise SourceConflictError(f"El archivo ya está cargado en la fuente {owner}.")

            session.execute(delete(Document).where(Document.source_id == source_id))
            _requeue_duplicates(session, source_id)

            file_type = file_suffix(filename)
            source.name = filename
            source.original_filename = filename
            source.file_type = file_type
            source.mime_type = DOCUMENT_MIME_TYPES[file_type]
            source.file_size_bytes = len(content)
            source.file_content = content
            source.content_hash = digest
            source.processing_status = ProcessingStatus.UPLOADED
            source.processing_report = {}
            source.semantic_index_status = SemanticIndexStatus.PENDING
            source.updated_at = _now()
            session.commit()
            return True

    #
    # Vuelve a poner en cola una fuente fallida cuando se reenvía el mismo archivo.
    #
    @staticmethod
    def _retry_failed(session: Session, source: Source) -> bool:
        if source.processing_status != ProcessingStatus.FAILED:
            return False

        source.processing_status = ProcessingStatus.UPLOADED
        source.updated_at = _now()
        session.commit()
        return True

    #
    # Verifica que la fuente se pueda eliminar sin perder referencias ni cortar un proceso.
    #
    def ensure_removable(self, source_id: UUID) -> None:
        with self._context.session() as session:
            _ensure_replaceable(session, _locked_source(session, source_id))

    #
    # Elimina una fuente documental con su contenido; sus copias vuelven a la cola.
    #
    def delete_source(self, source_id: UUID) -> None:
        with self._context.session() as session:
            _ensure_replaceable(session, _locked_source(session, source_id))
            session.execute(delete(Document).where(Document.source_id == source_id))
            _requeue_duplicates(session, source_id)
            session.execute(delete(Source).where(Source.id == source_id))
            session.commit()


#
# Devuelve el identificador del registro enlazado por posición, si lo hay.
#
def _linked_id(records: dict[int, Any], position: int | None) -> UUID | None:
    return records[position].id if position is not None else None
