import re
from dataclasses import dataclass

import semchunk
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
    DocChunk,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
)
from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ImageRefMode,
    TableItem,
)
from transformers import AutoTokenizer

from aquai.application.contracts.documents import PreparedChunk, PreparedElement
from aquai.application.Rules.documents import element_mentions, is_references_heading
from aquai.domain.enums.documentContent import DocumentChunkKind
from aquai.infrastructure.config.documents import DocumentSettings
from aquai.infrastructure.processing.documents.structure import (
    DocumentOutline,
    item_locations,
    item_notes,
    page_span,
    table_lines,
)


_FORMULA_MARKS = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_WORD = re.compile(r"[^\W\d_]{2,}")
_MIN_FIGURE_WORDS = 4


class _OmittedTableSerializer(MarkdownTableSerializer):
    #
    # Deja las tablas y sus celdas fuera del texto corrido: se fragmentan aparte, con su leyenda.
    #
    def serialize(self, *, item, doc_serializer, doc, **kwargs) -> SerializationResult:
        visited = kwargs.get("visited")
        if visited is not None:
            for child, _ in doc.iterate_items(root=item, with_groups=True):
                if child.self_ref != item.self_ref:
                    visited.add(child.self_ref)

        return create_ser_result()


class _OmittedPictureSerializer(MarkdownPictureSerializer):
    #
    # Deja las figuras fuera del texto corrido: su leyenda se conserva en un fragmento propio.
    #
    def serialize(self, *, item, doc_serializer, doc, **kwargs) -> SerializationResult:
        return create_ser_result()


class _BodyTextSerializerProvider(ChunkingSerializerProvider):
    #
    # Serializa sólo el texto corrido, sin marcas de formato ni enlaces ajenos al texto de origen.
    #
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=_OmittedTableSerializer(),
            picture_serializer=_OmittedPictureSerializer(),
            params=MarkdownParams(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder="",
                escape_underscores=False,
                escape_html=False,
                include_hyperlinks=False,
                include_formatting=False,
            ),
        )


@dataclass
class _OrderedChunk:
    order: int
    chunk: PreparedChunk


class DocumentChunker:
    #
    # Prepara el tokenizador del modelo de embeddings y el chunker híbrido de Docling.
    #
    def __init__(self, settings: DocumentSettings) -> None:
        self._max_tokens = settings.chunk_max_tokens
        self._tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(settings.chunk_tokenizer),
            max_tokens=settings.chunk_max_tokens,
        )
        self._chunker = HybridChunker(
            tokenizer=self._tokenizer,
            merge_peers=True,
            serializer_provider=_BodyTextSerializerProvider(),
        )

    #
    # Fragmenta texto, tablas y leyendas de figuras y los devuelve en orden de lectura.
    #
    def chunk(
        self,
        doc: DoclingDocument,
        outline: DocumentOutline,
        elements: dict[str, PreparedElement],
    ) -> list[PreparedChunk]:
        ordered = [
            *self._text_chunks(doc, outline),
            *self._table_chunks(doc, outline, elements),
            *self._figure_chunks(doc, outline, elements),
            *self._figure_text_chunks(doc, outline, elements),
            *self._note_chunks(doc, outline),
        ]
        ordered.sort(key=lambda entry: entry.order)

        chunks = [entry.chunk for entry in ordered]
        for position, chunk in enumerate(chunks):
            chunk.position = position

        return chunks

    #
    # Fragmenta el texto corrido respetando títulos, párrafos y el límite de tokens.
    #
    def _text_chunks(self, doc: DoclingDocument, outline: DocumentOutline) -> list[_OrderedChunk]:
        entries = []
        for chunk in self._chunker.chunk(doc):
            if chunk.text.strip():
                entries.append(self._text_chunk(doc, outline, chunk))

        return entries

    #
    # Arma un fragmento de texto con su sección, sus páginas y las tablas o figuras que menciona.
    #
    def _text_chunk(
        self, doc: DoclingDocument, outline: DocumentOutline, chunk: DocChunk
    ) -> _OrderedChunk:
        first_ref = chunk.meta.doc_items[0].self_ref
        section = outline.section_by_ref.get(first_ref)
        locations = [
            location for item in chunk.meta.doc_items for location in item_locations(doc, item)
        ]
        page_start, page_end = page_span(locations)

        return _OrderedChunk(
            order=outline.order_by_ref.get(first_ref, 0),
            chunk=PreparedChunk(
                position=0,
                kind=_text_kind(outline, section),
                text=_FORMULA_MARKS.sub(r"\1", chunk.text).strip(),
                token_count=self._tokenizer.count_tokens(self._chunker.contextualize(chunk)),
                section_position=section,
                page_start=page_start,
                page_end=page_end,
                locations=locations,
                mentions=element_mentions(chunk.text),
            ),
        )

    #
    # Fragmenta cada tabla por filas, repitiendo su leyenda y su encabezado en cada parte.
    #
    def _table_chunks(
        self,
        doc: DoclingDocument,
        outline: DocumentOutline,
        elements: dict[str, PreparedElement],
    ) -> list[_OrderedChunk]:
        entries = []
        for table in doc.tables:
            element = elements.get(table.self_ref)
            if element is None:
                continue

            order = outline.order_by_ref[table.self_ref]
            for text in self._table_parts(doc, table, element, outline):
                entries.append(_OrderedChunk(order, self._element_chunk(element, text, "table")))

        return entries

    #
    # Reparte una tabla en textos dentro del límite de tokens, repitiendo su contexto en cada uno.
    #
    def _table_parts(
        self,
        doc: DoclingDocument,
        table: TableItem,
        element: PreparedElement,
        outline: DocumentOutline,
    ) -> list[str]:
        lines = table_lines(element.cells or []) if _has_values(element) else []
        headers = element.header_rows if element.header_rows < len(lines) else 0
        section = self._section_tokens(element, outline)
        limit = max(self._max_tokens - section, self._max_tokens // 2)
        context = "\n".join(
            filter(None, [element.caption, *item_notes(doc, table), *lines[:headers]])
        )
        body = lines[headers:]

        if self._tokenizer.count_tokens(context) > limit // 2:
            body = [*self._pieces(context, limit), *body]
            context = ""

        budget = limit - self._tokenizer.count_tokens(context)
        groups = self._grouped(body, budget)
        return ["\n".join(filter(None, [context, *group])) for group in groups if group or context]

    #
    # Agrupa líneas sin superar el presupuesto de tokens, partiendo las que no entran solas.
    #
    def _grouped(self, lines: list[str], budget: int) -> list[list[str]]:
        groups: list[list[str]] = [[]]
        used = 0
        for line in lines:
            for piece in self._pieces(line, budget):
                tokens = self._tokenizer.count_tokens(piece)
                if groups[-1] and used + tokens > budget:
                    groups.append([])
                    used = 0
                groups[-1].append(piece)
                used += tokens

        return groups

    #
    # Parte un texto en trozos dentro del límite de tokens; si entra entero lo devuelve igual.
    #
    def _pieces(self, text: str, limit: int) -> list[str]:
        if self._tokenizer.count_tokens(text) <= limit:
            return [text]

        return semchunk.chunkerify(self._tokenizer.get_tokenizer(), chunk_size=limit)(text)

    #
    # Cuenta los tokens que ocupa la ruta de la sección en el texto que se embebe.
    #
    def _section_tokens(self, element: PreparedElement, outline: DocumentOutline) -> int:
        if element.section_position is None:
            return 0

        section = outline.sections[element.section_position]
        return self._tokenizer.count_tokens("\n".join(section.path))

    #
    # Crea fragmentos con la leyenda y las notas de cada figura, sin interpretar el gráfico.
    #
    def _figure_chunks(
        self,
        doc: DoclingDocument,
        outline: DocumentOutline,
        elements: dict[str, PreparedElement],
    ) -> list[_OrderedChunk]:
        entries = []
        for picture in doc.pictures:
            element = elements.get(picture.self_ref)
            if element is None:
                continue

            legend = "\n".join(filter(None, [element.caption, *item_notes(doc, picture)]))
            if not legend:
                continue

            order = outline.order_by_ref[picture.self_ref]
            for text in self._pieces(legend, self._max_tokens):
                entries.append(_OrderedChunk(order, self._element_chunk(element, text, "figure")))

        return entries

    #
    # Crea fragmentos con el texto impreso dentro de las figuras, sin interpretar el gráfico.
    #
    def _figure_text_chunks(
        self,
        doc: DoclingDocument,
        outline: DocumentOutline,
        elements: dict[str, PreparedElement],
    ) -> list[_OrderedChunk]:
        entries = []
        for picture in doc.pictures:
            element = elements.get(picture.self_ref)
            printed = _printed_text(element.content) if element else ""
            if not printed:
                continue

            title = element.label or "figura"
            order = outline.order_by_ref[picture.self_ref]
            for text in self._pieces(f"Texto impreso en {title}:\n{printed}", self._max_tokens):
                entries.append(_OrderedChunk(order, self._element_chunk(element, text, "figure")))

        return entries

    #
    # Conserva las notas al pie que Docling deja fuera del cuerpo, como las de DOCX.
    #
    def _note_chunks(self, doc: DoclingDocument, outline: DocumentOutline) -> list[_OrderedChunk]:
        order = len(outline.order_by_ref)
        entries = []
        for item, _ in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE}):
            if item.label != DocItemLabel.FOOTNOTE or not item.text.strip():
                continue

            locations = item_locations(doc, item)
            page_start, page_end = page_span(locations)
            for text in self._pieces(item.text.strip(), self._max_tokens):
                entries.append(
                    _OrderedChunk(
                        order,
                        PreparedChunk(
                            position=0,
                            kind=DocumentChunkKind.TEXT,
                            text=text,
                            token_count=self._tokenizer.count_tokens(text),
                            page_start=page_start,
                            page_end=page_end,
                            locations=locations,
                            mentions=element_mentions(text),
                        ),
                    )
                )

        return entries

    #
    # Fragmenta el texto de las páginas que Docling no pudo ubicar en ningún elemento.
    #
    def page_chunks(self, pages: dict[int, str], position: int) -> list[PreparedChunk]:
        chunks = []
        for number, text in pages.items():
            for piece in self._pieces(text, self._max_tokens):
                chunks.append(
                    PreparedChunk(
                        position=position + len(chunks),
                        kind=DocumentChunkKind.TEXT,
                        text=piece,
                        token_count=self._tokenizer.count_tokens(piece),
                        page_start=number,
                        page_end=number,
                    )
                )

        return chunks

    #
    # Arma el fragmento de una tabla o figura, enlazado al elemento del que proviene.
    #
    def _element_chunk(self, element: PreparedElement, text: str, kind: str) -> PreparedChunk:
        return PreparedChunk(
            position=0,
            kind=DocumentChunkKind(kind),
            text=text,
            token_count=self._tokenizer.count_tokens(text),
            section_position=element.section_position,
            element_position=element.position,
            page_start=element.page_start,
            page_end=element.page_end,
            locations=element.locations,
            mentions=[element.label] if element.label else [],
        )


#
# Devuelve el tipo de un fragmento: las subsecciones no heredan la bibliografía, por si es un error.
#
def _text_kind(outline: DocumentOutline, section: int | None) -> DocumentChunkKind:
    if section is not None and is_references_heading(outline.sections[section].title):
        return DocumentChunkKind.REFERENCES

    return DocumentChunkKind.TEXT


#
# Devuelve el texto de una figura si tiene palabras: una figura con sólo un par de rótulos no
# ayuda a buscarla. Se conserva entero porque un número suelto puede ser un dato.
#
def _printed_text(content: str | None) -> str:
    return content if content and len(_WORD.findall(content)) >= _MIN_FIGURE_WORDS else ""


#
# Indica si una tabla tiene alguna celda con contenido.
#
def _has_values(element: PreparedElement) -> bool:
    return any(value for row in element.cells or [] for value in row)
