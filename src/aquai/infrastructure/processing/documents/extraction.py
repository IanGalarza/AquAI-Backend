"""Convierte un documento con Docling en un proceso aparte y deja el resultado en un JSON."""

import json
import logging
import re
import sys
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import pypdfium2
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.backend_options import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_EGRET_LARGE
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    LayoutOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_ocr_model import BaseOcrModel
from docling.models.stages.page_assemble.page_assemble_model import PageAssembleModel
from docling_core.types.doc import ContentLayer, DocItemLabel

from aquai.application.contracts.documents import PreparedDocument, PreparedElement
from aquai.application.errors import DocumentInputError
from aquai.application.Rules.documents import (
    SUSPECT_MICRO_UNIT,
    UNMAPPED_CHARACTER,
    detect_language,
    normalized_text,
)
from aquai.domain.enums.documentContent import DocumentChunkKind, DocumentElementKind
from aquai.infrastructure.config.documents import DocumentSettings
from aquai.infrastructure.processing.documents.chunking import DocumentChunker
from aquai.infrastructure.processing.documents.ocrReader import OcrReader, ocr_model_paths
from aquai.infrastructure.processing.documents.pageText import render_box
from aquai.infrastructure.processing.documents.reader import (
    EXPECTED_ERROR_EXIT,
    PageProfile,
    page_profiles,
)
from aquai.infrastructure.processing.documents.structure import (
    attach_loose_captions,
    build_elements,
    build_outline,
    clean_text,
    detach_merged_captions,
    document_doi,
    document_title,
    join_split_ranges,
    release_unnamed_captions,
    restore_headings,
    restore_raw_formulas,
)
from aquai.infrastructure.processing.documents.symbols import (
    PageSymbols,
    SymbolReport,
    restore_symbols,
)
from aquai.infrastructure.processing.documents.tableCheck import verify_tables
from aquai.infrastructure.processing.documents.textOrder import restore_line_order

_DROPPED_CELLS = re.compile(r"^(\d+) of \d+ pdf cells matched neither")
_NUMERIC_RANGE_BREAK = re.compile(r"\d[-–]$")
_LISTED_PAGES = 20
_FIGURE_OCR_SCALE = 3
_MIN_FIGURE_SIDE = 60
_PDF_PARSER_THREADS = 1
_SCANNED_COVERAGE = 0.7
_PDF_METADATA_FIELDS = {
    "Title": "title",
    "Author": "author",
    "Subject": "subject",
    "Keywords": "keywords",
    "Creator": "creator",
    "Producer": "producer",
    "CreationDate": "creation_date",
}


class _DroppedCellCounter(logging.Handler):
    #
    # Prepara el contador de celdas de PDF que no entraron en la grilla de una tabla.
    #
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.dropped = 0

    #
    # Suma las celdas descartadas que informa la reconstrucción de tablas.
    #
    def emit(self, record: logging.LogRecord) -> None:
        match = _DROPPED_CELLS.match(record.getMessage())
        if match:
            self.dropped += int(match.group(1))


#
# Devuelve las páginas sin capa de texto cubiertas por imágenes, que requieren OCR.
#
def pages_needing_ocr(profiles: list[PageProfile], settings: DocumentSettings) -> list[int]:
    return [
        profile.number
        for profile in profiles
        if profile.text_chars <= settings.ocr_max_page_chars
        and (
            profile.image_coverage >= settings.ocr_min_image_coverage
            or profile.has_nested_image
        )
    ]


#
# Limita el OCR a las páginas sin capa de texto: en las demás se conserva el texto del PDF.
#
def _restrict_ocr(pages: list[int]) -> None:
    selected = set(pages)
    ocr_rects = BaseOcrModel.get_ocr_rects

    def selected_rects(self, page):
        return ocr_rects(self, page) if page.page_no in selected else []

    BaseOcrModel.get_ocr_rects = selected_rects


#
# Une las líneas partidas tras un número y un guion, para que Docling no lo borre: 2-4 no es 24.
#
def _joined_numeric_ranges(lines: list[str]) -> list[str]:
    joined: list[str] = []
    for line in lines:
        if joined and _NUMERIC_RANGE_BREAK.search(joined[-1]) and line[:1].isalnum():
            joined[-1] += line
        else:
            joined.append(line)

    return joined


#
# Aplica la unión de rangos numéricos antes de que Docling normalice las líneas de un párrafo.
#
def _keep_numeric_ranges() -> None:
    sanitize_text = PageAssembleModel.sanitize_text

    def sanitized(self, lines: list[str]) -> str:
        return sanitize_text(self, _joined_numeric_ranges(lines))

    PageAssembleModel.sanitize_text = sanitized


#
# Configura el OCR con los modelos incluidos en rapidocr, para que nunca intente descargarlos.
#
def _ocr_options() -> RapidOcrOptions:
    paths = ocr_model_paths()
    return RapidOcrOptions(
        lang=["es"],
        backend="onnxruntime",
        det_model_path=paths["det"],
        rec_model_path=paths["rec"],
        cls_model_path=paths["cls"],
    )


#
# Arma el conversor: el OCR sólo se activa si alguna página no tiene capa de texto. Las páginas
# se leen de a una como mitigación: con lectura paralela hubo corrupción de memoria intermitente
# durante la conversión, sin causa demostrada, y leer en un hilo cuesta poco.
#
def _converter(use_ocr: bool, settings: DocumentSettings) -> DocumentConverter:
    pipeline = PdfPipelineOptions(
        do_ocr=use_ocr,
        do_table_structure=True,
        layout_options=LayoutOptions(model_spec=DOCLING_LAYOUT_EGRET_LARGE),
        generate_page_images=False,
        generate_picture_images=False,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        accelerator_options=AcceleratorOptions(
            num_threads=settings.conversion_threads, device=AcceleratorDevice.CPU
        ),
    )
    if use_ocr:
        pipeline.ocr_options = _ocr_options()

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline,
                backend_options=ThreadedDoclingParseBackendOptions(
                    parser_threads=_PDF_PARSER_THREADS
                ),
            )
        },
    )


#
# Lee los metadatos declarados en el PDF, sin los campos vacíos.
#
def _pdf_metadata(path: Path) -> dict[str, str]:
    document = pypdfium2.PdfDocument(str(path))
    try:
        metadata = document.get_metadata_dict()
    finally:
        document.close()

    return {
        name: metadata[key].strip()
        for key, name in _PDF_METADATA_FIELDS.items()
        if metadata.get(key, "").strip()
    }


#
# Devuelve las páginas que no aportaron ningún elemento al documento extraído.
#
def _pages_without_content(doc: Any, page_count: int) -> list[int]:
    found = {
        prov.page_no for item, _ in doc.iterate_items() for prov in getattr(item, "prov", [])
    }
    return [page for page in range(1, page_count + 1) if page not in found]


#
# Cuenta los encabezados y pies de página que quedan fuera del texto por no ser contenido.
#
def _excluded_furniture(doc: Any) -> int:
    items = doc.iterate_items(included_content_layers={ContentLayer.FURNITURE})
    return sum(item.label != DocItemLabel.FOOTNOTE for item, _ in items)


#
# Escribe una lista de páginas, acotada para que el aviso siga siendo legible.
#
def _pages_text(pages: list[int]) -> str:
    listed = ", ".join(str(page) for page in pages[:_LISTED_PAGES])
    return listed + ("…" if len(pages) > _LISTED_PAGES else "")


#
# Devuelve las páginas de las tablas de una lista de revisión.
#
def _review_pages(entries: list[dict[str, Any]]) -> str:
    return _pages_text(sorted({entry["page"] for entry in entries}))


#
# Reúne los avisos sobre lo que no pudo recuperarse con fidelidad, sin ocultar ninguna pérdida.
#
def _quality_warnings(
    quality: dict[str, Any], ocr_pages: list[int], prior_ocr_pages: list[int]
) -> list[str]:
    warnings = []
    if quality["conversion_errors"]:
        warnings.append(
            f"La conversión fue parcial ({len(quality['conversion_errors'])} errores); "
            "parte del contenido no se pudo extraer y se conserva en el original."
        )
    silent = set(quality["pages_without_content"]) - set(quality["pages_with_lost_text"])
    if silent:
        warnings.append(
            f"Las páginas {_pages_text(sorted(silent))} no aportaron contenido; "
            "pueden estar en blanco o tener contenido que no se pudo leer."
        )
    if quality["pages_with_lost_text"]:
        warnings.append(
            f"Las páginas {_pages_text(quality['pages_with_lost_text'])} no aportaron elementos; "
            "su texto se conserva sin estructura, tal como lo trae el PDF."
        )
    if ocr_pages:
        warnings.append(
            f"Se aplicó OCR en las páginas {_pages_text(ocr_pages)}, que no tienen capa de texto; "
            "el texto reconocido puede contener errores."
        )
    if prior_ocr_pages:
        warnings.append(
            f"Las páginas {_pages_text(prior_ocr_pages)} son imágenes con una capa de texto hecha "
            "por otro OCR; sus cifras y fórmulas pueden tener errores o faltar, y deben "
            "verificarse en el original."
        )
    symbols = quality["symbols"]
    if symbols["decoded"] + symbols["verified"]:
        warnings.append(
            f"Se corrigieron {symbols['decoded'] + symbols['verified']} palabras con símbolos "
            "que el PDF declaraba mal, como µ leída como m; el detalle queda en la extracción."
        )
    if symbols["unresolved"]:
        warnings.append(
            f"{symbols['unresolved']} palabras tienen símbolos que no pudieron resolverse, en las "
            f"páginas {_pages_text(symbols['unresolved_pages'])}; deben verificarse en el original."
        )
    tables = quality["tables"]
    if tables["rebuilt"]:
        warnings.append(
            f"{len(tables['rebuilt'])} tablas no coincidían con su página y se rehicieron desde "
            f"sus líneas visuales, en las páginas {_review_pages(tables['rebuilt'])}: cada línea "
            "impresa es una fila y un registro puede ocupar varias; deben verificarse en el "
            "original."
        )
    if tables["read_by_ocr"]:
        warnings.append(
            f"{len(tables['read_by_ocr'])} tablas perdieron filas en la capa de texto y se leyeron "
            f"por OCR, en las páginas {_review_pages(tables['read_by_ocr'])}; sus valores deben "
            "verificarse en el original."
        )
    if tables["uncertain"]:
        warnings.append(
            f"{len(tables['uncertain'])} tablas no coinciden del todo con su página, en las "
            f"páginas {_review_pages(tables['uncertain'])}; deben verificarse en el original."
        )
    if tables["unverifiable"]:
        warnings.append(
            f"{len(tables['unverifiable'])} tablas no pudieron contrastarse con su página, en "
            f"las páginas {_review_pages(tables['unverifiable'])}, por no tener texto legible "
            "u ocupar varias páginas; deben verificarse en el original."
        )
    if quality["figures_read_by_ocr"]:
        warnings.append(
            f"El texto de {quality['figures_read_by_ocr']} figuras se leyó por OCR y puede "
            "contener errores."
        )
    if quality["suspect_micro_units"]:
        warnings.append(
            f"{quality['suspect_micro_units']} unidades parecen tener una µ mal leída "
            "(lg, pg, ,ug) y deben verificarse en el original."
        )
    if quality["unmapped_characters"]:
        warnings.append(
            f"{quality['unmapped_characters']} caracteres del PDF no tienen equivalente de texto; "
            "se conservan como � y deben verificarse en el original."
        )
    if quality["undecoded_formulas"]:
        warnings.append(
            f"{quality['undecoded_formulas']} fórmulas se conservan como texto crudo sin decodificar."
        )
    if quality["empty_tables"]:
        warnings.append(
            f"{quality['empty_tables']} tablas no tienen celdas recuperadas; "
            "su contenido debe verificarse en el original."
        )
    if quality["dropped_table_cells"]:
        warnings.append(
            f"{quality['dropped_table_cells']} celdas de PDF no pudieron ubicarse en la grilla "
            "de sus tablas; deben verificarse en el original."
        )

    return warnings


#
# Lee el texto de las figuras que son imagen y no traen texto propio, sin interpretarlas.
#
def _read_figures(
    path: Path, elements: list[PreparedElement], reader: OcrReader, limit: int
) -> int:
    figures = [element for element in elements if _is_readable_figure(element)][:limit]
    if not figures:
        return 0

    pdf = pypdfium2.PdfDocument(str(path))
    read = 0
    try:
        for element in figures:
            location = element.locations[0]
            page = pdf[location["page"] - 1]
            lines = reader.confident_lines(render_box(page, location["bbox"], _FIGURE_OCR_SCALE))
            if lines:
                element.content = "\n".join(lines)
                read += 1
    finally:
        pdf.close()

    return read


#
# Indica si vale la pena leer una figura: sin texto propio y de un tamaño razonable.
#
def _is_readable_figure(element: PreparedElement) -> bool:
    if element.kind != DocumentElementKind.FIGURE or element.content or not element.locations:
        return False

    left, top, right, bottom = element.locations[0]["bbox"]
    return min(right - left, bottom - top) >= _MIN_FIGURE_SIDE


#
# Lee el texto de las páginas que Docling no pudo ubicar en ningún elemento.
#
def _page_texts(path: Path, pages: list[int]) -> dict[int, str]:
    if not pages:
        return {}

    pdf = pypdfium2.PdfDocument(str(path))
    try:
        return {
            number: pdf[number - 1].get_textpage().get_text_bounded().strip()
            for number in pages
        }
    finally:
        pdf.close()


#
# Calcula la huella del contenido extraído, independiente del archivo que lo contiene.
#
def _text_hash(chunks: list) -> str:
    text = " ".join(chunk.text for chunk in chunks)
    return sha256(normalized_text(text).encode()).hexdigest()


#
# Convierte un documento y arma su estructura, sus fragmentos y su informe de calidad.
#
def extract(path: Path, settings: DocumentSettings) -> PreparedDocument:
    is_pdf = path.suffix.lower() == ".pdf"
    profiles = page_profiles(path) if is_pdf else []
    ocr_pages = pages_needing_ocr(profiles, settings)
    metadata = _pdf_metadata(path) if is_pdf else {}
    _restrict_ocr(ocr_pages)
    _keep_numeric_ranges()

    cells = _DroppedCellCounter()
    logging.getLogger("MatchingPostProcessor").addHandler(cells)

    started = perf_counter()
    result = _converter(bool(ocr_pages), settings).convert(path, raises_on_error=False)
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        raise DocumentInputError("Docling no pudo convertir el documento.")

    doc = result.document
    undecoded_formulas = restore_raw_formulas(doc)
    release_unnamed_captions(doc)
    detach_merged_captions(doc)
    reordered_paragraphs = restore_line_order(path, doc) if is_pdf else 0
    clean_text(doc)
    restore_headings(doc)
    join_split_ranges(doc)
    reader = OcrReader()
    scanned = {page.number for page in profiles if page.image_coverage >= _SCANNED_COVERAGE}
    prior_ocr_pages = sorted(scanned - set(ocr_pages))
    symbols, glyph_symbols = (
        restore_symbols(path, doc, reader.read, settings.symbol_max_readings, scanned)
        if is_pdf
        else (SymbolReport(), PageSymbols())
    )
    linked_captions = attach_loose_captions(doc)
    outline = build_outline(doc)
    elements, elements_by_ref = build_elements(doc, outline)
    tables = verify_tables(path, elements, glyph_symbols, reader.boxes, scanned)
    figures_read = _read_figures(path, elements, reader, settings.figure_ocr_max) if is_pdf else 0
    chunker = DocumentChunker(settings)
    chunks = chunker.chunk(doc, outline, elements_by_ref)

    empty_pages = _pages_without_content(doc, len(profiles))
    text_chars = {profile.number: profile.text_chars for profile in profiles}
    lost_pages = [page for page in empty_pages if text_chars[page] > settings.ocr_max_page_chars]
    chunks += chunker.page_chunks(_page_texts(path, lost_pages), len(chunks))
    if not chunks:
        raise DocumentInputError("No se pudo extraer texto del documento.")

    quality = {
        "conversion_errors": [str(error.error_message)[:300] for error in result.errors][:20],
        "pages_without_content": empty_pages,
        "pages_with_lost_text": lost_pages,
        "unmapped_characters": sum(len(UNMAPPED_CHARACTER.findall(c.text)) for c in chunks),
        "symbols": asdict(symbols),
        "tables": tables,
        "linked_captions": linked_captions,
        "reordered_paragraphs": reordered_paragraphs,
        "figures_read_by_ocr": figures_read,
        "suspect_micro_units": sum(len(SUSPECT_MICRO_UNIT.findall(c.text)) for c in chunks),
        "undecoded_formulas": undecoded_formulas,
        "dropped_table_cells": cells.dropped,
        "empty_tables": sum(
            element.kind == DocumentElementKind.TABLE and not element.cells for element in elements
        ),
        "excluded_page_furniture": _excluded_furniture(doc),
    }
    title, title_source = document_title(doc, metadata.get("title"))
    prose = " ".join(chunk.text for chunk in chunks if chunk.kind == DocumentChunkKind.TEXT)

    return PreparedDocument(
        text_hash=_text_hash(chunks),
        title=title,
        language=detect_language(prose),
        page_count=len(profiles) or None,
        details={
            "doi": document_doi(doc),
            "title_source": title_source,
            "pdf_metadata": metadata,
            "converter": {"name": "docling", "version": version("docling")},
            "ocr": {
                "applied": bool(ocr_pages),
                "pages_without_text": ocr_pages,
                "pages_with_prior_ocr": prior_ocr_pages,
            },
            "quality": quality,
            "elapsed_seconds": round(perf_counter() - started, 1),
        },
        sections=outline.sections,
        elements=elements,
        chunks=chunks,
        warnings=_quality_warnings(quality, ocr_pages, prior_ocr_pages),
    )


#
# Atiende una solicitud de extracción y deja el resultado o el error esperado en el JSON de salida.
#
def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(request["output"])
    settings = DocumentSettings(**request["settings"])

    try:
        prepared = extract(Path(request["input"]), settings)
    except DocumentInputError as error:
        output.write_text(json.dumps({"error": str(error)}), encoding="utf-8")
        return EXPECTED_ERROR_EXIT

    output.write_text(json.dumps(asdict(prepared)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
