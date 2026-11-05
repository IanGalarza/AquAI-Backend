from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from zipfile import BadZipFile, ZipFile

import pypdfium2
import pypdfium2.raw as pdfium_raw

from aquai.application.errors import DocumentInputError, FileTooLargeError
from aquai.application.Rules.documents import DOCUMENT_FILE_TYPES, file_suffix
from aquai.infrastructure.config.documents import DocumentSettings
from aquai.infrastructure.processing.tabular.reader import TableReader

_PDF_SIGNATURE = b"%PDF-"
_PDF_SIGNATURE_WINDOW = 1024
_DOCX_REQUIRED_PARTS = {"[Content_Types].xml", "word/document.xml"}
_ZIP_ENCRYPTED_FLAG = 0x1
EXPECTED_ERROR_EXIT = 2
_pdfium_lock = Lock()


@dataclass(frozen=True)
class PageProfile:
    number: int
    text_chars: int
    image_coverage: float
    has_nested_image: bool = False


#
# Mide la capa de texto y la superficie de imágenes de cada página de un PDF.
#
def page_profiles(path: Path) -> list[PageProfile]:
    document = pypdfium2.PdfDocument(str(path))
    try:
        return [_page_profile(number, page) for number, page in enumerate(document, 1)]
    finally:
        document.close()


#
# Mide los caracteres visibles y la proporción de la página cubierta por imágenes. Una imagen
# dentro de un formulario informa medidas propias, así que sólo se registra que existe.
#
def _page_profile(number: int, page) -> PageProfile:
    width, height = page.get_size()
    area = max(width * height, 1)
    text = page.get_textpage().get_text_bounded()
    images = list(page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE], max_depth=3))
    covered = sum(_object_area(image) for image in images)
    nested = any(image.level > 0 for image in images)

    return PageProfile(number, len("".join(text.split())), min(covered / area, 1.0), nested)


#
# Devuelve el área que ocupa un objeto de la página.
#
def _object_area(page_object) -> float:
    left, bottom, right, top = page_object.get_bounds()
    return abs(right - left) * abs(top - bottom)


class DocumentReader:
    #
    # Guarda la configuración con los límites de lectura de documentos.
    #
    def __init__(self, settings: DocumentSettings) -> None:
        self.settings = settings

    #
    # Valida nombre, extensión, tamaño y estructura del documento y devuelve el nombre seguro.
    #
    def validate(self, filename: str, content: bytes) -> str:
        filename = TableReader.safe_filename(filename)
        file_type = file_suffix(filename)

        if file_type not in DOCUMENT_FILE_TYPES:
            raise DocumentInputError("Sólo se admiten documentos PDF y DOCX.")

        if not content:
            raise DocumentInputError("El archivo está vacío.")

        if len(content) > self.settings.max_upload_bytes:
            raise FileTooLargeError("El archivo supera el tamaño permitido.")

        if file_type == "pdf":
            self._check_pdf(content)
        else:
            self._check_docx(content)

        return filename

    #
    # Verifica que el contenido sea un PDF legible, sin contraseña y dentro del límite de páginas.
    #
    def _check_pdf(self, content: bytes) -> None:
        if _PDF_SIGNATURE not in content[:_PDF_SIGNATURE_WINDOW]:
            raise DocumentInputError("El archivo no tiene un formato PDF válido.")

        pages = self._pdf_page_count(content)
        if pages == 0:
            raise DocumentInputError("El PDF no tiene páginas.")
        if pages > self.settings.max_pages:
            raise DocumentInputError("El PDF supera la cantidad máxima de páginas.")

    #
    # Cuenta las páginas de un PDF; PDFium no admite llamadas simultáneas entre hilos.
    #
    @staticmethod
    def _pdf_page_count(content: bytes) -> int:
        with _pdfium_lock:
            try:
                document = pypdfium2.PdfDocument(content)
            except pypdfium2.PdfiumError as error:
                raise DocumentInputError(
                    "El PDF no se puede abrir: está dañado o protegido con contraseña."
                ) from error

            try:
                return len(document)
            finally:
                document.close()

    #
    # Verifica que el contenido sea un DOCX sin cifrar y que no se expanda de forma desmedida.
    #
    def _check_docx(self, content: bytes) -> None:
        try:
            with ZipFile(BytesIO(content)) as archive:
                entries = archive.infolist()
        except BadZipFile as error:
            raise DocumentInputError("El archivo no tiene un formato DOCX válido.") from error

        names = {entry.filename for entry in entries}
        if not _DOCX_REQUIRED_PARTS.issubset(names):
            raise DocumentInputError("El archivo no tiene un formato DOCX válido.")
        if any(entry.flag_bits & _ZIP_ENCRYPTED_FLAG for entry in entries):
            raise DocumentInputError("El DOCX está protegido con contraseña.")
        if sum(entry.file_size for entry in entries) > self.settings.max_expanded_bytes:
            raise DocumentInputError("El DOCX supera el tamaño expandido permitido.")
