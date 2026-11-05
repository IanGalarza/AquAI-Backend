from pathlib import Path
from typing import Any

import rapidocr
from rapidocr import RapidOCR

from aquai.infrastructure.processing.documents.pageText import Glyph

_OCR_MODELS = {
    "det": "PP-OCRv6_det_small.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}
_MIN_SCORE = 0.8
_MIN_BOX_SCORE = 0.6


#
# Devuelve las rutas de los modelos incluidos en rapidocr, para no descargar ninguno.
#
def ocr_model_paths() -> dict[str, str]:
    models = Path(rapidocr.__file__).parent / "models"
    return {name: str(models / filename) for name, filename in _OCR_MODELS.items()}


class OcrReader:
    #
    # Prepara el lector; el modelo se carga recién cuando hace falta leer algo.
    #
    def __init__(self) -> None:
        self._engine: RapidOCR | None = None

    #
    # Lee el texto de una imagen.
    #
    def read(self, image: Any) -> str:
        result = self._ocr(image)
        return " ".join(result.txts or ()) if result is not None else ""

    #
    # Devuelve las líneas leídas con buena confianza, para no guardar ruido.
    #
    def confident_lines(self, image: Any) -> list[str]:
        result = self._ocr(image)
        if result is None or not result.txts:
            return []

        return [
            text
            for text, score in zip(result.txts, result.scores, strict=True)
            if score >= _MIN_SCORE
        ]

    #
    # Devuelve cada texto leído con su ubicación en la imagen, para ordenarlo como la página.
    #
    def boxes(self, image: Any) -> list[Glyph]:
        result = self._ocr(image)
        if result is None or not result.txts:
            return []

        return [
            Glyph(
                text,
                float(min(point[0] for point in box)),
                float(min(point[1] for point in box)),
                float(max(point[0] for point in box)),
                float(max(point[1] for point in box)),
                "",
            )
            for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
            if score >= _MIN_BOX_SCORE
        ]

    #
    # Lee la imagen con el motor de OCR, creándolo la primera vez.
    #
    def _ocr(self, image: Any) -> Any:
        if self._engine is None:
            paths = ocr_model_paths()
            self._engine = RapidOCR(
                params={
                    "Global.log_level": "error",
                    "Det.model_path": paths["det"],
                    "Rec.model_path": paths["rec"],
                    "Cls.model_path": paths["cls"],
                }
            )

        return self._engine(image)
