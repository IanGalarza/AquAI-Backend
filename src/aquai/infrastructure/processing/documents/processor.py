import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from tempfile import mkdtemp

from aquai.application.contracts.documents import (
    PreparedChunk,
    PreparedDocument,
    PreparedElement,
    PreparedSection,
)
from aquai.application.errors import DocumentInputError
from aquai.application.Rules.documents import file_suffix
from aquai.infrastructure.config.documents import DocumentSettings
from aquai.infrastructure.processing.documents.reader import (
    EXPECTED_ERROR_EXIT,
    DocumentReader,
)

logger = logging.getLogger(__name__)

_EXTRACTION_MODULE = "aquai.infrastructure.processing.documents.extraction"
_LOG_TAIL_BYTES = 2000
_PROCESS_FLAGS = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0) | getattr(
    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
)
_CLEANUP_ATTEMPTS = 5
_CLEANUP_PAUSE_SECONDS = 0.5


#
# Reconstruye el documento preparado a partir del JSON que dejó el proceso de extracción.
#
def _prepared_document(data: dict) -> PreparedDocument:
    return PreparedDocument(
        **{
            **data,
            "sections": [PreparedSection(**section) for section in data["sections"]],
            "elements": [PreparedElement(**element) for element in data["elements"]],
            "chunks": [PreparedChunk(**chunk) for chunk in data["chunks"]],
        }
    )


#
# Borra la carpeta de trabajo, esperando a que Windows libere los archivos del proceso cortado.
#
def _remove_workspace(workspace: Path) -> None:
    for _ in range(_CLEANUP_ATTEMPTS):
        shutil.rmtree(workspace, ignore_errors=True)
        if not workspace.exists():
            return
        time.sleep(_CLEANUP_PAUSE_SECONDS)


class DoclingDocumentProcessor:
    #
    # Guarda la configuración y el lector que valida los documentos cargados.
    #
    def __init__(self, settings: DocumentSettings) -> None:
        self._settings = settings
        self._reader = DocumentReader(settings)

    #
    # Valida el documento cargado y devuelve su nombre seguro.
    #
    def validate_upload(self, filename: str | None, content: bytes) -> str:
        return self._reader.validate(filename, content)

    #
    # Extrae el documento en un proceso aparte: aísla memoria y fallos, y permite cortarlo.
    #
    async def prepare(self, filename: str, content: bytes) -> PreparedDocument:
        workspace = Path(await asyncio.to_thread(mkdtemp, prefix="aquai-document-"))
        try:
            request = await asyncio.to_thread(self._write_request, workspace, filename, content)
            exit_code = await self._run(workspace, request)

            return await asyncio.to_thread(self._read_result, workspace, exit_code)
        finally:
            await asyncio.shield(asyncio.to_thread(_remove_workspace, workspace))

    #
    # Escribe el original y la solicitud de extracción en la carpeta de trabajo.
    #
    def _write_request(self, workspace: Path, filename: str, content: bytes) -> Path:
        source = workspace / f"document.{file_suffix(filename)}"
        source.write_bytes(content)

        request = workspace / "request.json"
        request.write_text(
            json.dumps(
                {
                    "input": str(source),
                    "output": str(workspace / "result.json"),
                    "settings": self._settings.model_dump(),
                }
            ),
            encoding="utf-8",
        )
        return request

    #
    # Ejecuta la extracción y la corta si vence el tiempo máximo o se cancela el procesamiento.
    #
    async def _run(self, workspace: Path, request: Path) -> int:
        with (workspace / "extraction.log").open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", _EXTRACTION_MODULE, str(request)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self._environment(),
                creationflags=_PROCESS_FLAGS,
            )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(process.wait),
                timeout=self._settings.conversion_timeout_seconds,
            )
        except TimeoutError as error:
            await self._stop(process)
            raise DocumentInputError(
                "La conversión del documento superó el tiempo máximo."
            ) from error
        except asyncio.CancelledError:
            await self._stop(process)
            raise

    #
    # Corta el proceso de extracción y espera a que termine.
    #
    @staticmethod
    async def _stop(process: subprocess.Popen) -> None:
        process.kill()
        await asyncio.shield(asyncio.to_thread(process.wait))

    #
    # Devuelve el entorno del proceso de extracción con sus hilos acotados.
    #
    def _environment(self) -> dict[str, str]:
        threads = str(self._settings.conversion_threads)
        return {
            **os.environ,
            "OMP_NUM_THREADS": threads,
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "PYTHONIOENCODING": "utf-8",
        }

    #
    # Lee el resultado de la extracción o traduce su fallo a un error de entrada.
    #
    def _read_result(self, workspace: Path, exit_code: int) -> PreparedDocument:
        result = workspace / "result.json"
        if exit_code == EXPECTED_ERROR_EXIT and result.exists():
            error = json.loads(result.read_text(encoding="utf-8"))["error"]
            raise DocumentInputError(error)

        if exit_code != 0 or not result.exists():
            log = (workspace / "extraction.log").read_bytes()[-_LOG_TAIL_BYTES:]
            logger.error(
                "Falló la extracción (código %s): %s", exit_code, log.decode(errors="replace")
            )
            raise DocumentInputError(
                f"La conversión del documento terminó de forma inesperada (código {exit_code})."
            )

        return _prepared_document(json.loads(result.read_text(encoding="utf-8")))
