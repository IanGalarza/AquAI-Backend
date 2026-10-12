"""Coordina la cola durable de fuentes y su ciclo de vida."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import logging
from uuid import UUID

from aquai.application.interfaces.sources import SourceCommands
from aquai.application.useCases.sources.constants import DATASET_STORAGE
from aquai.application.useCases.sources.processSourceHandler import ProcessSourceHandler


logger = logging.getLogger(__name__)


class SourceWorker:
    def __init__(
        self,
        commands: SourceCommands,
        processor: ProcessSourceHandler,
        *,
        poll_seconds: float = 1.0,
    ) -> None:
        self._commands = commands
        self._processor = processor
        self._poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="aquai-source-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def wake(self) -> None:
        self._wake.set()

    async def run(self) -> None:
        while True:
            try:
                if await self.run_once():
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("No se pudo consultar o avanzar la cola de fuentes.")
            try:
                await asyncio.wait_for(self._wake.wait(), self._poll_seconds)
            except TimeoutError:
                pass
            self._wake.clear()

    async def run_once(self) -> bool:
        # El lock usa una conexión independiente sin una transacción abierta
        # durante inspección, inferencia, ejecución ni validación.
        with self._commands.worker_lock() as acquired:
            if not acquired:
                return False
            await asyncio.to_thread(self._commands.recover_interrupted)
            source_id = await self._claim_next()
            if source_id is None:
                return False
            await self._process(source_id)
            return True

    async def _claim_next(self) -> UUID | None:
        claim = asyncio.create_task(asyncio.to_thread(self._commands.claim_next))
        try:
            return await asyncio.shield(claim)
        except asyncio.CancelledError:
            source_id = await asyncio.shield(claim)
            if source_id is not None:
                await asyncio.shield(self._record_failure(
                    source_id, "interrupted",
                    "El procesamiento fue interrumpido al reclamar la fuente.",
                ))
            raise

    async def _process(self, source_id: UUID) -> None:
        try:
            await self._processor.handle(source_id)
        except asyncio.CancelledError:
            await asyncio.shield(self._record_failure(
                source_id, "interrupted", "El procesamiento fue interrumpido.",
            ))
            raise
        except Exception as exc:
            logger.exception("Falló el procesamiento de la fuente %s.", source_id)
            await self._record_failure(source_id, "processing_error", str(exc))

    async def _record_failure(self, source_id: UUID, code: str, message: str) -> None:
        def update(source: dict) -> tuple[dict, dict]:
            if source["processing_status"] != "processing":
                return {}, {}
            # Releer bajo el lock de fila conserva un checkpoint que terminó
            # de guardarse mientras se cancelaba el procesamiento.
            data = deepcopy(source["recipe_data"])
            data["_workflow"]["stage"] = "interrupted" if code == "interrupted" else "failed"
            report = deepcopy(source.get("validation_report") or {})
            report.setdefault("technical_errors", []).append({
                "code": code,
                "message": message,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            report["reviews"] = []
            report["dataset_storage"] = DATASET_STORAGE
            return {
                "recipe_data": data,
                "validation_report": report,
                "processing_status": "failed",
                "validation_status": "failed",
            }, {}

        await asyncio.to_thread(self._commands.mutate, source_id, update)
