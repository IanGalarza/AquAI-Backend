"""Cola PostgreSQL con un solo procesamiento e inferencia a la vez."""
import asyncio
from contextlib import asynccontextmanager
import logging
from uuid import UUID

from aquai.application.interfaces.sources import SourceCommands
from aquai.application.useCases.sources.processSourceHandler import ProcessSourceHandler
from aquai.application.utils.background import complete_write

logger = logging.getLogger(__name__)


class SourceWorker:
    def __init__(self, commands: SourceCommands, processor: ProcessSourceHandler,
                 *, poll_seconds: float = 1.0) -> None:
        self._commands = commands
        self._processor = processor
        self._poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None:
            self._stopping = False
            self._task = asyncio.create_task(self.run(), name="aquai-source-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping = True
        self._wake.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def wake(self) -> None:
        self._wake.set()

    async def reprocess(self, source_id: UUID) -> None:
        async with self._worker_lock() as acquired:
            if not acquired:
                raise RuntimeError("Otro worker está procesando; intentar nuevamente cuando termine.")
            await self._processor.handle(source_id, reprocess=True)

    async def run(self) -> None:
        while not self._stopping:
            try:
                if await self.run_once():
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Cola de fuentes no disponible (%s).", type(error).__name__)
            try:
                async with asyncio.timeout(self._poll_seconds):
                    await self._wake.wait()
            except TimeoutError:
                pass
            self._wake.clear()

    async def run_once(self) -> bool:
        # Conexión autocommit reservada para un advisory lock, sin sesión ORM abierta.
        async with self._worker_lock() as acquired:
            if not acquired:
                return False
            await complete_write(self._commands.recover_interrupted)
            source_id = await self._claim()
            if source_id is not None:
                await self._process(source_id)
                return True
            index_id = await self._claim(semantic=True)
            if index_id is None:
                return False
            await self._processor.reindex(index_id)
            return True

    @asynccontextmanager
    async def _worker_lock(self):
        manager = self._commands.worker_lock()
        task = asyncio.create_task(asyncio.to_thread(manager.__enter__))
        try:
            acquired = await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            await complete_write(manager.__exit__, None, None, None)
            raise
        try:
            yield acquired
        finally:
            await complete_write(manager.__exit__, None, None, None)

    async def _claim(self, *, semantic: bool = False) -> UUID | None:
        operation = self._commands.claim_index_next if semantic else self._commands.claim_next
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            source_id = await asyncio.shield(task)
            if source_id is not None:
                if semantic:
                    await complete_write(self._commands.set_index_status, source_id, "pending")
                else:
                    await complete_write(
                        self._commands.record_failure, source_id, "interrupted",
                        "Procesamiento interrumpido al reclamar la fuente.",
                    )
            raise

    async def _process(self, source_id: UUID) -> None:
        try:
            await self._processor.handle(source_id)
        except asyncio.CancelledError:
            await complete_write(
                self._commands.record_failure, source_id, "interrupted",
                "Procesamiento interrumpido; el original y los datos confirmados se conservan.",
            )
            raise
        except Exception as error:
            logger.error("Falló la fuente %s (%s).", source_id, type(error).__name__)
            await complete_write(
                self._commands.record_failure, source_id, "processing_error",
                f"No se pudo completar la preparación ({type(error).__name__}).",
            )
