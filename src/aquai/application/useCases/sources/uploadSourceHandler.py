"""Acepta y persiste el archivo original para su procesamiento posterior."""

import asyncio
from collections.abc import Callable
from hashlib import sha256

from aquai.application.interfaces.sources import SourceCommands, SourceQueries
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.models.source import Source


class UploadSourceHandler:
    def __init__(
        self,
        commands: SourceCommands,
        queries: SourceQueries,
        evaluator: TabularEvaluationService,
        *,
        max_upload_bytes: int,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._commands = commands
        self._queries = queries
        self._evaluator = evaluator
        self._notify = notify
        self.max_upload_bytes = max_upload_bytes

    async def handle(self, filename: str | None, content: bytes) -> dict:
        filename = self._evaluator.validate_upload(filename, content)
        source = self._build_source(filename, content)
        source_id, created = await asyncio.to_thread(
            self._commands.create_upload, source, self._initial_recipe(content)
        )
        if self._notify is not None:
            self._notify()
        if created:
            return {
                "id": source_id,
                "processing_status": "uploaded",
                "validation_status": "pending",
            }
        existing = await asyncio.to_thread(self._queries.get, source_id)
        return {
            key: existing[key]
            for key in ("id", "processing_status", "validation_status")
        }

    @staticmethod
    def _build_source(filename: str, content: bytes) -> Source:
        file_type = filename.rsplit(".", 1)[-1].lower()
        mime_type = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }[file_type]
        return Source(
            name=filename,
            original_filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            file_size_bytes=len(content),
            is_active=False,
            processing_status_id=1,
            file_content=content,
        )

    @staticmethod
    def _initial_recipe(content: bytes) -> dict:
        return {
            "proposal": None,
            "effective": None,
            "_workflow": {
                "version": 1,
                "content_sha256": sha256(content).hexdigest(),
                "stage": "uploaded",
                "revision": 0,
            },
        }
