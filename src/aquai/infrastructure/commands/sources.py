"""Escrituras atómicas y exclusión de workers sobre Source/SourceRecipe existentes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock
from typing import Any
from uuid import UUID
from weakref import WeakKeyDictionary

from sqlalchemy import Engine, and_, func, select, text
from sqlalchemy.orm import Session

from aquai.application.errors import SourceConflictError, SourceNotFoundError
from aquai.domain.models import Source, SourceRecipe
from aquai.infrastructure.persistence.configurations.statusSeeds import (
    PROCESSING_STATUS_SEEDS, VALIDATION_STATUS_SEEDS,
)
from aquai.infrastructure.persistence.dbContext import DbContext
from aquai.infrastructure.queries.sources import source_snapshot


_PROCESSING = {status.name: status.id for status in PROCESSING_STATUS_SEEDS}
_VALIDATION = {status.name: status.id for status in VALIDATION_STATUS_SEEDS}
_WORKER_KEY = int.from_bytes(sha256(b"aquai-source-worker-v1").digest()[:8], "big", signed=True)
_engine_locks: WeakKeyDictionary[Engine, tuple[Lock, Lock]] = WeakKeyDictionary()
_engine_locks_guard = Lock()
_UNSET = object()


def _sqlite_locks(engine: Engine) -> tuple[Lock, Lock]:
    # SQLite se usa exclusivamente para tests aislados; PostgreSQL coordina procesos.
    with _engine_locks_guard:
        if engine not in _engine_locks:
            _engine_locks[engine] = (Lock(), Lock())
        return _engine_locks[engine]


def managed_workflow_predicate(dialect_name: str):
    """Reconoce nuestra versión sin castear JSON histórico arbitrario en PostgreSQL."""
    if dialect_name == "sqlite":
        return and_(
            func.json_type(SourceRecipe.recipe_data, "$._workflow.version") == "integer",
            SourceRecipe.recipe_data["_workflow"]["version"].as_integer() == 1,
        )
    return SourceRecipe.recipe_data.contains({"_workflow": {"version": 1}})


class SqlAlchemySourceCommands:
    """Cada cambio confirma su propia transacción, sin sostenerla durante inferencias."""

    def __init__(self, context: DbContext) -> None:
        self._context = context

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        engine = self._context.engine
        lock = _sqlite_locks(engine)[1] if engine.dialect.name == "sqlite" else None
        if lock is not None:
            lock.acquire()
        try:
            with self._context.session() as session:
                yield session
                session.commit()
        finally:
            if lock is not None:
                lock.release()

    @contextmanager
    def worker_lock(self) -> Iterator[bool]:
        """Reserva una conexión en autocommit; el lock de sesión no es una transacción."""
        engine = self._context.engine
        if engine.dialect.name == "sqlite":
            lock = _sqlite_locks(engine)[0]
            acquired = lock.acquire(blocking=False)
            try:
                yield acquired
            finally:
                if acquired:
                    lock.release()
            return
        if engine.dialect.name != "postgresql":
            raise RuntimeError("El worker durable requiere PostgreSQL.")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            acquired = bool(connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": _WORKER_KEY}))
            connection.commit()
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": _WORKER_KEY})
                    connection.commit()

    def create_upload(self, source: Source, recipe_data: dict[str, Any]) -> tuple[UUID, bool]:
        """Persiste original y receta pendiente juntos, deduplicados por contenido."""
        document = deepcopy(recipe_data)
        workflow = document.setdefault("_workflow", {})
        digest = sha256(source.file_content).hexdigest()
        if workflow.get("content_sha256", digest) != digest:
            raise ValueError("La huella de la fuente no corresponde al original.")
        workflow["content_sha256"] = digest
        workflow.setdefault("version", 1)
        if type(workflow["version"]) is not int or workflow["version"] != 1:
            raise ValueError("Versión de procesamiento de fuente no admitida.")
        workflow.setdefault("stage", "uploaded")
        workflow.setdefault("revision", 0)
        if source.processing_status_id != _PROCESSING["uploaded"]:
            raise ValueError("Una fuente nueva debe comenzar en uploaded.")
        with self._write_session() as session:
            if session.bind is not None and session.bind.dialect.name == "postgresql":
                # Serializa sólo cargas del mismo contenido sin requerir nuevos índices.
                upload_key = int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=True)
                session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": upload_key})
            existing = session.scalar(
                select(Source.id).join(SourceRecipe, SourceRecipe.source_id == Source.id)
                .where(
                    managed_workflow_predicate(self._context.engine.dialect.name),
                    SourceRecipe.recipe_data["_workflow"]["content_sha256"].as_string() == digest,
                )
                .order_by(Source.created_at, Source.id).limit(1)
            )
            if existing is not None:
                return existing, False
            session.add(source)
            # El ORM conoce la dependencia FK aunque la navegación no esté asignada.
            session.add(SourceRecipe(source_id=source.id, recipe_data=document, rules_snapshot={}))
            session.flush()
            return source.id, True

    def claim_next(self) -> UUID | None:
        """Reclama una fuente; el caller mantiene worker_lock durante todo el pipeline."""
        with self._write_session() as session:
            source = session.scalar(
                select(Source).join(SourceRecipe, SourceRecipe.source_id == Source.id)
                .where(
                    Source.processing_status_id == _PROCESSING["uploaded"],
                    managed_workflow_predicate(self._context.engine.dialect.name),
                )
                .order_by(Source.created_at, Source.id).limit(1)
                .with_for_update(of=Source, skip_locked=True)
            )
            if source is None:
                return None
            recipe = session.scalar(select(SourceRecipe).where(SourceRecipe.source_id == source.id).with_for_update())
            if recipe is None:
                raise SourceConflictError("La fuente pendiente no tiene SourceRecipe.")
            source.processing_status_id = _PROCESSING["processing"]
            source.updated_at = datetime.now(timezone.utc)
            document = deepcopy(recipe.recipe_data)
            workflow = document.setdefault("_workflow", {})
            if workflow.get("stage") == "uploaded":
                workflow["stage"] = "inspection"
            workflow["claimed_at"] = source.updated_at.isoformat()
            recipe.recipe_data = document
            recipe.updated_at = source.updated_at
            return source.id

    @staticmethod
    def _locked_entities(session: Session, source_id: UUID) -> tuple[Source, SourceRecipe]:
        source = session.scalar(select(Source).where(Source.id == source_id).with_for_update())
        if source is None:
            raise SourceNotFoundError("La fuente no existe.")
        recipe = session.scalar(select(SourceRecipe).where(SourceRecipe.source_id == source_id).with_for_update())
        if recipe is None:
            raise SourceConflictError("La fuente no tiene SourceRecipe.")
        return source, recipe

    @staticmethod
    def _apply(source: Source, recipe: SourceRecipe, updates: dict[str, Any]) -> None:
        allowed = {
            "recipe_data", "rules_snapshot", "validation_report", "processing_status",
            "validation_status", "is_active",
        }
        if set(updates) - allowed:
            raise ValueError("La actualización incluye campos de fuente no admitidos.")
        if "processing_status" in updates:
            value = updates["processing_status"]
            if value not in _PROCESSING:
                raise ValueError("Estado de procesamiento no admitido.")
            source.processing_status_id = _PROCESSING[value]
        if "validation_status" in updates:
            value = updates["validation_status"]
            if value not in _VALIDATION:
                raise ValueError("Estado de validación no admitido.")
            recipe.validation_status_id = _VALIDATION[value]
        if "is_active" in updates:
            if not isinstance(updates["is_active"], bool):
                raise ValueError("is_active debe ser booleano.")
            source.is_active = updates["is_active"]
        for field in ("recipe_data", "rules_snapshot", "validation_report"):
            if field in updates:
                value = updates[field]
                if not isinstance(value, dict) and not (field == "validation_report" and value is None):
                    raise ValueError(f"{field} debe ser un objeto JSON.")
                setattr(recipe, field, deepcopy(value))
        now = datetime.now(timezone.utc)
        source.updated_at = now
        recipe.updated_at = now

    def save(
        self, source_id: UUID, *, recipe_data: Any = _UNSET, rules_snapshot: Any = _UNSET,
        validation_report: Any = _UNSET, processing_status: Any = _UNSET,
        validation_status: Any = _UNSET, is_active: Any = _UNSET,
        expected_processing_status: str | None = None,
    ) -> bool:
        values = {
            "recipe_data": recipe_data, "rules_snapshot": rules_snapshot,
            "validation_report": validation_report, "processing_status": processing_status,
            "validation_status": validation_status, "is_active": is_active,
        }
        updates = {key: value for key, value in values.items() if value is not _UNSET}
        with self._write_session() as session:
            source, recipe = self._locked_entities(session, source_id)
            if expected_processing_status is not None:
                if expected_processing_status not in _PROCESSING:
                    raise ValueError("Estado esperado no admitido.")
                if source.processing_status_id != _PROCESSING[expected_processing_status]:
                    return False
            self._apply(source, recipe, updates)
        return True

    def mutate(
        self, source_id: UUID,
        callback: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
    ) -> dict[str, Any]:
        """Decisión idempotente y validación de vigencia en una transacción corta."""
        with self._write_session() as session:
            source, recipe = self._locked_entities(session, source_id)
            snapshot = source_snapshot(session, source_id)
            assert snapshot is not None
            updates, response = callback(snapshot)
            if updates:
                self._apply(source, recipe, updates)
            return deepcopy(response)

    def recover_interrupted(self) -> int:
        """Bajo worker_lock, registra ejecuciones abandonadas sin borrar su evidencia."""
        count = 0
        with self._write_session() as session:
            sources = session.scalars(
                select(Source).join(SourceRecipe, SourceRecipe.source_id == Source.id)
                .where(
                    Source.processing_status_id == _PROCESSING["processing"],
                    managed_workflow_predicate(self._context.engine.dialect.name),
                )
                .with_for_update(of=Source)
            ).all()
            for source in sources:
                recipe = session.scalar(select(SourceRecipe).where(SourceRecipe.source_id == source.id).with_for_update())
                if recipe is None:
                    source.processing_status_id = _PROCESSING["failed"]
                    source.updated_at = datetime.now(timezone.utc)
                    source.is_active = False
                    count += 1
                    continue
                document = deepcopy(recipe.recipe_data)
                workflow = document.setdefault("_workflow", {})
                previous_stage = workflow.get("stage")
                workflow["stage"] = "interrupted"
                workflow["revision"] = workflow.get("revision", 0) + 1
                report = deepcopy(recipe.validation_report or {})
                report.setdefault("technical_errors", []).append({
                    "code": "processing_interrupted", "stage": previous_stage,
                    "message": "El procesamiento se interrumpió antes de completar la fuente.",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                self._apply(source, recipe, {
                    "recipe_data": document, "validation_report": report,
                    "processing_status": "failed", "validation_status": "failed", "is_active": False,
                })
                count += 1
        return count
