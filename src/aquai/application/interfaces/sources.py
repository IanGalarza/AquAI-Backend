from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol
from uuid import UUID

from aquai.application.contracts.ingestion import PreparedTable
from aquai.application.contracts.retrieval import VerifiedDataset
from aquai.application.contracts.routing import CatalogDataset
from aquai.application.dtos.requests.listSources import SourceSort
from aquai.domain.models.source import Source


class SourceQueries(Protocol):
    #
    # Devuelve los datasets del catálogo que coinciden con una referencia.
    #
    def matching_datasets(self, reference: str) -> list[CatalogDataset]: ...

    #
    # Devuelve un resumen del catálogo de fuentes y datasets.
    #
    def catalog_overview(self) -> list[dict]: ...

    #
    # Devuelve los documentos cargados, que el chat lista pero todavía no consulta.
    #
    def document_overview(self) -> list[dict]: ...

    #
    # Devuelve los valores de una columna de un dataset que coinciden con un texto.
    #
    def matching_column_values(
        self, dataset_id: UUID, version: int, column: str, value: str
    ) -> list[str]: ...

    #
    # Busca valores de texto de los datasets que contienen los términos indicados.
    #
    def find_text_values(
        self, datasets: list[VerifiedDataset], terms: list[str], *, per_column: int = 50
    ) -> dict[UUID, dict[str, list[dict]]]: ...

    #
    # Devuelve los valores de texto distintos de las columnas indicadas de un dataset.
    #
    def distinct_text_values(
        self, dataset: VerifiedDataset, columns: list[str], *, limit: int = 500
    ) -> dict[str, list[str]]: ...

    #
    # Devuelve los valores de texto más frecuentes de una columna de un dataset.
    #
    def frequent_text_values(
        self, dataset: VerifiedDataset, column: str, *, measured: str | None = None, limit: int = 12
    ) -> list[dict]: ...

    #
    # Devuelve una página de fuentes con filtros por nombre y estado, y un orden.
    #
    def list(
        self,
        *,
        page: int,
        page_size: int,
        search: str | None = None,
        processing_status: str | None = None,
        sort: SourceSort = SourceSort.NEWEST,
    ) -> dict: ...

    #
    # Devuelve el detalle de una fuente, con su contenido si se solicita.
    #
    def get(self, source_id: UUID, *, include_content: bool = False) -> dict: ...

    #
    # Devuelve los identificadores de los datasets que se pueden consultar.
    #
    def usable_dataset_ids(
        self, dataset_ids: list[UUID], *, versions: dict[UUID, int] | None = None
    ) -> set[UUID]: ...

    #
    # Devuelve los datasets verificados, opcionalmente en versiones concretas.
    #
    def verified_datasets(
        self, dataset_ids: list[UUID], *, versions: dict[UUID, int] | None = None
    ) -> list[VerifiedDataset]: ...


class SourceCommands(Protocol):
    #
    # Registra una fuente cargada y devuelve su identificador e indicador de duplicado.
    #
    def create_upload(self, source: Source) -> tuple[UUID, bool]: ...

    #
    # Obtiene el bloqueo que garantiza un único worker de procesamiento activo.
    #
    def worker_lock(self) -> AbstractContextManager[bool]: ...

    #
    # Reclama la siguiente fuente pendiente de procesar, si existe.
    #
    def claim_next(self) -> UUID | None: ...

    #
    # Recupera las fuentes cuyo procesamiento quedó interrumpido y devuelve cuántas son.
    #
    def recover_interrupted(self) -> int: ...

    #
    # Guarda las tablas preparadas de una fuente junto con su informe de procesamiento.
    #
    def save_prepared(
        self,
        source_id: UUID,
        tables: list[PreparedTable],
        report: dict,
        *,
        replace_existing: bool = False,
    ) -> None: ...

    #
    # Guarda la descripción semántica de un dataset y la de sus columnas.
    #
    def save_dataset_semantics(
        self,
        dataset_id: UUID,
        description: str,
        columns: list[dict],
        warnings: list[str] | None = None,
    ) -> None: ...

    #
    # Actualiza el estado del índice semántico de una fuente.
    #
    def set_index_status(
        self,
        source_id: UUID,
        status: str,
        error: str | None = None,
        expected_active: bool | None = None,
    ) -> None: ...

    #
    # Reclama la siguiente fuente pendiente de indexar, si existe.
    #
    def claim_index_next(self) -> UUID | None: ...

    #
    # Activa o desactiva una fuente.
    #
    def set_active(self, source_id: UUID, active: bool) -> None: ...

    #
    # Registra el fallo de procesamiento de una fuente con su código y mensaje.
    #
    def record_failure(self, source_id: UUID, code: str, message: str) -> None: ...

    #
    # Solicita volver a indexar una fuente.
    #
    def request_reindex(self, source_id: UUID) -> None: ...
