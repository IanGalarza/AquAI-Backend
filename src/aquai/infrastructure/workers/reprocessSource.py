"""Reprocesa un original: las planillas conservan sus IDs; los documentos regeneran sus fragmentos."""

import argparse
import asyncio
from uuid import UUID

from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.sources import create_source_runtime
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries


#
# Reprocesa el original de una fuente conservando sus identificadores desde la línea de comandos.
#
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", type=UUID)
    args = parser.parse_args()
    runtime = create_source_runtime()

    try:
        source = SqlAlchemySourceQueries(get_db_context()).get(args.source_id)
        asyncio.run(runtime.worker_for(source["file_type"]).reprocess(args.source_id))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        get_db_context().dispose()

    print(
        "Original reprocesado; los estados de SQL y del índice están disponibles en GET /sources/{id}."
    )


if __name__ == "__main__":
    main()
