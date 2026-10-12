"""Reprocesa un original conservando IDs y protegiendo las referencias existentes."""
import argparse
import asyncio
from uuid import UUID

from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.providers.sources import create_source_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", type=UUID)
    args = parser.parse_args()
    runtime = create_source_runtime()
    try:
        asyncio.run(runtime.worker.reprocess(args.source_id))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        get_db_context().dispose()
    print("Original reprocesado; los estados de SQL y del índice están disponibles en GET /sources/{id}.")


if __name__ == "__main__":
    main()
