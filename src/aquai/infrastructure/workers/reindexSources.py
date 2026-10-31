"""Solicita reconstruir el índice desde SQL; el worker procesa la solicitud."""

import argparse
from uuid import UUID

from aquai.bootstrap import initialize_application
from aquai.infrastructure.commands.sources import SqlAlchemySourceCommands
from aquai.infrastructure.providers.database import get_db_context
from aquai.infrastructure.queries.sources import SqlAlchemySourceQueries

READY_STATUSES = {"ready", "ready_with_warnings"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", type=UUID)
    args = parser.parse_args()
    initialize_application()

    context = get_db_context()
    state = SqlAlchemySourceQueries(context).get(args.source_id)
    if not state["datasets"] or state["processing_status"] not in READY_STATUSES:
        parser.error("La fuente no tiene datasets confirmados para indexar.")

    SqlAlchemySourceCommands(context).request_reindex(args.source_id)
    print("Índice pendiente; iniciar el backend para que su worker complete la reconstrucción.")
    context.dispose()


if __name__ == "__main__":
    main()
