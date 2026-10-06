"""Punto de inicialización de la aplicación, independiente del contexto de base de datos."""

from aquai.infrastructure.persistence.mapping import configure_mappings


def initialize_application() -> None:
    """Registra los mapeos al arrancar, antes de ejecutar los casos de uso."""
    configure_mappings()
