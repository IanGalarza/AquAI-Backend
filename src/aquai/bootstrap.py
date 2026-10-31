from aquai.infrastructure.persistence.mapping import configure_mappings


def initialize_application() -> None:
    configure_mappings()
