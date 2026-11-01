from aquai.infrastructure.persistence.mapping import configure_mappings


#
# Inicializa la aplicación configurando los mapeos de persistencia.
#
def initialize_application() -> None:
    configure_mappings()
