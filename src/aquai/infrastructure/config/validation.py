from pydantic import AnyHttpUrl, TypeAdapter


#
# Valida que la URL sea HTTP sin parámetros ni fragmentos y la devuelve sin la barra final.
#
def http_base_url(value: str) -> str:
    url = TypeAdapter(AnyHttpUrl).validate_python(value)
    if url.query or url.fragment:
        raise ValueError("La URL no debe incluir parámetros ni fragmentos.")
    return str(url).rstrip("/")


#
# Valida que el nombre de un modelo o colección no esté vacío y lo devuelve sin espacios.
#
def non_empty_name(value: str) -> str:
    name = value.strip()
    if not name:
        raise ValueError("El modelo o colección no puede estar vacío.")
    return name
