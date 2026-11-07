from typing import Protocol

from aquai.application.contracts.answering import ResponseContextRecord, ResponseReply


class ResponseProvider(Protocol):
    #
    # Devuelve el tamaño máximo en bytes del contexto que admite el modelo de respuesta.
    #
    @property
    def context_budget_bytes(self) -> int: ...

    #
    # Devuelve el nombre del modelo que genera la respuesta.
    #
    @property
    def model(self) -> str: ...

    #
    # Genera la respuesta al usuario a partir del contexto de evidencia validada.
    #
    async def generate(self, context: ResponseContextRecord) -> ResponseReply: ...


class TextEmbeddings(Protocol):
    #
    # Devuelve el vector semántico de cada texto.
    #
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
