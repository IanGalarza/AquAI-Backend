from typing import Protocol

from aquai.application.contracts.routing import GeneralReply, RoutingReply


class RoutingProvider(Protocol):
    model: str
    context_budget_bytes: int

    #
    # Interpreta la pregunta del usuario y decide su ruta y menciones.
    #
    async def interpret(self, context: dict) -> RoutingReply: ...


class GeneralResponseProvider(Protocol):
    model: str
    context_budget_bytes: int

    #
    # Genera una respuesta general que no requiere consultar datos.
    #
    async def generate(self, context: dict) -> GeneralReply: ...
