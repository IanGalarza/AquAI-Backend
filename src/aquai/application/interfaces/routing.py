from typing import Protocol

from aquai.application.contracts.routing import GeneralReply, RoutingReply


class RoutingProvider(Protocol):
    model: str
    context_budget_bytes: int

    async def interpret(self, context: dict) -> RoutingReply: ...


class GeneralResponseProvider(Protocol):
    model: str
    context_budget_bytes: int

    async def generate(self, context: dict) -> GeneralReply: ...
