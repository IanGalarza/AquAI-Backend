from typing import Protocol

from aquai.application.contracts.planning import PlannerReply, PlanningContextRecord


class PlannerProvider(Protocol):
    @property
    def context_budget_bytes(self) -> int: ...
    async def propose(self, context: PlanningContextRecord) -> PlannerReply: ...
