from typing import Protocol

from aquai.application.contracts.planning import PlannerReply, PlanningContextRecord


class PlannerProvider(Protocol):
    #
    # Devuelve el tamaño máximo en bytes del contexto que admite el modelo planificador.
    #
    @property
    def context_budget_bytes(self) -> int: ...

    #
    # Propone un plan de análisis a partir del contexto de planificación.
    #
    async def propose(self, context: PlanningContextRecord) -> PlannerReply: ...
