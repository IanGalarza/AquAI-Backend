from typing import Protocol

from aquai.application.contracts.answering import ResponseContextRecord, ResponseReply


class ResponseProvider(Protocol):
    @property
    def context_budget_bytes(self) -> int: ...
    @property
    def model(self) -> str: ...
    async def generate(self, context: ResponseContextRecord) -> ResponseReply: ...
