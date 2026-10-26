"""Adaptador de planificación que reutiliza el transporte y configuración Ollama."""

import json

from pydantic import TypeAdapter, ValidationError

from aquai.application.contracts.planning import PlannerOutput, PlannerReply, PlanningContextRecord
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.planning import PLANNER_PROMPT
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import _OllamaClient


class OllamaPlannerProvider:
    def __init__(self, settings: TabularSettings) -> None:
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._output = TypeAdapter(PlannerOutput)
        self._schema = self._output.json_schema()
        self._output_tokens = min(4096, settings.llm_num_ctx // 4)
        overhead = len(PLANNER_PROMPT.encode("utf-8")) + len(json.dumps(self._schema).encode("utf-8")) + 1024
        self._budget = max(0, min(96_000, 2 * (settings.llm_num_ctx - self._output_tokens) - overhead))

    @property
    def context_budget_bytes(self) -> int:
        return self._budget

    async def propose(self, context: PlanningContextRecord) -> PlannerReply:
        content = json.dumps(context.evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self._budget:
            raise LLMProviderError("El catálogo excede el presupuesto de contexto del Planner.")
        body = await self._client.post("/api/chat", {
            "model": self._settings.default_model, "stream": False, "think": self._settings.llm_think,
            "format": self._schema,
            "options": {"temperature": 0, "num_ctx": self._settings.llm_num_ctx, "num_predict": self._output_tokens},
            "messages": [{"role": "system", "content": PLANNER_PROMPT}, {
                "role": "user", "content": content,
            }],
        }, self._settings.llm_timeout_seconds)
        message = body.get("message")
        if body.get("done") is not True or body.get("done_reason") == "length" or not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError("Ollama devolvió una planificación incompleta.")
        if body.get("prompt_eval_count", 0) >= self._settings.llm_num_ctx - self._output_tokens - 256:
            raise LLMProviderError("El contexto del Planner se aproxima al límite real del modelo; hay que acotar la consulta.")
        try:
            decision = self._output.validate_json(message["content"])
        except ValidationError as error:
            raise LLMProviderError("La planificación de Ollama no cumple el contrato JSON.") from error
        return PlannerReply(decision=decision, model=body["model"],
                            input_tokens=body.get("prompt_eval_count"), output_tokens=body.get("eval_count"))
