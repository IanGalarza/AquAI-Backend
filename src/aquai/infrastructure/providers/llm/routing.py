"""Router y redacción general reutilizando el transporte y la cola de Ollama."""

import json

from pydantic import ValidationError

from aquai.application.contracts.routing import (
    GeneralDraft, GeneralReply, RoutingDraft, RoutingReply,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.routing import GENERAL_PROMPT, ROUTER_PROMPT
from aquai.infrastructure.providers.llm.ollama import _OllamaClient
from aquai.infrastructure.providers.llm.response import ResponseGenerationError


class _StructuredConversationProvider:
    def __init__(self, settings, prompt, output, *, output_tokens):
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._prompt = prompt
        self._output = output
        self._output_tokens = min(output_tokens, settings.llm_num_ctx // 4)
        overhead = len(prompt.encode()) + len(json.dumps(output.model_json_schema())) + 1024
        self.context_budget_bytes = max(0, min(
            24_000, 2 * (settings.llm_num_ctx - self._output_tokens) - overhead
        ))
        self.model = settings.default_model

    async def _generate(self, context, schema=None):
        content = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self.context_budget_bytes:
            raise LLMProviderError("El contexto conversacional supera el límite permitido.")
        body = await self._client.post("/api/chat", {
            "model": self.model, "stream": False, "think": self._settings.llm_think,
            "format": schema or self._output.model_json_schema(),
            "options": {
                "temperature": 0, "num_ctx": self._settings.llm_num_ctx,
                "num_predict": self._output_tokens,
            },
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": content},
            ],
        }, self._settings.llm_timeout_seconds)
        metadata = {
            "model": body.get("model", self.model),
            "input_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
        }
        message = body.get("message")
        if (
            body.get("done") is not True or body.get("done_reason") == "length"
            or not isinstance(message, dict) or not isinstance(message.get("content"), str)
            or body.get("prompt_eval_count", 0)
            >= self._settings.llm_num_ctx - self._output_tokens - 256
        ):
            raise ResponseGenerationError("Respuesta conversacional incompleta.", **metadata)
        try:
            decision = self._output.model_validate_json(message["content"])
        except ValidationError as error:
            raise ResponseGenerationError("Salida conversacional inválida.", **metadata) from error
        return decision, metadata


class OllamaRoutingProvider(_StructuredConversationProvider):
    def __init__(self, settings):
        super().__init__(settings, ROUTER_PROMPT, RoutingDraft, output_tokens=768)

    async def interpret(self, context):
        schema = RoutingDraft.model_json_schema()
        schema["properties"]["reason"]["maxLength"] = 240  # Un motivo breve acorta la generación.
        ids = [item["message_id"] for item in context["history"]]
        if ids:
            schema["properties"]["history_message_ids"]["items"]["enum"] = ids
        else:
            schema["properties"]["history_message_ids"]["maxItems"] = 0
        decision, metadata = await self._generate(context, schema)
        return RoutingReply(decision=decision, **metadata)


class OllamaGeneralResponseProvider(_StructuredConversationProvider):
    def __init__(self, settings):
        super().__init__(settings, GENERAL_PROMPT, GeneralDraft, output_tokens=1536)

    async def generate(self, context):
        draft, metadata = await self._generate(context)
        return GeneralReply(draft=draft, **metadata)
