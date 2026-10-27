"""Redacción estructurada con el transporte, cola y configuración Ollama existentes."""

import json

from pydantic import ValidationError

from aquai.application.contracts.answering import ResponseContextRecord, ResponseDraft, ResponseReply
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.response import RESPONSE_PROMPT
from aquai.infrastructure.providers.llm.ollama import _OllamaClient
from aquai.infrastructure.config.answering import AnsweringSettings
from aquai.infrastructure.config.tabular import TabularSettings


class ResponseGenerationError(LLMProviderError):
    def __init__(self, message, *, model=None, input_tokens=None, output_tokens=None):
        super().__init__(message)
        self.model, self.input_tokens, self.output_tokens = model, input_tokens, output_tokens


class OllamaResponseProvider:
    def __init__(self, settings: TabularSettings, answering_settings: AnsweringSettings):
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._output_tokens = min(2048, settings.llm_num_ctx // 4)
        overhead = len(RESPONSE_PROMPT.encode("utf-8")) + len(json.dumps(ResponseDraft.model_json_schema()).encode("utf-8")) + 4096
        self._budget = max(0, min(answering_settings.context_max_bytes,
                                  2 * (settings.llm_num_ctx - self._output_tokens) - overhead))

    @property
    def context_budget_bytes(self) -> int:
        return self._budget

    @property
    def model(self) -> str:
        return self._settings.default_model

    async def generate(self, context: ResponseContextRecord) -> ResponseReply:
        content = json.dumps(context.evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self._budget:
            raise LLMProviderError("La evidencia excede el contexto de redacción permitido.")
        schema = ResponseDraft.model_json_schema()
        sentences = context.evidence["sentence_catalog"]
        schema["properties"]["sentences"]["minItems"] = len(sentences)
        schema["properties"]["sentences"]["maxItems"] = len(sentences)
        schema["$defs"]["DraftSentence"]["properties"]["sentence_id"]["enum"] = [item["sentence_id"] for item in sentences]
        schema["$defs"]["DraftSentence"]["properties"]["evidence_ids"]["items"]["enum"] = list(dict.fromkeys(
            key for item in sentences for key in item["evidence_ids"]))
        schema["$defs"]["DraftSentence"]["properties"]["evidence_ids"]["maxItems"] = 1
        body = await self._client.post("/api/chat", {
            "model": self.model, "stream": False, "think": self._settings.llm_think, "format": schema,
            "options": {"temperature": 0, "num_ctx": self._settings.llm_num_ctx, "num_predict": self._output_tokens},
            "messages": [{"role": "system", "content": RESPONSE_PROMPT}, {"role": "user", "content": content}],
        }, self._settings.llm_timeout_seconds)
        metadata = {"model": body.get("model", self.model), "input_tokens": body.get("prompt_eval_count"),
                    "output_tokens": body.get("eval_count")}
        message = body.get("message")
        if (body.get("done") is not True or body.get("done_reason") == "length" or not isinstance(message, dict)
                or not isinstance(message.get("content"), str)
                or body.get("prompt_eval_count", 0) >= self._settings.llm_num_ctx - self._output_tokens - 256):
            raise ResponseGenerationError("Ollama devolvió una redacción incompleta o sin contexto suficiente.", **metadata)
        try:
            draft = ResponseDraft.model_validate_json(message["content"])
        except ValidationError as error:
            raise ResponseGenerationError("La redacción no cumple el contrato estructurado.", **metadata) from error
        return ResponseReply(draft=draft, **metadata)
