import json

from pydantic import ValidationError

from aquai.application.contracts.routing import (
    GeneralDraft,
    GeneralReply,
    RoutingDraft,
    RoutingReply,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.routing import GENERAL_PROMPT, ROUTER_PROMPT
from aquai.infrastructure.providers.llm.ollama import (
    _OllamaClient,
    exceeds_context,
    prompt_tokens,
)
from aquai.infrastructure.providers.llm.response import ResponseGenerationError


class _StructuredConversationProvider:
    #
    # Guarda la configuración, el prompt y el contrato de salida, y calcula el presupuesto de contexto.
    #
    def __init__(self, settings, prompt, output, *, output_tokens):
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._prompt = prompt
        self._output = output
        self._output_tokens = min(output_tokens, settings.llm_num_ctx // 4)

        overhead = len(prompt.encode()) + len(json.dumps(output.model_json_schema())) + 1024
        self.context_budget_bytes = max(
            0, min(24_000, 2 * (settings.llm_num_ctx - self._output_tokens) - overhead)
        )
        self.model = settings.default_model

    #
    # Envía el contexto a Ollama y devuelve la salida estructurada con sus metadatos.
    #
    async def _generate(self, context, schema=None):
        content = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self.context_budget_bytes:
            raise LLMProviderError("El contexto conversacional supera el límite permitido.")

        body = await self._client.post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": self._settings.llm_think,
                "format": schema or self._output.model_json_schema(),
                "options": {
                    "temperature": 0,
                    "num_ctx": self._settings.llm_num_ctx,
                    "num_predict": self._output_tokens,
                },
                "messages": [
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": content},
                ],
            },
            self._settings.llm_timeout_seconds,
        )
        metadata = {
            "model": body.get("model", self.model),
            "input_tokens": prompt_tokens(body),
            "output_tokens": body.get("eval_count"),
        }
        decision = self._parse_output(self._message_content(body, metadata), metadata)

        return decision, metadata

    #
    # Extrae el contenido de la respuesta y falla si está incompleta o sin contexto suficiente.
    #
    def _message_content(self, body, metadata):
        message = body.get("message")
        context_limit = self._settings.llm_num_ctx - self._output_tokens - 256
        incomplete = (
            body.get("done") is not True
            or body.get("done_reason") == "length"
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or exceeds_context(body, context_limit)
        )
        if incomplete:
            raise ResponseGenerationError("Respuesta conversacional incompleta.", **metadata)

        return message["content"]

    #
    # Valida el contenido devuelto contra el contrato de salida estructurada.
    #
    def _parse_output(self, content, metadata):
        try:
            return self._output.model_validate_json(content)
        except ValidationError as error:
            raise ResponseGenerationError("Salida conversacional inválida.", **metadata) from error


class OllamaRoutingProvider(_StructuredConversationProvider):
    #
    # Configura el proveedor con el prompt y el contrato del enrutamiento.
    #
    def __init__(self, settings):
        super().__init__(settings, ROUTER_PROMPT, RoutingDraft, output_tokens=768)

    #
    # Interpreta la consulta con el historial y devuelve la decisión de enrutamiento.
    #
    async def interpret(self, context):
        schema = self._routing_schema(context["history"])
        decision, metadata = await self._generate(context, schema)

        return RoutingReply(decision=decision, **metadata)

    #
    # Construye el esquema del enrutamiento limitado a los mensajes del historial.
    #
    @staticmethod
    def _routing_schema(history):
        schema = RoutingDraft.model_json_schema()
        schema["properties"]["reason"]["maxLength"] = 240

        history_ids = [item["message_id"] for item in history]
        history_property = schema["properties"]["history_message_ids"]
        if history_ids:
            history_property["items"]["enum"] = history_ids
        else:
            history_property["maxItems"] = 0

        return schema


class OllamaGeneralResponseProvider(_StructuredConversationProvider):
    #
    # Configura el proveedor con el prompt y el contrato de la respuesta general.
    #
    def __init__(self, settings):
        super().__init__(settings, GENERAL_PROMPT, GeneralDraft, output_tokens=1536)

    #
    # Genera la respuesta general sin consultar datos y devuelve su borrador.
    #
    async def generate(self, context):
        draft, metadata = await self._generate(context)

        return GeneralReply(draft=draft, **metadata)
