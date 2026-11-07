import json

from pydantic import ValidationError

from aquai.application.contracts.answering import (
    ResponseContextRecord,
    ResponseDraft,
    ResponseReply,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.response import LANGUAGE_ORDERS, RESPONSE_PROMPT
from aquai.infrastructure.config.llm import OllamaSettings, ResponseModelSettings
from aquai.infrastructure.providers.llm.ollama import (
    _OllamaClient,
    exceeds_context,
    prompt_tokens,
)


_MAX_SEGMENTS = 5


class ResponseGenerationError(LLMProviderError):
    #
    # Guarda el mensaje del error junto con el modelo y los tokens usados.
    #
    def __init__(self, message, *, model=None, input_tokens=None, output_tokens=None):
        super().__init__(message)
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class OllamaResponseProvider:
    #
    # Guarda la configuración y calcula el presupuesto de contexto de la redacción.
    #
    def __init__(self, ollama: OllamaSettings, agent: ResponseModelSettings):
        self._agent = agent
        self._client = _OllamaClient(ollama, None)
        self._output_tokens = min(agent.max_output_tokens, agent.num_ctx // 4)

        overhead = (
            len(RESPONSE_PROMPT.encode("utf-8"))
            + len(json.dumps(ResponseDraft.model_json_schema()).encode("utf-8"))
            + 4096
        )
        self._budget = max(
            0,
            min(agent.context_max_bytes, 2 * (agent.num_ctx - self._output_tokens) - overhead),
        )

    #
    # Devuelve el presupuesto de contexto de la redacción en bytes.
    #
    @property
    def context_budget_bytes(self) -> int:
        return self._budget

    #
    # Devuelve el nombre del modelo usado para redactar.
    #
    @property
    def model(self) -> str:
        return self._agent.model

    #
    # Pide a Ollama que redacte la respuesta a partir de la evidencia y devuelve el borrador.
    #
    async def generate(self, context: ResponseContextRecord) -> ResponseReply:
        content = json.dumps(context.evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self._budget:
            raise LLMProviderError("La evidencia excede el contexto de redacción permitido.")

        schema = self._draft_schema(context.evidence)
        body = await self._client.post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": self._agent.think,
                "format": schema,
                "options": {
                    "temperature": self._agent.temperature,
                    "num_ctx": self._agent.num_ctx,
                    "num_predict": self._output_tokens,
                },
                "messages": [
                    {
                        "role": "system",
                        "content": f"{RESPONSE_PROMPT}\n{LANGUAGE_ORDERS[context.evidence['language']]}",
                    },
                    {"role": "user", "content": content},
                ],
            },
            self._agent.timeout_seconds,
        )
        metadata = {
            "model": body.get("model", self.model),
            "input_tokens": prompt_tokens(body),
            "output_tokens": body.get("eval_count"),
        }
        draft = self._parse_draft(self._message_content(body, metadata), metadata)

        return ResponseReply(draft=draft, **metadata)

    #
    # Construye el esquema del borrador limitado a la evidencia y a las funciones disponibles.
    #
    @staticmethod
    def _draft_schema(evidence: dict) -> dict:
        schema = ResponseDraft.model_json_schema()
        segment = schema["$defs"]["DraftSegment"]["properties"]
        facts = [item["id"] for item in evidence.get("facts", [])]
        documents = [item["id"] for item in evidence.get("documents", [])]

        _limit_ids(segment["fact_ids"], facts)
        _limit_ids(segment["document_ids"], documents)
        segment["role"]["enum"] = _roles(facts, documents, evidence.get("ambiguities"))
        schema["$defs"]["DraftSegment"]["required"] = list(segment)
        if documents and not facts:
            segment["document_ids"]["minItems"] = 1
        schema["properties"]["segments"]["maxItems"] = _MAX_SEGMENTS
        return schema

    #
    # Extrae el contenido de la respuesta y falla si está incompleta o sin contexto suficiente.
    #
    def _message_content(self, body: dict, metadata: dict) -> str:
        message = body.get("message")
        context_limit = self._agent.num_ctx - self._output_tokens - 256
        incomplete = (
            body.get("done") is not True
            or body.get("done_reason") == "length"
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or exceeds_context(body, context_limit)
        )
        if incomplete:
            raise ResponseGenerationError(
                "Ollama devolvió una redacción incompleta o sin contexto suficiente.", **metadata
            )

        return message["content"]

    #
    # Valida el contenido devuelto contra el contrato del borrador de respuesta.
    #
    @staticmethod
    def _parse_draft(content: str, metadata: dict) -> ResponseDraft:
        try:
            return ResponseDraft.model_validate_json(content)
        except ValidationError as error:
            raise ResponseGenerationError(
                "La redacción no cumple el contrato estructurado.", **metadata
            ) from error


#
# Restringe una lista de identificadores a los valores autorizados.
#
def _limit_ids(field: dict, allowed: list) -> None:
    if allowed:
        field["items"]["enum"] = allowed
    else:
        field["maxItems"] = 0


#
# Devuelve las funciones de párrafo que la evidencia disponible permite usar.
#
def _roles(facts: list, documents: list, ambiguities) -> list:
    roles = ["result"] if facts else []
    if documents:
        roles.append("documented")
    if documents and facts:
        roles.append("relation")
    if documents and ambiguities:
        roles.append("hypothesis")

    return roles
