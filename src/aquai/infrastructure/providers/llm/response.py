import json

from pydantic import ValidationError

from aquai.application.contracts.answering import (
    ResponseContextRecord,
    ResponseDraft,
    ResponseReply,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.response import RESPONSE_PROMPT
from aquai.infrastructure.config.answering import AnsweringSettings
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import (
    _OllamaClient,
    exceeds_context,
    prompt_tokens,
)


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
    def __init__(self, settings: TabularSettings, answering_settings: AnsweringSettings):
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._output_tokens = min(2048, settings.llm_num_ctx // 4)

        overhead = (
            len(RESPONSE_PROMPT.encode("utf-8"))
            + len(json.dumps(ResponseDraft.model_json_schema()).encode("utf-8"))
            + 4096
        )
        self._budget = max(
            0,
            min(
                answering_settings.context_max_bytes,
                2 * (settings.llm_num_ctx - self._output_tokens) - overhead,
            ),
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
        return self._settings.default_model

    #
    # Pide a Ollama que redacte la respuesta a partir de la evidencia y devuelve el borrador.
    #
    async def generate(self, context: ResponseContextRecord) -> ResponseReply:
        content = json.dumps(context.evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self._budget:
            raise LLMProviderError("La evidencia excede el contexto de redacción permitido.")

        schema = self._draft_schema(context.evidence["sentence_catalog"])
        body = await self._client.post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": self._settings.llm_think,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": self._settings.llm_num_ctx,
                    "num_predict": self._output_tokens,
                },
                "messages": [
                    {"role": "system", "content": RESPONSE_PROMPT},
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
        draft = self._parse_draft(self._message_content(body, metadata), metadata)

        return ResponseReply(draft=draft, **metadata)

    #
    # Construye el esquema del borrador limitado a las oraciones y evidencias del catálogo.
    #
    @staticmethod
    def _draft_schema(sentences: list) -> dict:
        schema = ResponseDraft.model_json_schema()
        sentence_list = schema["properties"]["sentences"]
        sentence_list["minItems"] = len(sentences)
        sentence_list["maxItems"] = len(sentences)

        draft_sentence = schema["$defs"]["DraftSentence"]["properties"]
        draft_sentence["sentence_id"]["enum"] = [item["sentence_id"] for item in sentences]
        draft_sentence["evidence_ids"]["items"]["enum"] = _unique_evidence_ids(sentences)
        draft_sentence["evidence_ids"]["maxItems"] = 1
        return schema

    #
    # Extrae el contenido de la respuesta y falla si está incompleta o sin contexto suficiente.
    #
    def _message_content(self, body: dict, metadata: dict) -> str:
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
# Devuelve los identificadores de evidencia sin repetir de las oraciones del catálogo.
#
def _unique_evidence_ids(sentences: list) -> list:
    evidence_ids = []
    for item in sentences:
        evidence_ids.extend(item["evidence_ids"])

    return list(dict.fromkeys(evidence_ids))
