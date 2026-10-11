import json

import httpx

from aquai.application.agents.prompts.tabular import TABULAR_RECIPE_PROMPT
from aquai.application.dtos.tabular import LLMResponse, PreparationRecipe, StructuralSummary
from aquai.application.errors import LLMProviderError
from aquai.infrastructure.config.tabular import TabularSettings


class OllamaProvider:
    def __init__(
        self,
        settings: TabularSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    async def generate_recipe(
        self,
        summary: StructuralSummary,
        model: str,
        known_rules: str | None = None,
    ) -> LLMResponse:
        schema = PreparationRecipe.model_json_schema()
        payload = {
            "model": model,
            "stream": False,
            "think": self._settings.llm_think,
            "format": schema,
            "options": {"temperature": 0, "num_ctx": self._settings.llm_num_ctx},
            "messages": [
                {
                    "role": "system",
                    "content": TABULAR_RECIPE_PROMPT + json.dumps(
                        schema, ensure_ascii=False, separators=(",", ":")
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"structural_summary": self._compact_summary(summary), "known_rules": known_rules},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._settings.llm_timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    f"{self._settings.ollama_base_url}/api/chat", json=payload
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMProviderError("Ollama superó el tiempo máximo de respuesta.") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama respondió con un error HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise LLMProviderError("No se pudo completar la conexión con Ollama.") from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError("Ollama devolvió una respuesta HTTP que no es JSON.") from exc

        if not isinstance(body, dict) or body.get("error"):
            raise LLMProviderError("Ollama devolvió una respuesta de protocolo inválida.")
        message = body.get("message")
        returned_model = body.get("model")
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or not isinstance(returned_model, str)
            or not returned_model.strip()
            or body.get("done") is not True
        ):
            raise LLMProviderError("Ollama devolvió una respuesta incompleta o inválida.")

        metadata = {key: value for key, value in body.items() if key != "message"}
        metadata["message"] = {key: value for key, value in message.items() if key != "content"}
        return LLMResponse(model=returned_model, content=message["content"], metadata=metadata)

    @staticmethod
    def _compact_summary(summary: StructuralSummary) -> dict:
        document = summary.model_dump(mode="json", exclude={"record_overlaps"})
        document["sheets"] = [
            sheet.model_dump(mode="json", exclude_defaults=True) for sheet in summary.sheets
        ]
        document["record_overlaps"] = []
        for relation in summary.record_overlaps:
            overlap = relation.model_dump(mode="json", exclude={"matches", "left_only_rows", "right_only_rows"})
            overlap["matches_count"] = len(relation.matches)
            overlap["exclusive_counts"] = {"left": len(relation.left_only_rows), "right": len(relation.right_only_rows)}
            overlap["samples_truncated"] = (
                len(relation.matches) > 8 or len(relation.left_only_rows) > 12 or len(relation.right_only_rows) > 12
            )
            overlap["matches"] = [match.model_dump(mode="json") for match in relation.matches[:8]]
            overlap["left_only_rows"] = relation.left_only_rows[:12]
            overlap["right_only_rows"] = relation.right_only_rows[:12]
            document["record_overlaps"].append(overlap)
        for sheet in document["sheets"]:
            for formula in sheet.get("formula_sample", []):
                expression = formula["formula"]
                if len(expression) > 200:
                    formula["formula"] = expression[:200]
                    formula["formula_characters"] = len(expression)
                    formula["formula_is_preview"] = True
        return document
