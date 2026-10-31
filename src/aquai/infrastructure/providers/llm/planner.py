import json

from pydantic import TypeAdapter, ValidationError

from aquai.application.contracts.planning import (
    PlannerOutput,
    PlannerReply,
    PlanningContextRecord,
)
from aquai.application.errors import LLMProviderError
from aquai.application.prompts.planning import PLANNER_PROMPT
from aquai.infrastructure.config.tabular import TabularSettings
from aquai.infrastructure.providers.llm.ollama import _OllamaClient

_FREE_TEXT_LIMITS = {"reason": 400, "purpose": 160}


def _cap_free_text(schema):
    """Los textos explicativos no se ejecutan: acotarlos reduce la generación sin cambiar el plan."""
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            _cap_properties(node.get("properties", {}))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _cap_properties(properties):
    for key, value in properties.items():
        if key not in _FREE_TEXT_LIMITS or not isinstance(value, dict) or "maxLength" not in value:
            continue

        value["maxLength"] = min(value["maxLength"], _FREE_TEXT_LIMITS[key])


def _restrict_identifiers(schema, evidence):
    """Los UUID y sql_name sólo pueden copiarse del contexto: el modelo altera dígitos o usa alias como columnas.

    La validación sigue comprobando que cada columna pertenezca a su propio dataset."""
    datasets = [item["dataset_id"] for item in evidence.get("datasets", [])]
    columns = _catalog_columns(evidence)
    if not datasets or not columns:
        return

    selection = schema["$defs"]["DatasetSelection"]["properties"]
    column_ref = schema["$defs"]["ColumnRef"]["properties"]
    selection["dataset_id"]["enum"] = datasets
    column_ref["dataset_id"]["enum"] = datasets
    selection["columns"]["items"]["enum"] = columns
    column_ref["column"]["enum"] = columns


def _catalog_columns(evidence):
    columns = set()
    for rows in evidence.get("column_catalogs", {}).values():
        for row in rows:
            columns.add(row[0])

    return sorted(columns)


class OllamaPlannerProvider:
    def __init__(self, settings: TabularSettings) -> None:
        self._settings = settings
        self._client = _OllamaClient(settings, None)
        self._output = TypeAdapter(PlannerOutput)
        self._schema = self._output.json_schema()
        self._output_tokens = min(4096, settings.llm_num_ctx // 4)

        overhead = (
            len(PLANNER_PROMPT.encode("utf-8"))
            + len(json.dumps(self._schema).encode("utf-8"))
            + 1024
        )
        self._budget = max(
            0, min(96_000, 2 * (settings.llm_num_ctx - self._output_tokens) - overhead)
        )

    @property
    def context_budget_bytes(self) -> int:
        return self._budget

    async def propose(self, context: PlanningContextRecord) -> PlannerReply:
        content = json.dumps(context.evidence, ensure_ascii=False, separators=(",", ":"))
        if len(content.encode("utf-8")) > self._budget:
            raise LLMProviderError("El catálogo excede el presupuesto de contexto del Planner.")

        schema = self._constrained_schema(context.evidence)
        body = await self._client.post(
            "/api/chat",
            {
                "model": self._settings.default_model,
                "stream": False,
                "think": self._settings.llm_think,
                "format": schema,
                "options": {
                    "temperature": 0,
                    "num_ctx": self._settings.llm_num_ctx,
                    "num_predict": self._output_tokens,
                },
                "messages": [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": content},
                ],
            },
            self._settings.llm_timeout_seconds,
        )
        decision = self._parse_decision(self._message_content(body))

        return PlannerReply(
            decision=decision,
            model=body["model"],
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )

    def _constrained_schema(self, evidence: dict) -> dict:
        schema = self._output.json_schema()
        _cap_free_text(schema)
        _restrict_identifiers(schema, evidence)

        mentions = evidence.get("query_mentions", [])
        if any(item["kind"] == "value" for item in mentions):
            schema["$defs"]["ReadPlanStep"]["properties"]["filters"]["minItems"] = 1

        return schema

    def _message_content(self, body: dict) -> str:
        message = body.get("message")
        incomplete = (
            body.get("done") is not True
            or body.get("done_reason") == "length"
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
        )
        if incomplete:
            raise LLMProviderError("Ollama devolvió una planificación incompleta.")

        context_limit = self._settings.llm_num_ctx - self._output_tokens - 256
        if body.get("prompt_eval_count", 0) >= context_limit:
            raise LLMProviderError(
                "El contexto del Planner se aproxima al límite real del modelo; hay que acotar la consulta."
            )

        return message["content"]

    def _parse_decision(self, content: str) -> PlannerOutput:
        try:
            return self._output.validate_json(content)
        except ValidationError as error:
            first_error = error.errors()[0] if error.errors() else {}
            location = ".".join(str(part) for part in first_error.get("loc", ()))
            raise LLMProviderError(
                "La planificación de Ollama no cumple el contrato JSON "
                f"({location}: {first_error.get('msg', 'inválido')})."
            ) from error
