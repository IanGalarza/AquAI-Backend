import json

from pydantic import ValidationError

from aquai.application.contracts.retrieval import (
    EvidenceUsage,
    FragmentSelection,
    ReferenceDraft,
    ReferenceExtraction,
    SearchTermsDraft,
)
from aquai.application.prompts.documents import (
    REFERENCES_PROMPT,
    SEARCH_TERMS_PROMPT,
    SELECTION_PROMPT,
)
from aquai.infrastructure.providers.llm.response import ResponseGenerationError
from aquai.infrastructure.providers.llm.routing import _StructuredConversationProvider

_MAX_REFERENCES = 6
_ITEM_SEPARATORS = " \t\r\n,"


class OllamaDocumentEvidenceProvider:
    #
    # Configura las tareas auxiliares de evidencia documental sobre un mismo agente.
    #
    def __init__(self, ollama, agent):
        self.model = agent.model
        self._terms = _StructuredConversationProvider(
            ollama, agent, SEARCH_TERMS_PROMPT, SearchTermsDraft
        )
        self._selection = _StructuredConversationProvider(
            ollama, agent, SELECTION_PROMPT, FragmentSelection
        )
        self._references = _StructuredConversationProvider(
            ollama, agent, REFERENCES_PROMPT, ReferenceExtraction
        )

    #
    # Convierte pedidos en búsquedas autónomas en español y en inglés.
    #
    async def search_terms(self, context):
        schema = SearchTermsDraft.model_json_schema()
        size = len(context["requests"])
        schema["properties"]["searches"].update({"minItems": size, "maxItems": size})
        draft, metadata = await self._terms._generate(context, schema)

        return draft, EvidenceUsage(**metadata)

    #
    # Ordena los fragmentos por utilidad y descarta los que no sirven para la consulta.
    #
    async def select(self, context):
        size = len(context["fragments"])
        schema = FragmentSelection.model_json_schema()
        schema["properties"]["relevant"]["items"]["enum"] = list(range(1, size + 1))
        schema["properties"]["relevant"]["maxItems"] = size
        draft, metadata = await self._selection._generate(context, schema)

        return list(dict.fromkeys(draft.relevant)), EvidenceUsage(**metadata)

    #
    # Propone valores de referencia de los fragmentos para las variables indicadas; si la salida llega
    # cortada o con un valor inválido, conserva los valores completos, que igual se verifican después.
    #
    async def references(self, context, documents):
        schema = ReferenceExtraction.model_json_schema()
        definition = schema["$defs"]["ReferenceDraft"]
        definition["properties"]["document_id"]["enum"] = documents
        definition["required"] = list(definition["properties"])
        body, metadata = await self._references._reply(context, schema)
        try:
            content = self._references._message_content(body, metadata)
            draft = self._references._parse_output(content, metadata)
        except ResponseGenerationError:
            drafts = _complete_references(body)
            if not drafts:
                raise
            return ReferenceExtraction(references=drafts), EvidenceUsage(**metadata, complete=False)

        return draft, EvidenceUsage(**metadata)


#
# Devuelve los valores que llegaron completos y válidos en una salida cortada o parcialmente inválida.
#
def _complete_references(body) -> list[ReferenceDraft]:
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    start = content.find("[") if isinstance(content, str) else -1
    if start < 0:
        return []

    decoder, position, found = json.JSONDecoder(), start + 1, []
    while len(found) < _MAX_REFERENCES:
        while position < len(content) and content[position] in _ITEM_SEPARATORS:
            position += 1
        if position >= len(content) or content[position] != "{":
            break
        try:
            item, position = decoder.raw_decode(content, position)
            found.append(ReferenceDraft.model_validate(item))
        except json.JSONDecodeError:
            break
        except ValidationError:
            continue

    return found
