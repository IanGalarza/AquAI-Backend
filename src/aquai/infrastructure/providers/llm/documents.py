from aquai.application.contracts.retrieval import (
    EvidenceUsage,
    FragmentSelection,
    ReferenceExtraction,
    SearchTermsDraft,
)
from aquai.application.prompts.documents import (
    REFERENCES_PROMPT,
    SEARCH_TERMS_PROMPT,
    SELECTION_PROMPT,
)
from aquai.infrastructure.providers.llm.routing import _StructuredConversationProvider


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
    # Propone valores de referencia de los fragmentos para las variables indicadas.
    #
    async def references(self, context, documents):
        schema = ReferenceExtraction.model_json_schema()
        schema["$defs"]["ReferenceDraft"]["properties"]["document_id"]["enum"] = documents
        draft, metadata = await self._references._generate(context, schema)

        return draft, EvidenceUsage(**metadata)
