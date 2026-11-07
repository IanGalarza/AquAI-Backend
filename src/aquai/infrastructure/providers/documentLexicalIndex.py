import asyncio
import math
import re
import unicodedata
from collections import Counter, defaultdict
from uuid import UUID

from aquai.application.interfaces.documents import DocumentQueries
from aquai.application.Rules.evidence import SEARCH_STOPWORDS

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_CONSONANT_PLURAL = re.compile(r"[lrndz]es$")
_K1 = 1.2
_B = 0.75
_MIN_STEM_LENGTH = 4


#
# Devuelve las palabras de un texto sin acentos, mayúsculas, plurales ni palabras vacías.
#
def _terms(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.casefold().replace("µ", "μ"))
    plain = "".join(char for char in folded if not unicodedata.combining(char))

    return [_stem(word) for word in _TOKEN.findall(plain) if word not in SEARCH_STOPWORDS]


#
# Quita el plural de una palabra sin tocar siglas ni códigos.
#
def _stem(word: str) -> str:
    if len(word) <= _MIN_STEM_LENGTH or any(char.isdigit() for char in word):
        return word

    if _CONSONANT_PLURAL.search(word):
        return word[:-2]

    return word[:-1] if word.endswith("s") else word


class _Bm25:
    #
    # Construye el índice invertido de los fragmentos con su sección y su leyenda.
    #
    def __init__(self, rows):
        self._ids = [row["chunk_id"] for row in rows]
        self._documents = [row["document_id"] for row in rows]
        counts = [Counter(_terms(_searchable_text(row))) for row in rows]
        self._lengths = [sum(item.values()) for item in counts]
        self._average = sum(self._lengths) / max(1, len(self._lengths))
        self._postings = defaultdict(list)
        for position, item in enumerate(counts):
            for term, count in item.items():
                self._postings[term].append((position, count))

    #
    # Devuelve los fragmentos con mayor puntaje BM25 para los términos.
    #
    def top(self, query, limit, document_ids) -> list[UUID]:
        allowed = set(document_ids) if document_ids else None
        scores = defaultdict(float)
        for term in set(_terms(query)):
            postings = self._postings.get(term, ())
            weight = self._weight(len(postings))
            for position, count in postings:
                if allowed is None or self._documents[position] in allowed:
                    scores[position] += weight * self._saturated(count, position)

        ranked = sorted(scores, key=scores.get, reverse=True)[:limit]

        return [self._ids[position] for position in ranked]

    #
    # Devuelve el peso de un término según en cuántos fragmentos aparece.
    #
    def _weight(self, frequency) -> float:
        return math.log(1 + (len(self._ids) - frequency + 0.5) / (frequency + 0.5))

    #
    # Devuelve la contribución saturada de un término según el largo del fragmento.
    #
    def _saturated(self, count, position) -> float:
        length = 1 - _B + _B * self._lengths[position] / self._average

        return count * (_K1 + 1) / (count + _K1 * length)


#
# Devuelve el texto buscable de un fragmento: el nombre de sitio a veces sólo está en la sección.
#
def _searchable_text(row) -> str:
    context = [*(row["section_path"] or [])[-2:], row["element_caption"], row["text"]]

    return "\n".join(dict.fromkeys(part for part in context if part))


class Bm25DocumentIndex:
    #
    # Guarda las consultas de documentos; el índice se construye al primer uso.
    #
    def __init__(self, queries: DocumentQueries):
        self._queries = queries
        self._index = None
        self._version = None
        self._lock = asyncio.Lock()

    #
    # Devuelve los identificadores de los fragmentos que mejor coinciden con los términos.
    #
    async def search(self, query, limit, *, document_ids=None) -> list[UUID]:
        index = await self._current()

        return await asyncio.to_thread(index.top, query, limit, document_ids)

    #
    # Devuelve el índice vigente, reconstruyéndolo si el contenido consultable cambió.
    #
    async def _current(self):
        async with self._lock:
            version = await asyncio.to_thread(self._queries.searchable_version)
            if self._index is None or version != self._version:
                rows = await asyncio.to_thread(self._queries.searchable_chunks)
                self._index = await asyncio.to_thread(_Bm25, rows)
                self._version = version

            return self._index
