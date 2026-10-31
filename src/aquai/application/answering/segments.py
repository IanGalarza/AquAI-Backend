"""Párrafos estructurados de una respuesta y su equivalente en texto plano."""

import re

from aquai.application.contracts.answering import AnswerSegment


# Respuestas anteriores al formato estructurado citaban su evidencia con marcadores [eN].
_LEGACY_REFERENCE = re.compile(r"\s*\[(e\d+)\]")


def answer_text(segments: list[AnswerSegment]) -> str:
    """Texto conservado en el historial: sin marcadores ni identificadores técnicos."""
    paragraphs = []
    for segment in segments:
        paragraphs.append("\n".join([segment.text, *(f"• {item}" for item in segment.items)]))
    return "\n\n".join(paragraphs)


def segments_from_text(text: str, evidence_ids=(), hidden=()) -> list[AnswerSegment]:
    """Divide en párrafos y convierte sólo los marcadores de evidencia conocida en referencias."""
    known, omitted = set(evidence_ids), set(hidden)
    segments = []
    for paragraph in text.split("\n\n"):
        cited = [key for key in _LEGACY_REFERENCE.findall(paragraph) if key in known]
        clean = _LEGACY_REFERENCE.sub(lambda match: "" if match.group(1) in known else match.group(0), paragraph).strip()
        if clean and clean not in omitted:
            segments.append(AnswerSegment(text=clean, evidence_ids=list(dict.fromkeys(cited))))
    return segments
