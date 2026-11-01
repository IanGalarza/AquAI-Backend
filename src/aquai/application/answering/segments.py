import re

from aquai.application.contracts.answering import AnswerSegment

_LEGACY_REFERENCE = re.compile(r"\s*\[(e\d+)\]")


#
# Devuelve el texto completo de la respuesta a partir de sus segmentos.
#
def answer_text(segments: list[AnswerSegment]) -> str:
    paragraphs = [_paragraph(segment) for segment in segments]

    return "\n\n".join(paragraphs)


#
# Devuelve un segmento como un párrafo con sus elementos como viñetas.
#
def _paragraph(segment):
    bullets = [f"• {item}" for item in segment.items]

    return "\n".join([segment.text, *bullets])


#
# Convierte un texto con marcadores heredados de evidencia en segmentos de respuesta.
#
def segments_from_text(text: str, evidence_ids=(), hidden=()) -> list[AnswerSegment]:
    known = set(evidence_ids)
    omitted = set(hidden)
    segments = []
    for paragraph in text.split("\n\n"):
        cited = [key for key in _LEGACY_REFERENCE.findall(paragraph) if key in known]
        clean = _without_known_references(paragraph, known)
        if clean and clean not in omitted:
            segments.append(AnswerSegment(text=clean, evidence_ids=list(dict.fromkeys(cited))))

    return segments


#
# Quita de un párrafo los marcadores de evidencia conocidos.
#
def _without_known_references(paragraph, known):
    def removed_if_known(match):
        return "" if match.group(1) in known else match.group(0)

    return _LEGACY_REFERENCE.sub(removed_if_known, paragraph).strip()
