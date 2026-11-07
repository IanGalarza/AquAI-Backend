import unittest
from decimal import Decimal
from types import SimpleNamespace

from aquai.application.answering.drafting import (
    WriterRequest,
    _retaken_facts,
    published_segments,
    review_draft,
)
from aquai.application.answering.hypotheses import unit_factor
from aquai.application.answering.measures import MeasuredResult
from aquai.application.answering.readings import number_readings, number_values, sentences, without_numbers
from aquai.application.answering.references import verified_references
from aquai.application.answering.verification import DraftEvidence, DraftVerifier
from aquai.application.contracts.answering import DraftSegment, ResponseDraft
from aquai.application.contracts.retrieval import ReferenceDraft
from aquai.application.matching import without_reference_request
from aquai.application.Rules.documents import chunk_doubts
from aquai.application.Rules.evidence import CITATION_QUESTION, REPEAT_REQUEST

_FACTS = {
    "s1": "El pH promedio en Punta Lara durante 2021 fue 7,2.",
    "s2": "La conductividad promedio en Punta Lara durante 2021 fue 441,01 µS/cm.",
    "s3": "La turbidez registrada fue 850.",
}
_DOCUMENTS = {
    "d1": "The WHO guideline value for nitrate is 50 mg/L. Blooms reached 5,000 cells/mL.",
    "d2": "Conductivity around 850 µS/cm is typical. Lead was <0.01 mg/L in all samples.",
    "d3": "Tabla escaneada: microcistina 77 µg/L.",
}


def _evidence(**changes) -> DraftEvidence:
    values = {
        "facts": dict(_FACTS),
        "fact_values": {"s1": ["7.2035"], "s2": ["441.01"], "s3": ["850"]},
        "required_facts": set(_FACTS),
        "documents": dict(_DOCUMENTS),
        "document_languages": {"d1": "en", "d2": "en", "d3": "es"},
        "doubtful_documents": {"d3"},
        "document_labels": {"d1": "WHO 2017, p. 183", "d2": "Sathicq et al 2014, p. 12", "d3": "Escaneo, p. 7"},
        "question": "¿Cuál es el pH promedio en Punta Lara durante 2021?",
        "places": ["Punta Lara"],
        "document_contexts": dict(_DOCUMENTS),
    }

    return DraftEvidence(**{**values, **changes})


def _segment(role, text, facts=(), documents=()) -> DraftSegment:
    return DraftSegment(role=role, text=text, fact_ids=list(facts), document_ids=list(documents), items=[])


def _codes(segment, evidence=None) -> set[str]:
    return {item.code for item in DraftVerifier(evidence or _evidence()).check_segment(segment)}


class NumberReadingTests(unittest.TestCase):
    def test_language_decides_the_thousands_separator(self) -> None:
        self.assertEqual(number_values("5,000", "en"), {Decimal("5000")})
        self.assertEqual(number_values("5,000", "es"), {Decimal("5")})
        self.assertEqual(number_values("7.200"), {Decimal("7.2"), Decimal("7200")})

    def test_unit_with_exponent_is_part_of_the_reading(self) -> None:
        reading = number_readings("3,4 mg L-1 en superficie", "es")[0]

        self.assertEqual(reading.unit, "mg L-1")
        self.assertEqual(len(number_readings("3,4 mg L-1 en superficie", "es")), 1)

    def test_huge_exponent_is_not_a_number(self) -> None:
        self.assertEqual(number_readings("1e9999999 mg/L"), [])

    def test_abbreviations_do_not_end_a_sentence(self) -> None:
        self.assertEqual(len(sentences("Según Kruk et al. el valor sube. Luego baja.")), 2)

    def test_doubtful_text_hides_glued_numbers(self) -> None:
        masked = without_numbers("pH7,9 en la toma, densidad 1,5x10⁴ cél/mL.")

        self.assertNotIn("7,9", masked)
        self.assertNotIn("10⁴", masked)
        self.assertNotIn("[valor no [", masked)


class DraftVerificationTests(unittest.TestCase):
    def test_result_keeps_to_the_words_of_its_facts(self) -> None:
        honest = _segment("result", _FACTS["s1"], ["s1"])
        glossed = _segment("result", f"{_FACTS['s1']} Es un agua sin problemas.", ["s1"])

        self.assertEqual(_codes(honest), set())
        self.assertIn("unsupported_result_word", _codes(glossed))

    def test_value_cannot_move_to_another_variable(self) -> None:
        text = "En Punta Lara durante 2021 el pH promedio fue 441,01 y la conductividad promedio fue 7,2."

        self.assertIn("misaligned_value", _codes(_segment("result", text, ["s1", "s2"])))

    def test_respectively_keeps_values_aligned(self) -> None:
        text = (
            "En Punta Lara durante 2021, el pH y la conductividad promedio fueron 7,2 y 441,01 µS/cm, "
            "respectivamente."
        )

        self.assertEqual(_codes(_segment("result", text, ["s1", "s2"])), set())

    def test_exact_values_belong_to_the_cited_fact(self) -> None:
        self.assertEqual(_codes(_segment("result", "El pH promedio fue 7,2035.", ["s1"])), set())
        self.assertIn(
            "unsupported_number", _codes(_segment("result", "El pH promedio fue 441,01.", ["s1"]))
        )

    def test_question_does_not_authorize_numbers(self) -> None:
        evidence = _evidence(question="¿Es cierto que el pH fue 999?")

        self.assertIn(
            "unsupported_number", _codes(_segment("result", "El pH promedio fue 999.", ["s1"]), evidence)
        )

    def test_doubtful_fragment_gives_no_numbers(self) -> None:
        text = "El escaneo informa microcistina de 77 µg/L en la toma."

        self.assertIn("unsupported_number", _codes(_segment("documented", text, documents=["d3"])))

    def test_document_number_follows_the_language_of_the_document(self) -> None:
        text = "Se reportaron floraciones de 5 cells/mL en embalses."

        self.assertIn("unsupported_number", _codes(_segment("documented", text, documents=["d1"])))

    def test_bound_cannot_become_an_exact_value(self) -> None:
        exact = _segment("documented", "El plomo fue 0,01 mg/L en todas las muestras.", documents=["d2"])
        bounded = _segment("documented", "El plomo fue menor a 0,01 mg/L en las muestras.", documents=["d2"])

        self.assertIn("dropped_bound", _codes(exact))
        self.assertNotIn("dropped_bound", _codes(bounded))

    def test_value_without_unit_cannot_take_one_from_a_fragment(self) -> None:
        segment = _segment("documented", "La turbidez registrada fue 850 µS/cm.", ["s3"], ["d2"])

        self.assertIn("unsupported_unit", _codes(segment))

    def test_verdict_about_results_needs_support(self) -> None:
        self.assertIn(
            "unsupported_verdict",
            _codes(_segment("relation", "El pH de 7,2 indica que el agua es segura.", ["s1"], ["d1"])),
        )

    def test_comparison_needs_a_verified_contrast(self) -> None:
        segment = _segment("relation", "La conductividad de 441,01 µS/cm supera ese valor.", ["s2"], ["d2"])

        self.assertIn("unsupported_comparison", _codes(segment))

    def test_contrast_cannot_change_direction(self) -> None:
        evidence = _evidence(
            facts={**_FACTS, "c1": "El pH (7,2) está por debajo de lo reportado en ese lugar: 8,5."},
            fact_directions={"c1": "below"},
        )
        segment = _segment("relation", "El pH de 7,2 cae dentro del rango reportado en ese lugar.", ["c1"], ["d1"])

        self.assertIn("wrong_direction", _codes(segment, evidence))

    def test_relation_without_facts_still_speaks_of_results(self) -> None:
        segment = _segment("relation", "Esto demuestra que los vertidos elevaron la conductividad.", (), ["d2"])

        self.assertIn("unsupported_causality", _codes(segment))

    def test_attribution_must_name_a_cited_source(self) -> None:
        invented = _segment("documented", "Según Pérez, la conductividad ronda 850 µS/cm.", (), ["d2"])
        real = _segment("documented", "Según Sathicq, la conductividad ronda 850 µS/cm.", (), ["d2"])
        translated = _segment("documented", "Según la OMS, el valor guía de nitrato es 50 mg/L.", (), ["d1"])

        self.assertIn("unsupported_attribution", _codes(invented))
        self.assertEqual(_codes(real), set())
        self.assertEqual(_codes(translated), set())

    def test_table_row_of_another_site_is_not_of_the_place(self) -> None:
        table = "Sites (PL: Punta Lara, BAL: Balandra).\n| Sample | pH |\n| BAL-S | 8.6 ±0.1 |\n| PL-A | 7.7 ±0.5 |"
        evidence = _evidence(documents={"d1": table}, document_contexts={"d1": table})
        foreign = _segment("documented", "En Punta Lara se reportó un pH de 8,6 en verano.", (), ["d1"])
        own = _segment("documented", "En Punta Lara se reportó un pH de 7,7 en otoño.", (), ["d1"])

        self.assertIn("unsupported_place", _codes(foreign, evidence))
        self.assertEqual(_codes(own, evidence), set())

    def test_organization_values_come_from_fragments_that_name_it(self) -> None:
        documents = {
            "d1": "La OMS sugiere un valor guía de 1 µg/l.",
            "d2": "Las guías limitan las microcistinas a 2 µg/L para la población general.",
        }
        evidence = _evidence(documents=documents, document_contexts=documents)
        mixed = _segment("documented", "La OMS sugiere valores de hasta 2 µg/L en el agua.", (), ["d1", "d2"])
        kept = _segment("documented", "La OMS sugiere un valor guía de 1 µg/l.", (), ["d1", "d2"])

        self.assertIn("unsupported_attribution", _codes(mixed, evidence))
        self.assertEqual(_codes(kept, evidence), set())

    def test_frequency_needs_a_source_that_expresses_it(self) -> None:
        documents = {"d2": "Conductivity of 850 µS/cm was measured in the estuary."}
        evidence = _evidence(documents=documents, document_contexts=documents)
        added = _segment("documented", "La conductividad suele rondar 850 µS/cm en el estuario.", (), ["d2"])

        self.assertIn("unsupported_frequency", _codes(added, evidence))

    def test_internal_terms_are_not_published(self) -> None:
        segment = _segment("documented", "La conductividad indicada en las ambigüedades es 850 µS/cm.", (), ["d2"])

        self.assertIn("internal_term", _codes(segment))

    def test_hypothesis_needs_an_ambiguity_and_a_condition(self) -> None:
        text = "La turbidez sería 850 µS/cm si esa fuera la unidad."
        segment = _segment("hypothesis", text, ["s3"], ["d2"])
        evidence = _evidence(allows_hypotheses=True, hypothesis_values=["850 µS/cm"])
        asserted = _segment("hypothesis", "La turbidez es 850 µS/cm.", ["s3"], ["d2"])

        self.assertIn("hypothesis_without_ambiguity", _codes(segment))
        self.assertEqual(_codes(segment, evidence), set())
        self.assertIn("hypothesis_not_conditional", _codes(asserted, evidence))


class PublishedDraftTests(unittest.TestCase):
    def test_required_fact_is_restored_when_the_draft_drops_it(self) -> None:
        catalog = {
            key: {"sentence_id": key, "kind": "result", "direct": text, "evidence_ids": ["e1"], "attachment": None}
            for key, text in _FACTS.items()
        }
        draft = ResponseDraft(segments=[_segment("result", "El pH promedio fue 999.", ["s1"])])
        evidence = _evidence()
        request = WriterRequest(record=SimpleNamespace(), evidence=evidence, catalog=catalog)
        reviewed = review_draft(draft, evidence)
        texts = [item.text for item in published_segments(reviewed, request, "es")]

        self.assertEqual(sorted(texts), sorted(_FACTS.values()))


class ReferenceVerificationTests(unittest.TestCase):
    _TABLE = (
        "| Sample | Temp [°C] | pH | Cond [µS/cm] |\n"
        "| PL-A | 16.3 ± 1.9 | 7.7 ±0.5 | 390 ± 98 |\n"
        "| PL-W | 15.5 ± 2.2 | 8.1 ±0.2 | 400 ± 10 |"
    )

    def _references(self, low, high):
        fragment = SimpleNamespace(
            evidence_id="d1",
            text=self._TABLE,
            section=["Punta Lara"],
            element_caption="Table 1",
            title=None,
            doubts=[],
        )
        result = MeasuredResult(
            evidence_id="e1",
            operation_id="op1",
            field="ph_mean",
            label="pH",
            variable="pH",
            column="pH",
            unit=None,
            unit_status="dimensionless",
            basis=None,
            function="mean",
            values=(Decimal("7.2"),),
            sources=(),
        )
        draft = ReferenceDraft(
            document_id="d1",
            variable="pH",
            quote="| PL-A | 16.3 ± 1.9 | 7.7 ±0.5 | 390 ± 98 |",
            low=low,
            high=high,
            unit=None,
            kind="observed",
        )

        return verified_references([draft], [fragment], [result], "pH", ["Punta Lara"])

    def test_mean_of_the_row_is_the_one_the_model_read(self) -> None:
        references = self._references(7.2, 8.2)

        self.assertEqual([(item.low, item.high) for item in references], [(7.7, 7.7)])

    def test_value_of_another_column_is_not_the_variable(self) -> None:
        self.assertEqual(self._references(16.3, 16.3), [])


class FollowUpFactsTests(unittest.TestCase):
    def test_results_and_contrasts_with_a_visible_document_are_retaken(self) -> None:
        contrast = "El pH (7,2) está por debajo de lo reportado en ese lugar: 8,5."
        unseen = "El pH (7,2) está por debajo de lo reportado en otro estudio: 8,1."
        message = SimpleNamespace(
            documents=[{"evidence_id": "d7", "chunk_id": "a"}, {"evidence_id": "d8", "chunk_id": "b"}],
            segments=[
                {"role": "result", "text": "El pH promedio fue 7,2."},
                {"role": "documented", "text": "El pH varía por estación.", "document_ids": ["d7"]},
                {"role": "relation", "text": contrast, "document_ids": ["d7"]},
                {"role": "relation", "text": unseen, "document_ids": ["d8"]},
            ],
        )

        catalog = _retaken_facts([message], [SimpleNamespace(chunk_id="a", evidence_id="d1")])

        self.assertEqual([item["kind"] for item in catalog.values()], ["result", "comparison"])
        self.assertEqual(catalog["p2"]["document_ids"], ["d1"])
        self.assertEqual(catalog["p2"]["direction"], "below")


class QuestionRulesTests(unittest.TestCase):
    def test_reference_clause_keeps_the_filters(self) -> None:
        self.assertEqual(
            without_reference_request("Plot chlorophyll by month and add the guideline value in Berisso"),
            "Plot chlorophyll by month in Berisso",
        )
        self.assertEqual(
            without_reference_request("Graficá la turbiedad con umbral mayor a 100"),
            "Graficá la turbiedad con umbral mayor a 100",
        )

    def test_citation_question_needs_a_reference_to_the_answer(self) -> None:
        self.assertIsNotNone(CITATION_QUESTION.search("¿De dónde sacaste eso?"))
        self.assertIsNone(CITATION_QUESTION.search("¿De dónde sale el agua de Berisso?"))

    def test_repeat_request_is_an_order(self) -> None:
        self.assertIsNotNone(REPEAT_REQUEST.search("Repetí ese resultado"))
        self.assertIsNone(REPEAT_REQUEST.search("¿Hay valores repetidos?"))


class UnitFactorTests(unittest.TestCase):
    def test_prefixes_convert_compatible_units(self) -> None:
        self.assertEqual(unit_factor("mS/cm", "µS/cm"), Decimal("1000"))

    def test_incompatible_units_have_no_factor(self) -> None:
        self.assertIsNone(unit_factor("mg/L", "µS/cm"))


class DoubtMarkTests(unittest.TestCase):
    def test_uncertain_table_and_previous_ocr_mark_the_fragment(self) -> None:
        details = {
            "quality": {"tables": {"uncertain": [{"element": 4}]}},
            "ocr": {"pages_with_prior_ocr": [2]},
        }

        doubts = chunk_doubts(details, element_position=4, page_start=2, page_end=2, text="Valor")

        self.assertEqual(doubts, ["table_uncertain", "prior_ocr_page"])

    def test_prose_of_another_page_has_no_marks(self) -> None:
        details = {"quality": {"tables": {"uncertain": [{"element": 4}]}}, "ocr": {}}

        self.assertEqual(chunk_doubts(details, element_position=9, page_start=3, page_end=3, text="Valor"), [])


if __name__ == "__main__":
    unittest.main()
