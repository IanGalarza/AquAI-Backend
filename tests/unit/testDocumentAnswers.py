import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from aquai.application.answering.drafting import (
    WriterRequest,
    _lines_and_pairs,
    _retaken_facts,
    _without_repetitions,
    published_segments,
    review_draft,
)
from aquai.application.answering.hypotheses import unit_factor
from aquai.application.answering.measures import MeasuredResult
from aquai.application.answering.readings import number_readings, number_values, sentences, without_numbers
from aquai.application.answering.references import verified_references
from aquai.application.answering.verification import DraftEvidence, DraftVerifier, paraphrase_drift
from aquai.application.contracts.answering import AnswerSegment, DraftSegment, ResponseDraft
from aquai.application.contracts.retrieval import ReferenceDraft
from aquai.application.contracts.routing import DataMention, DatasetBinding, HistoryMessage
from aquai.application.matching import abbreviates, named_documents, without_reference_request
from aquai.application.Rules.documents import chunk_doubts
from aquai.application.Rules.evidence import CITATION_QUESTION, CITATION_YEAR, REPEAT_REQUEST
from aquai.application.Services.generateResponseService import _numbered_lines, _scoped_claim
from aquai.application.Services.routeQueryService import RouteQueryService, _narrowed_to_sheets

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
        paraphrased = _segment(
            "result", "Según el análisis del conjunto de mediciones, en Punta Lara el pH medio de 2021 fue 7,2.", ["s1"]
        )
        glossed = _segment("result", f"{_FACTS['s1']} Es un agua sin problemas.", ["s1"])
        assessed = _segment("result", f"{_FACTS['s1']} Es un valor elevado.", ["s1"])
        elsewhere = _segment("result", "En Punta Lara y en Balandra el pH promedio de 2021 fue 7,2.", ["s1"])
        another = _segment("result", "La turbidez promedio en Punta Lara durante 2021 fue 7,2.", ["s1"])

        self.assertEqual(_codes(honest), set())
        self.assertEqual(_codes(paraphrased), set())
        self.assertTrue(_codes(glossed))
        self.assertIn("unsupported_result_word", _codes(assessed))
        self.assertIn("unsupported_result_word", _codes(elsewhere))
        self.assertIn("unsupported_result_word", _codes(another))

    def test_comparison_between_results_follows_their_values(self) -> None:
        facts = {
            "s1": "La turbiedad promedio en Brown 419 fue 64,88 UNT.",
            "s2": "La turbiedad promedio en Brown 330 fue 41,72 UNT.",
        }
        evidence = _evidence(facts=facts, required_facts=set(facts), fact_values={}, question="Compará la turbiedad")
        right = "La turbiedad promedio en Brown 419 fue 64,88 UNT, superior a los 41,72 UNT de Brown 330."
        reversed_ = "La turbiedad promedio en Brown 419 fue 41,72 UNT, superior a los 64,88 UNT de Brown 330."

        self.assertNotIn("unsupported_comparison", _codes(_segment("result", right, ["s1", "s2"]), evidence))
        self.assertIn("wrong_direction", _codes(_segment("result", reversed_, ["s1", "s2"]), evidence))

    def test_value_cannot_move_to_another_variable(self) -> None:
        text = "En Punta Lara durante 2021 el pH promedio fue 441,01 y la conductividad promedio fue 7,2."

        self.assertIn("misaligned_value", _codes(_segment("result", text, ["s1", "s2"])))

    def test_one_wrong_sentence_does_not_take_the_rest_of_a_result_paragraph(self) -> None:
        text = (
            "El pH promedio en Punta Lara durante 2021 fue 7,2. "
            "La conductividad promedio en Punta Lara durante 2021 fue 999 µS/cm."
        )
        reviewed = review_draft(ResponseDraft(segments=[_segment("result", text, ["s1", "s2"])]), _evidence())

        self.assertEqual([item.text for item in reviewed.accepted], ["El pH promedio en Punta Lara durante 2021 fue 7,2."])
        self.assertIn("s2", reviewed.missing_facts)

    def test_fragments_of_one_document_are_one_source(self) -> None:
        documents = {**_DOCUMENTS, "d1": "A level of 20000 cells/ml.", "d2": "A level of 100000 cells/ml."}
        text = "La guía propone niveles de 20.000 y 100.000 células por mililitro."
        same = _evidence(documents=documents, document_contexts=documents, document_owners={"d1": "who", "d2": "who"})
        other = _evidence(documents=documents, document_contexts=documents, document_owners={"d1": "who", "d2": "epa"})

        self.assertNotIn("mixed_sources", _codes(_segment("documented", text, (), ["d1", "d2"]), same))
        self.assertIn("mixed_sources", _codes(_segment("documented", text, (), ["d1", "d2"]), other))

    def test_digits_in_a_source_name_are_not_figures(self) -> None:
        text = "En BaseCiano_PARA BELEN_5MAR2024 (2), el pH promedio en Punta Lara durante 2021 fue 7,2."
        named = _evidence(source_names=["BaseCiano_PARA BELEN_5MAR2024 (2)"])

        self.assertEqual(_codes(_segment("result", text, ["s1"]), named), set())
        self.assertTrue({"unread_number", "unsupported_number"} & _codes(_segment("result", text, ["s1"])))

    def test_the_parts_of_a_date_are_not_misaligned_values(self) -> None:
        facts = {**_FACTS, "s4": "El pH promedio fue 7,2 y los registros van del 01/09/2021 al 31/03/2022."}
        evidence = _evidence(facts=facts, fact_values={**_evidence().fact_values, "s4": ["7.2"]})
        dated = _segment("result", "El pH promedio fue 7,2, con registros entre el 01/09/2021 y el 31/03/2022.", ["s4"])
        moved = _segment("result", "El pH promedio fue 7,2, con registros entre el 05/09/2021 y el 31/03/2022.", ["s4"])

        self.assertNotIn("misaligned_value", _codes(dated, evidence))
        self.assertIn("unsupported_number", _codes(moved, evidence))

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

    def test_an_ion_formula_is_not_an_unread_number(self) -> None:
        documents = {**_DOCUMENTS, "d2": "Nutrients (particularly NO3 - and NH4 + ) modulated the distribution."}
        evidence = _evidence(documents=documents, document_contexts=documents)
        segment = _segment("documented", "Los nutrientes, en particular NO3- y NH4+, modularon la distribución.", (), ["d2"])

        self.assertNotIn("unread_number", _codes(segment, evidence))

    def test_where_the_data_come_from_is_not_a_cause(self) -> None:
        origin = _segment("relation", "Esos valores provienen de archivos distintos.", (), ["d2"])
        cause = _segment("relation", "Esos valores provienen de la erosión de los suelos.", (), ["d2"])

        self.assertNotIn("unsupported_causality", _codes(origin))
        self.assertIn("unsupported_causality", _codes(cause))

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


class ParaphraseScopeTests(unittest.TestCase):
    _SATHICQ = (
        "The statistical analysis performed in this study showed that the variables that primarily modulate the "
        "temporal distribution of cyanobacteria were temperature, pH, light penetration, conductivity, and nutrients."
    )
    _ECHENIQUE = (
        "Estos organismos, desde microscópicos hasta visibles a simple vista, presentan una gran diversidad de formas, "
        "y son capaces de habitar casi todos los rincones del planeta; desde los desiertos polares hasta los "
        "ecuatoriales, desde marinos hasta cuerpos de agua continentales, aguas dulces y salobres."
    )

    def test_naming_the_place_a_title_qualifies_keeps_the_qualifier(self) -> None:
        heading = "Temporal distribution of cyanobacteria in the coast of a shallow temperate estuary (Río de la Plata)"
        named = (
            "Según el estudio de Sathicq et al. (2014), en la costa del Río de la Plata la distribución temporal de "
            "las cianobacterias dependió de la temperatura, el pH, la penetración de luz, la conductividad y los nutrientes."
        )
        unnamed = (
            "Según el estudio de Sathicq et al. (2014), la distribución temporal de las cianobacterias dependió de la "
            "temperatura, el pH, la penetración de luz, la conductividad y los nutrientes."
        )

        self.assertIsNone(paraphrase_drift(named, self._SATHICQ, heading=heading))
        self.assertEqual(paraphrase_drift(unnamed, self._SATHICQ, heading=heading).code, "lost_qualifier")

    def test_a_citation_year_does_not_align_a_claim_with_publication_data(self) -> None:
        lines = [
            ("d1", "Received: 20 December 2013 /Accepted: 30 June 2014 /Published online: 9 July 2014", "", ()),
            ("d1", "A guideline level of 20000 cyanobacterial cells/ml can be derived.", "", ()),
        ]
        cited = "Según Sathicq et al. (2014), la temperatura y el pH modularon la distribución temporal."
        measured = "Según la OMS (2003), se deriva un nivel guía de 20.000 células/ml."

        self.assertEqual(_numbered_lines(CITATION_YEAR.sub(" ", cited), lines, "es"), ([], []))
        self.assertEqual(_numbered_lines(CITATION_YEAR.sub(" ", measured), lines, "es")[0], [lines[1][1:]])

    def test_one_side_of_a_contrast_is_its_own_support_line(self) -> None:
        line = "Nutrients modulated the distribution, while the winds and tide had a secondary effect at an hourly scale."

        self.assertIn("the winds and tide had a secondary effect at an hourly scale.", _lines_and_pairs(line))

    def test_a_claim_that_only_loses_the_study_scope_is_scoped_instead_of_removed(self) -> None:
        claim = (
            "Las variables que modulan principalmente la distribución temporal de las cianobacterias son la "
            "temperatura, el pH, la penetración de luz, la conductividad y los nutrientes."
        )

        lost = paraphrase_drift(claim, self._SATHICQ)
        scoped = _scoped_claim(claim, "es")

        self.assertEqual(lost.family, "alcance del estudio")
        self.assertTrue(scoped.endswith("los nutrientes, según el estudio citado."))
        self.assertIsNone(paraphrase_drift(scoped, self._SATHICQ))

    def test_a_qualifier_of_another_clause_is_not_demanded_in_the_same_language(self) -> None:
        claim = "Estos organismos presentan una gran diversidad de formas, desde microscópicos hasta visibles a simple vista."

        self.assertIsNone(paraphrase_drift(claim, self._ECHENIQUE, same_language=True))
        self.assertIsNotNone(paraphrase_drift(claim, self._ECHENIQUE))

    def test_a_qualifier_of_the_paraphrased_clause_is_still_demanded(self) -> None:
        passage = "Algunas cepas de cianobacterias pueden producir toxinas potentes."
        claim = "Las cianobacterias pueden producir toxinas potentes."

        self.assertEqual(paraphrase_drift(claim, passage, same_language=True).family, "cuantificador parcial")

    def test_a_sentence_that_repeats_an_earlier_one_is_not_published_twice(self) -> None:
        first = AnswerSegment(
            role="documented",
            text="Los cambios hidrológicos, como sequías más extensas, favorecen a la mayoría de las cianobacterias.",
        )
        second = AnswerSegment(
            role="documented",
            text="Además, los cambios hidrológicos como sequías más extensas favorecen a las cianobacterias. "
            "El fósforo superó el valor citado para su desarrollo.",
        )

        published = _without_repetitions([first, second])

        self.assertEqual([item.text for item in published], [first.text, "El fósforo superó el valor citado para su desarrollo."])


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

    def test_a_restored_fact_keeps_its_place_after_the_results_it_follows(self) -> None:
        catalog = {
            key: {"sentence_id": key, "kind": "result", "direct": text, "evidence_ids": ["e1"], "attachment": None}
            for key, text in _FACTS.items()
        }
        draft = ResponseDraft(
            segments=[_segment("result", "La conductividad promedio en Punta Lara durante 2021 fue 441,01 µS/cm.", ["s2"])]
        )
        evidence = _evidence()
        request = WriterRequest(record=SimpleNamespace(), evidence=evidence, catalog=catalog)
        texts = [item.text for item in published_segments(review_draft(draft, evidence), request, "es")]

        self.assertEqual(texts, [_FACTS["s1"], _FACTS["s2"], _FACTS["s3"]])


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

    def test_a_stored_value_can_write_the_name_with_initials_and_symbols(self) -> None:
        address = "C: AVENIDA ALTE G BROWN NÂ° 419 Esq. Planta Potabilizdora [Entrada de Planta][AN]"

        self.assertTrue(abbreviates(address, "almirante brown 419"))
        self.assertTrue(abbreviates(address, "Brown 419"))
        self.assertTrue(abbreviates("Club de Pesca", "club pesca"))
        self.assertTrue(abbreviates("Pta Lara", "Punta Lara"))
        self.assertFalse(abbreviates(address, "brown 330"))
        self.assertFalse(abbreviates("Estación 1 3", "estación 3"))

    def test_a_document_is_named_despite_a_typo_and_the_period_asked(self) -> None:
        documents = [
            {"document_id": "a", "name": "Aguilera et al., 2018.pdf", "title": None},
            {"document_id": "g", "name": "Gianuzzi et al., 2011.pdf", "title": None},
            {"document_id": "s14", "name": "Sathicq et al., 2014.pdf", "title": None},
            {"document_id": "s15", "name": "Sathicq et al., 2015.pdf", "title": None},
            {"document_id": "b", "name": "Barros et al., Cambio Climatico.pdf", "title": None},
        ]

        def named(question):
            return [item["document_id"] for item in named_documents(question, documents)]

        self.assertEqual(named("En 2005, ¿hubo recuentos altos? Contrastalo con lo que comunica Giannuzzi et al."), ["g"])
        self.assertEqual(named("Según Sathicq et al. (2014), ¿qué variables se asociaron?"), ["s14"])
        self.assertEqual(named("¿Qué dice Sathicq sobre el viento?"), ["s14", "s15"])
        self.assertEqual(named("¿En qué barrios hubo floraciones en 2014?"), [])

    def test_a_statement_that_continues_the_last_topic_is_a_follow_up(self) -> None:
        question = HistoryMessage(
            message_id=uuid4(), role="user", text="¿Qué factores favorecen las floraciones de cianobacterias?"
        )
        answer = HistoryMessage(
            message_id=uuid4(),
            role="assistant",
            text="Otro factor reconocido es el aumento de la temperatura media, junto con los nutrientes.",
            documents=[{"evidence_id": "d1"}],
            scientific=True,
        )
        history = [question, answer]

        def continues(text):
            return RouteQueryService._continues_topic(text, answer, history, [])

        self.assertTrue(continues("Entonces la temperatura es un principal factor"))
        self.assertTrue(continues("entonces la temperatura es un principal factor?"))
        self.assertTrue(continues("o sea que el factor clave es otro"))
        self.assertFalse(continues("gracias, muy claro"))
        self.assertFalse(continues("Hola, ¿qué podés hacer?"))
        self.assertFalse(
            RouteQueryService._continues_topic(
                "la temperatura en Punta Lara", answer, history, [DataMention(text="Punta Lara", kind="value")]
            )
        )

        def states(text):
            return RouteQueryService._continues_statement(text, answer, history, [])

        self.assertTrue(states("Entonces la temperatura es un principal factor"))
        self.assertTrue(states("entonces la temperatura es un principal factor?"))
        self.assertFalse(states("¿Y qué es una cianobacteria?"))
        self.assertFalse(states("que factores afectan la turbidez"))


class SourceNamingTests(unittest.TestCase):
    _QUESTION = (
        "Compará La Balandra entre la hoja Hoja1 de Rio 12-1.xlsx y la hoja base completa fq de base completa fq.xlsx "
        "y la hoja tabla datos."
    )

    @staticmethod
    def _catalog():
        def source(name, *sheets):
            return {
                "source_id": uuid4(),
                "name": name,
                "original_filename": f"{name}.xlsx",
                "datasets": [
                    {"dataset_id": uuid4(), "name": f"{name} / {sheet} / A1:B2", "available": True} for sheet in sheets
                ],
            }

        return [
            source("datos ABSA PL", "Hoja1"),
            source("fito caudal", "Hoja1"),
            source("Tabla completa ciano RdlP", "tabla datos", "Hoja1"),
        ]

    def test_a_sheet_name_that_several_files_share_names_no_source(self) -> None:
        found = RouteQueryService._multiword_sources_in(self._QUESTION, self._catalog(), set())

        self.assertEqual(found, ["tabla datos"])

    def test_a_cued_word_inside_a_written_file_name_is_not_an_alias(self) -> None:
        router = RouteQueryService(None, None)
        found, aliases = asyncio.run(
            router._catalog_references(self._QUESTION, [], self._catalog(), ["base completa fq.xlsx"])
        )

        self.assertNotIn("Tabla completa ciano RdlP", found)
        self.assertEqual(aliases, [])

        question = "¿El archivo base completa fq y la planilla Rio 12-1 dan la misma turbidez?"
        found, aliases = asyncio.run(
            router._catalog_references(question, [], self._catalog(), ["base completa fq", "Rio 12-1"])
        )
        self.assertEqual((found, aliases), ([], []))

    def test_a_value_no_record_holds_is_replaced_by_its_stored_parts(self) -> None:
        stored = {
            "cruda superficial": {("absa", "lugar")},
            "punta lara": {("absa", "localidad"), ("rdlp", "sitios")},
            "pta lara": {("rdlp", "sitios")},
            "balandra": {("rdlp", "sitios")},
            "la balandra": {("rdlp", "sitios")},
            "berisso": {("rdlp", "sitios")},
            "03122004": {("ciano", "codigo_fecha")},
        }
        router = RouteQueryService(None, None)

        async def stored_terms(terms, catalog):
            return {term: stored[term.casefold()] for term in terms if term.casefold() in stored}

        router._stored_terms = stored_terms
        texts = ["agua cruda superficial de Punta Lara", "Pta Lara y Balandra", "La Balandra de Berisso", "Playa de Oro"]
        mentions = [*(DataMention(text=text, kind="value") for text in texts), DataMention(text="03122004", kind="date")]
        result = SimpleNamespace(data_mentions=mentions, conserved_mentions=[], warnings=[])
        asyncio.run(router._without_composite_values(result, []))

        self.assertEqual(
            [(item.kind, item.text) for item in result.data_mentions],
            [
                ("value", "cruda superficial"),
                ("value", "Punta Lara"),
                ("value", "Pta Lara"),
                ("value", "Balandra"),
                ("value", "La Balandra"),
                ("value", "Playa de Oro"),
                ("value", "03122004"),
            ],
        )

    def test_a_named_sheet_narrows_its_named_file(self) -> None:
        catalog = self._catalog()
        source = catalog[2]
        file = [DatasetBinding(dataset_id=item["dataset_id"], source_id=source["source_id"]) for item in source["datasets"]]
        sheet = [file[0]]

        narrowed = _narrowed_to_sheets({"Tabla completa ciano RdlP.xlsx": file, "tabla datos": sheet}, catalog)
        whole = _narrowed_to_sheets({"Tabla completa ciano RdlP.xlsx": file}, catalog)

        self.assertEqual(narrowed, sheet)
        self.assertEqual(whole, file)

    def test_every_missing_named_source_is_listed(self) -> None:
        router = RouteQueryService(None, None)
        missing = ["Rio 12-1.xlsx", "base completa fq.xlsx", "Hoja1"]

        with self.assertRaises(ValueError) as raised:
            router._resolve_sources(missing, self._QUESTION, self._catalog())

        self.assertIn("Rio 12-1.xlsx ni base completa fq.xlsx", str(raised.exception))


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
