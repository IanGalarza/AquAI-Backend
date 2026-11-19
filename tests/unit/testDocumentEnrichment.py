import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

from aquai.application.answering.drafting import (
    ReviewedDraft,
    WriterRequest,
    apply_support,
    backed_citations,
)
from aquai.application.answering.measures import MeasuredResult
from aquai.application.answering.references import (
    comparison_facts,
    local_references,
    verified_references,
)
from aquai.application.answering.verification import DraftEvidence, DraftVerifier
from aquai.application.contracts.answering import DraftSegment
from aquai.application.contracts.retrieval import ReferenceDraft
from aquai.application.Services import retrieveDocumentsService
from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.units import concentration_basis, counts_organisms

_MEASURE = "Se reportaron promedios de 1,37 mg L⁻¹ en condiciones neutrales."
_BY_ROW = """Table 1 Mean values (± standard deviation) for each period.
|  | Neutral | La Niña | El Niño |
| Turbidity (NTU) | 74 ± 62 | 309.4 ± 131.6 | 222 ± 223.1 |
| PO4 3−–P (mg L−1 ) | 0.3 ± 0.11 | 0.28 ± 0.12 | 0.33 ± 0.34 |
| NO3 −–N (mg L−1 ) | 1.37 ± 1.03 | 1.23 ± 0.48 | 0.75 ± 0.51 |"""
_BY_COLUMN = """Table 1 Parameters per site.
| Sample | Turb [NTU] | PO4 [mg P L− 1 ] | NO3 [mg N L− 1 ] |
| BAL-A | 120 ± 33 | 0.25 ± 0.05 | 0.14 ± 0.13 |"""
_PROSE = "Nitrate was measured every month and reached 1.37 in the neutral period, expressed in mg/L."
_SITE_ROWS = """| Sample | Temp [°C] | pH | PO4 [mg P L− 1 ] |
| BAL-A | 17.6 ± 3.3 | 7.6 ±0.3 | 0.25 ± 0.05 |
| PL-A | 16.3 ± 1.9 | 7.7 ±0.5 | 0.31 ± 0.04 |"""
_SITE_LEGEND = "Table 1 Parameters per site (PL: Punta Lara, BAL: Balandra)."
_SITE_TABLE = f"{_SITE_LEGEND}\n{_SITE_ROWS}"
_SITE_CARD = "- Conductividad [µS cm -1 ]: 360 ± 154\n- pH: 8,5 ± 1,1\n- Temperatura [°C]: 22,2 ± 7,7"
_LEAD = "En el estudio de la costa del estuario se analizaron varias especies de cianobacterias."
_DETAIL = "El análisis incluyó variables ambientales como conductividad, pH y temperatura."
_CLOSING = "Las abundancias se transformaron con un logaritmo natural antes del análisis."
_RESULT = "El valor más alto de abundancia fue 458.400 cél/ml."


def _segment(role, *texts) -> DraftSegment:
    return DraftSegment(role=role, text=" ".join(texts), document_ids=["d1"], fact_ids=[], items=[])


def _audited(segments, unsupported) -> ReviewedDraft:
    return apply_support(ReviewedDraft(segments, [], []), unsupported)


def _codes(reviewed) -> list[str]:
    return [problem.code for _, problem in reviewed.rejections]


def _unit_rejected(text, document) -> bool:
    evidence = DraftEvidence(
        documents={"d1": document},
        document_languages={"d1": "en"},
        document_labels={"d1": "Estudio, p. 4"},
        document_contexts={"d1": document},
        question="¿Cuál es el promedio?",
    )
    segment = DraftSegment(role="documented", text=text, document_ids=["d1"], fact_ids=[], items=[])

    return "unsupported_unit" in {item.code for item in DraftVerifier(evidence).check_segment(segment)}


def _fragment(text, section=(), caption=None) -> SimpleNamespace:
    return SimpleNamespace(
        evidence_id="d1", text=text, section=list(section), element_caption=caption, title=None, doubts=[]
    )


def _site_table() -> SimpleNamespace:
    return _fragment(_SITE_TABLE, caption=_SITE_LEGEND)


def _ph() -> MeasuredResult:
    return MeasuredResult(
        evidence_id="e1",
        operation_id="op1",
        field="ph_mean",
        label="pH promedio",
        variable="pH",
        column="pH",
        unit=None,
        unit_status="dimensionless",
        basis=None,
        function="mean",
        values=(Decimal("8.0"),),
        sources=(),
    )


def _proposed(fragment, quote, value) -> tuple:
    draft = ReferenceDraft(
        document_id="d1", variable="pH", quote=quote, low=value, high=value, unit=None, kind="observed"
    )
    references = verified_references([draft], [fragment], [_ph()], "pH", ["Punta Lara"])

    return references, comparison_facts(references, [_ph()], "es")


class _Embeddings:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, texts) -> list:
        self.calls.append(list(texts))

        return [self._vector(text.casefold()) for text in texts]

    @staticmethod
    def _vector(text) -> list:
        if "nitrat" in text:
            return [1.0, 0.0, 0.0]

        return [0.0, 0.0, 1.0] if "fosfato" in text else [0.0, 1.0, 0.0]


def _request_of(documents) -> WriterRequest:
    body = {"documents": [{"id": key, "text": text} for key, text in documents.items()], "facts": []}

    return WriterRequest(record=SimpleNamespace(evidence=body), evidence=DraftEvidence())


def _cited(segment, documents, embeddings=None) -> tuple:
    service = object.__new__(GenerateResponseService)
    service._embeddings = embeddings or _Embeddings()
    response = SimpleNamespace(document_trace=None)
    reviewed = ReviewedDraft([segment], [], [])

    return asyncio.run(service._audited(response, reviewed, _request_of(documents))), response


def _cited_segment(role, documents, *texts) -> DraftSegment:
    return DraftSegment(role=role, text=" ".join(texts), document_ids=list(documents), fact_ids=[], items=[])


def _result(variable, unit) -> SimpleNamespace:
    return SimpleNamespace(variable=variable, unit=unit, values=(1,), names=[variable], lacks_unit=False)


def _request(*results) -> object:
    state = SimpleNamespace(
        question="¿Cuál es el mayor valor?",
        response_language="es",
        python_execution=None,
        validation=SimpleNamespace(status="validated", usable_evidence=["e1"]),
        routing=SimpleNamespace(
            requested_period=None, original_question="¿Cuál es el mayor valor?", data_mentions=[], document_ids=[]
        ),
    )
    with mock.patch.object(retrieveDocumentsService, "measured_results", return_value=list(results)):
        return retrieveDocumentsService._result_request(state)


class AuditedParagraphTests(unittest.TestCase):
    def test_a_paragraph_that_loses_its_presentation_is_not_published(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL)

        reviewed = _audited([paragraph], [(paragraph, _LEAD)])

        self.assertEqual(reviewed.accepted, [])
        self.assertEqual(_codes(reviewed), ["unsupported_claim", "lost_presentation"])
        self.assertEqual(reviewed.rejections[-1][1].detail, _DETAIL)

    def test_the_rest_goes_with_it_however_long_it_is(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL, _CLOSING)

        self.assertEqual(_audited([paragraph], [(paragraph, _LEAD)]).accepted, [])

    def test_a_later_sentence_can_leave_alone(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL, _CLOSING)

        reviewed = _audited([paragraph], [(paragraph, _DETAIL)])

        self.assertEqual([item.text for item in reviewed.accepted], [f"{_LEAD} {_CLOSING}"])
        self.assertEqual(_codes(reviewed), ["unsupported_claim"])

    def test_the_clause_after_a_removed_one_goes_with_it(self) -> None:
        removed = "La acción del viento influye en el nivel de la marea;"
        example = "por ejemplo, los vientos del norte lo bajan."
        paragraph = _segment("documented", _LEAD, removed, example, _CLOSING)

        reviewed = _audited([paragraph], [(paragraph, removed)])

        self.assertEqual([item.text for item in reviewed.accepted], [f"{_LEAD} {_CLOSING}"])

    def test_a_paragraph_without_any_support_is_not_an_orphan(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL)

        reviewed = _audited([paragraph], [(paragraph, _LEAD), (paragraph, _DETAIL)])

        self.assertEqual(reviewed.accepted, [])
        self.assertEqual(_codes(reviewed), ["unsupported_claim", "unsupported_claim"])

    def test_a_single_sentence_is_not_an_orphan(self) -> None:
        paragraph = _segment("documented", _LEAD)

        reviewed = _audited([paragraph], [(paragraph, _LEAD)])

        self.assertEqual(reviewed.accepted, [])
        self.assertEqual(_codes(reviewed), ["unsupported_claim"])

    def test_other_roles_keep_what_is_supported(self) -> None:
        for role in ("relation", "hypothesis"):
            with self.subTest(role=role):
                paragraph = _segment(role, _LEAD, _DETAIL)

                reviewed = _audited([paragraph], [(paragraph, _LEAD)])

                self.assertEqual([item.text for item in reviewed.accepted], [_DETAIL])

    def test_the_other_paragraphs_stay(self) -> None:
        result = _segment("result", _RESULT)
        orphan = _segment("documented", _LEAD, _DETAIL)
        supported = _segment("documented", _CLOSING)

        reviewed = _audited([result, orphan, supported], [(orphan, _LEAD)])

        self.assertEqual([item.text for item in reviewed.accepted], [_RESULT, _CLOSING])

    def test_a_supported_paragraph_is_untouched(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL)

        reviewed = _audited([paragraph], [])

        self.assertIs(reviewed.accepted[0], paragraph)
        self.assertEqual(reviewed.rejections, [])

    def test_a_measured_value_stands_without_its_presentation(self) -> None:
        paragraph = _segment("documented", _LEAD, _MEASURE)

        reviewed = _audited([paragraph], [(paragraph, _LEAD)])

        self.assertEqual([item.text for item in reviewed.accepted], [_MEASURE])
        self.assertEqual(_codes(reviewed), ["unsupported_claim"])

    def test_only_the_measured_sentence_survives_the_lost_presentation(self) -> None:
        paragraph = _segment("documented", _LEAD, _DETAIL, _MEASURE)

        reviewed = _audited([paragraph], [(paragraph, _LEAD)])

        self.assertEqual([item.text for item in reviewed.accepted], [_MEASURE])
        self.assertEqual(_codes(reviewed), ["unsupported_claim", "lost_presentation"])

    def test_a_general_statement_with_its_own_source_stands_without_its_presentation(self) -> None:
        origin = "Según registros fósiles, son los primeros organismos fotosintéticos aparecidos en el planeta."
        diversity = "Presentan una gran diversidad de formas, desde microscópicas hasta visibles a simple vista."
        paragraph = _segment("documented", _LEAD, origin, diversity, _CLOSING)

        reviewed = _audited([paragraph], [(paragraph, _LEAD)])

        self.assertEqual([item.text for item in reviewed.accepted], [f"{origin} {diversity}"])
        self.assertEqual(reviewed.rejections[-1][1].detail, _CLOSING)

    def test_a_count_of_cases_is_not_a_measured_value(self) -> None:
        method = "Se realizó un análisis de redundancia sobre 31 casos con ln(x+1)."
        paragraph = _segment("documented", _LEAD, method)

        self.assertEqual(_audited([paragraph], [(paragraph, _LEAD)]).accepted, [])


class ParagraphCitationTests(unittest.TestCase):
    _NITRATE = "Nitrate averaged 1.37 in the neutral period."
    _PHOSPHATE = "Phosphate reached 0.25 in autumn at the station."
    _SAME_FIGURE = "The neutral period averaged 1.37 across the sites."
    _CLAIM = "El nitrato promedió 1,37 mg L⁻¹ en el período neutral."
    _GENERAL = "El nitrato varía según las condiciones climáticas."

    def test_a_document_that_backs_nothing_stops_being_cited(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._NITRATE}

        reviewed, _ = _cited(_cited_segment("documented", documents, self._CLAIM), documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d2"])

    def test_a_document_with_the_figures_backs_it_however_it_reads(self) -> None:
        documents = {"d1": self._NITRATE, "d2": self._SAME_FIGURE}

        reviewed, _ = _cited(_cited_segment("documented", documents, self._CLAIM), documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d1", "d2"])

    def test_one_figure_in_common_does_not_back_a_sentence(self) -> None:
        documents = {"d1": "Phosphate reached 1.37 in autumn.", "d2": "Nitrate averaged 1.37 and 1.23 in each period."}
        claim = "El nitrato promedió 1,37 mg L⁻¹ y 1,23 mg L⁻¹."

        reviewed, _ = _cited(_cited_segment("documented", documents, claim), documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d2"])

    def test_a_sentence_without_figures_is_backed_by_resemblance(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._NITRATE}

        reviewed, _ = _cited(_cited_segment("documented", documents, self._GENERAL), documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d2"])

    def test_each_sentence_can_have_its_own_document(self) -> None:
        documents = {"d1": self._NITRATE, "d2": "Nitrate depends on the climate."}
        paragraph = _cited_segment("documented", documents, self._CLAIM, self._GENERAL)

        reviewed, _ = _cited(paragraph, documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d1", "d2"])

    def test_a_document_cited_for_a_dropped_sentence_goes_with_it(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._NITRATE}
        lead = "En la estación se midió fosfato durante el otoño."
        paragraph = _cited_segment("documented", documents, lead, self._CLAIM)

        reviewed, _ = _cited(paragraph, documents)

        self.assertEqual(reviewed.accepted[0].text, self._CLAIM)
        self.assertEqual(reviewed.accepted[0].document_ids, ["d2"])

    def test_the_trace_says_which_citation_left(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._NITRATE}

        _, response = _cited(_cited_segment("documented", documents, self._CLAIM), documents)

        rejected = [item for item in response.document_trace.rejected_segments if item["code"] == "unbacked_citation"]
        self.assertEqual([item["detail"] for item in rejected], ["d1"])

    def test_a_paragraph_nothing_backs_keeps_its_citations(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._PHOSPHATE}
        paragraph = _cited_segment("documented", documents, _LEAD)

        reviewed = backed_citations(ReviewedDraft([paragraph], [], []), _request_of(documents), lambda *_: False)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d1", "d2"])
        self.assertEqual(reviewed.rejections, [])

    def test_other_roles_keep_their_citations(self) -> None:
        documents = {"d1": self._PHOSPHATE, "d2": self._NITRATE}

        reviewed, _ = _cited(_cited_segment("relation", documents, self._CLAIM), documents)

        self.assertEqual(reviewed.accepted[0].document_ids, ["d1", "d2"])

    def test_a_sentence_rewritten_by_the_audit_is_compared_as_published(self) -> None:
        documents = {"d2": self._NITRATE}
        embeddings = _Embeddings()
        lead = "En la estación se midió fosfato durante el otoño."
        connected = "Además, el nitrato promedió 1,37 mg L⁻¹ en el período neutral."
        paragraph = _cited_segment("documented", documents, lead, connected)

        reviewed, _ = _cited(paragraph, documents, embeddings)

        self.assertEqual(reviewed.accepted[0].text, self._CLAIM)
        self.assertEqual(reviewed.accepted[0].document_ids, ["d2"])
        self.assertEqual(len(embeddings.calls), 2)


class ReferencePlaceTests(unittest.TestCase):
    def test_a_row_of_another_site_is_not_the_same_place(self) -> None:
        references, contrasts = _proposed(_site_table(), "7.6 ±0.3", 7.6)

        self.assertEqual([item.same_place for item in references], [False])
        self.assertEqual(contrasts, [])

    def test_a_whole_row_of_another_site_is_not_the_same_place(self) -> None:
        row = "| BAL-A | 17.6 ± 3.3 | 7.6 ±0.3 | 0.25 ± 0.05 |"

        references, contrasts = _proposed(_site_table(), row, 7.6)

        self.assertEqual([item.same_place for item in references], [False])
        self.assertEqual(contrasts, [])

    def test_a_row_of_the_site_is_the_same_place(self) -> None:
        references, contrasts = _proposed(_site_table(), "7.7 ±0.5", 7.7)

        self.assertEqual([item.same_place for item in references], [True])
        self.assertEqual(len(contrasts), 1)

    def test_a_legend_kept_only_in_the_caption_also_defines_the_codes(self) -> None:
        fragment = _fragment(_SITE_ROWS, caption=_SITE_LEGEND)

        foreign, _ = _proposed(fragment, "7.6 ±0.3", 7.6)
        own, _ = _proposed(fragment, "7.7 ±0.5", 7.7)

        self.assertEqual([item.same_place for item in [*foreign, *own]], [False, True])

    def test_a_table_without_site_codes_trusts_its_heading(self) -> None:
        table = "| Season | pH |\n| Autumn | 7.7 ±0.5 |\n| Winter | 8.1 ±0.2 |"

        references, _ = _proposed(_fragment(table, section=["Punta Lara"]), "7.7 ±0.5", 7.7)

        self.assertEqual([item.same_place for item in references], [True])

    def test_a_site_card_keeps_its_contrast(self) -> None:
        fragment = _fragment(_SITE_CARD, section=["Punta Lara (PL)"])

        references = local_references([fragment], [_ph()], "pH", ["Punta Lara"])

        self.assertEqual([(item.low, item.same_place) for item in references], [(8.5, True)])
        self.assertEqual(len(comparison_facts(references, [_ph()], "es")), 1)

    def test_a_quote_read_without_accents_keeps_the_same_place(self) -> None:
        fragment = _fragment("- Oxígeno disuelto [mg L -1 ]: 9,7 ± 2,6", section=["Punta Lara (PL)"])
        result = MeasuredResult(
            evidence_id="e1",
            operation_id="op1",
            field="oxygen_mean",
            label="Oxígeno disuelto promedio",
            variable="Oxígeno disuelto",
            column="Oxígeno disuelto (mg/L)",
            unit="mg/L",
            unit_status="documented",
            basis=None,
            function="mean",
            values=(Decimal("9.0"),),
            sources=(),
        )

        references = local_references([fragment], [result], "oxígeno", ["Punta Lara"])

        self.assertEqual([(item.low, item.same_place) for item in references], [(9.7, True)])

    def test_a_place_nobody_names_is_not_the_same_place(self) -> None:
        references, _ = _proposed(_fragment("| Sample | pH |\n| Lab | 7.7 ±0.5 |"), "7.7 ±0.5", 7.7)

        self.assertEqual([item.same_place for item in references], [False])


class ReferenceCellTests(unittest.TestCase):
    def test_a_cell_of_another_column_is_not_the_variable(self) -> None:
        self.assertEqual(_proposed(_site_table(), "0.31 ± 0.04", 0.31)[0], [])

    def test_a_cell_of_the_column_of_the_variable_is_accepted(self) -> None:
        self.assertEqual(len(_proposed(_site_table(), "7.7 ±0.5", 7.7)[0]), 1)

    def test_a_cell_of_the_row_of_the_variable_is_accepted(self) -> None:
        table = "| | Neutral | La Niña |\n| Temperature (°C) | 26.1 ± 6.3 | 25.4 ± 5 |\n| pH | 8.2 ± 0.6 | 7.9 ± 0.4 |"

        self.assertEqual(len(_proposed(_fragment(table), "7.9 ± 0.4", 7.9)[0]), 1)

    def test_every_row_with_the_cell_counts(self) -> None:
        table = "| Sample | Temp [°C] | pH |\n| A | 7.7 ±0.5 | 8.0 ±0.1 |\n| B | 16.3 ± 1.9 | 7.7 ±0.5 |"

        self.assertEqual(len(_proposed(_fragment(table), "7.7 ±0.5", 7.7)[0]), 1)

    def test_a_quote_outside_the_table_needs_no_cell(self) -> None:
        text = f"The pH was 8.2 in the coastal strip.\n{_SITE_ROWS}"

        self.assertEqual(len(_proposed(_fragment(text), "The pH was 8.2", 8.2)[0]), 1)


class TableUnitTests(unittest.TestCase):
    def test_the_unit_of_a_row_label_authorizes_its_values(self) -> None:
        self.assertFalse(_unit_rejected("Se reportaron 1,37 mg L⁻¹ en el período neutral.", _BY_ROW))

    def test_the_basis_of_a_row_label_authorizes_its_values(self) -> None:
        self.assertFalse(_unit_rejected("Se reportaron 1,37 mg N L⁻¹ en el período neutral.", _BY_ROW))

    def test_another_basis_is_not_the_one_of_the_row(self) -> None:
        self.assertTrue(_unit_rejected("Se reportaron 1,37 mg P L⁻¹ en el período neutral.", _BY_ROW))

    def test_another_unit_is_not_the_one_of_the_row(self) -> None:
        self.assertTrue(_unit_rejected("Se reportaron 74 mg L⁻¹ en el período neutral.", _BY_ROW))

    def test_the_basis_of_a_column_header_authorizes_its_values(self) -> None:
        self.assertFalse(_unit_rejected("Se reportaron 0,14 mg N L⁻¹ en otoño.", _BY_COLUMN))

    def test_the_column_of_another_element_does_not_authorize(self) -> None:
        self.assertTrue(_unit_rejected("Se reportaron 0,25 mg N L⁻¹ en otoño.", _BY_COLUMN))

    def test_prose_still_needs_the_basis_next_to_the_number(self) -> None:
        self.assertTrue(_unit_rejected("Se reportaron 1,37 mg N L⁻¹ en el período neutral.", _PROSE))

    def test_prose_without_basis_keeps_working(self) -> None:
        self.assertFalse(_unit_rejected("Se reportaron 1,37 mg/L en el período neutral.", _PROSE))

    def test_the_name_of_a_nutrient_says_its_basis_in_any_dash(self) -> None:
        for label, basis in (
            ("NO3 −–N (mg L−1 )", "N"),
            ("PO4 3−–P (mg L−1 )", "P"),
            ("NH4 + -N (mg L−1 )", "N"),
            ("NO3 - -N (mg L -1 )", "N"),
            ("PO4 3- -P (mg L -1 )", "P"),
            ("NO3-N", "N"),
            ("N-NO3", "N"),
            ("P–PO4", "P"),
        ):
            with self.subTest(label=label):
                self.assertEqual(concentration_basis(label, None), basis)

    def test_a_name_without_element_has_no_basis(self) -> None:
        for label in ("NO3 (mg/L)", "Nitratos (mg/L)", "NO3 - Nitrate", "PO4 - Phosphate"):
            with self.subTest(label=label):
                self.assertIsNone(concentration_basis(label, None))


class OrganismCountTests(unittest.TestCase):
    def test_counts_of_organisms_in_any_spelling(self) -> None:
        for unit in ("cél/ml", "cel/mL", "células/ml", "cells/mL", "Cel mL-1", "ind/ml", "colonias/ml"):
            with self.subTest(unit=unit):
                self.assertTrue(counts_organisms(unit))

    def test_other_quantities_are_not_counts_of_organisms(self) -> None:
        for unit in ("mg/L", "mg N L-1", "µg/L", "NTU", "°C", "µS/cm", "%", "UFC/100 ml", "NMP/100ml", "", None):
            with self.subTest(unit=unit):
                self.assertFalse(counts_organisms(unit))


class EnrichmentSearchTests(unittest.TestCase):
    def test_a_count_of_organisms_is_searched_with_its_unit(self) -> None:
        request = _request(_result("Abundancia absoluta", "cél/ml"))

        self.assertEqual(request.requests[0], "Abundancia absoluta (cél/ml)")
        self.assertTrue(all(item.startswith("Abundancia absoluta (cél/ml)") for item in request.requests))
        self.assertTrue(request.focus.startswith("Abundancia absoluta (cél/ml):"))

    def test_a_concentration_keeps_its_plain_name(self) -> None:
        request = _request(_result("nitrato", "mg N/L"))

        self.assertEqual(request.requests[0], "nitrato")
        self.assertFalse([item for item in request.requests if "mg" in item])
        self.assertTrue(request.focus.startswith("nitrato:"))

    def test_a_faecal_indicator_keeps_its_plain_name(self) -> None:
        self.assertEqual(_request(_result("Coliformes Fecales", "UFC/100 ml")).requests[0], "Coliformes Fecales")

    def test_a_variable_without_unit_keeps_its_plain_name(self) -> None:
        self.assertEqual(_request(_result("Sitio", None)).requests[0], "Sitio")

    def test_the_subject_stays_the_name_of_the_variable(self) -> None:
        self.assertEqual(_request(_result("Abundancia absoluta", "cél/ml")).subject, ["Abundancia absoluta"])

    def test_the_same_count_is_searched_once(self) -> None:
        results = (_result("Abundancia absoluta", "cél/ml"), _result("Abundancia absoluta", "cél/ml"))

        self.assertEqual(len(_request(*results).requests), 3)


class ContiguousFragmentTests(unittest.TestCase):
    _SECTION = ["CHAPTER 8 Algae and cyanobacteria", "8.7 Guideline values"]

    @staticmethod
    def _chunk(position, path, doubts=()):
        return {
            "chunk_id": f"c{position}",
            "document_id": "who",
            "position": position,
            "section_path": path,
            "doubts": list(doubts),
            "kind": "text",
            "text": "A guideline level of cyanobacterial cells per millilitre.",
            "token_count": 300,
        }

    def _bridged(self, chosen, stored):
        queries = SimpleNamespace(
            chunks_at=lambda document_id, positions: {
                item["chunk_id"]: item for item in stored if item["position"] in positions
            }
        )
        service = retrieveDocumentsService.RetrieveDocumentsService(None, queries, None, fragments=6)
        return [item["position"] for item in asyncio.run(service._bridged(chosen))]

    def test_the_fragment_between_two_chosen_of_one_section_is_added(self) -> None:
        chosen = [self._chunk(417, self._SECTION), self._chunk(419, [*self._SECTION, "8.7.2 Moderate"])]
        stored = [self._chunk(418, [*self._SECTION, "8.7.1 Low"])]

        self.assertEqual(self._bridged(chosen, stored), [417, 418, 419])

    def test_a_doubtful_or_foreign_fragment_is_not_added(self) -> None:
        chosen = [self._chunk(417, self._SECTION), self._chunk(419, self._SECTION)]
        doubtful = [self._chunk(418, self._SECTION, doubts=["table_uncertain"])]
        foreign = [self._chunk(418, ["CHAPTER 9 Other"])]

        self.assertEqual(self._bridged(chosen, doubtful), [417, 419])
        self.assertEqual(self._bridged(chosen, foreign), [417, 419])


if __name__ == "__main__":
    unittest.main()
