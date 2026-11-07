import unittest
from types import SimpleNamespace
from uuid import uuid4

from aquai.application.answering.coverage import text_measurement_notes
from aquai.application.Services.generateResponseService import GenerateResponseService

_NITRATE = "n_no3_mg_n_l_1"
_SPANISH_NOTE = (
    "En Tabla completa ciano RdlP (hoja tabla datos), N-NO3 [mg N L-1] está guardada como texto "
    "y no tiene una versión numérica, así que no pude calcular lo pedido."
)
_ENGLISH_NOTE = (
    "In Tabla completa ciano RdlP (sheet tabla datos), N-NO3 [mg N L-1] is stored as text "
    "without a numeric version, so I could not calculate what you asked for."
)


def _column(sql_name, original_name, sql_type="TEXT", derived_from=None) -> SimpleNamespace:
    return SimpleNamespace(
        sql_name=sql_name,
        original_name=original_name,
        sql_type=sql_type,
        numeric_eligible=sql_type != "TEXT",
        derived_from=derived_from,
    )


def _dataset(*columns) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id=uuid4(),
        name="Tabla completa ciano RdlP / tabla datos / A1:X282",
        provenance=SimpleNamespace(source_name="Tabla completa ciano RdlP.xlsx"),
        columns_metadata=list(columns),
    )


def _text_dataset() -> SimpleNamespace:
    return _dataset(_column(_NITRATE, "N-NO3 [mg N L-1]"))


def _evidence(dataset) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id=dataset.dataset_id,
        variable_matches={"nitrato": [column.sql_name for column in dataset.columns_metadata]},
        value_matches={"Punta Lara": ["sitios"]},
    )


def _state(*datasets, status="insufficient_coverage", statistics=("mean",)) -> SimpleNamespace:
    return SimpleNamespace(
        retrieval=SimpleNamespace(
            candidates=list(datasets),
            evidence=[_evidence(item) for item in datasets],
            dimension_mentions=[],
            confirmed_values=[],
        ),
        planning=SimpleNamespace(
            status=status,
            operations=[],
            context=SimpleNamespace(evidence={"requested_statistics": list(statistics)}),
        ),
        routing=SimpleNamespace(
            data_mentions=[
                SimpleNamespace(text="nitrato", kind="variable"),
                SimpleNamespace(text="Punta Lara", kind="value"),
            ]
        ),
        execution=None,
        python_execution=None,
    )


class TextMeasurementNoteTests(unittest.TestCase):
    def test_a_measurement_stored_only_as_text_explains_the_missing_calculation(self) -> None:
        state = _state(_text_dataset())

        self.assertEqual(text_measurement_notes(state, "es"), [_SPANISH_NOTE])
        self.assertEqual(text_measurement_notes(state, "en"), [_ENGLISH_NOTE])

    def test_a_failed_plan_keeps_the_reason(self) -> None:
        state = _state(_text_dataset(), status="failed")

        self.assertEqual(text_measurement_notes(state, "es"), [_SPANISH_NOTE])

    def test_a_numeric_projection_is_not_a_text_only_measurement(self) -> None:
        projected = _dataset(
            _column(_NITRATE, "N-NO3 [mg N L-1]"),
            _column(
                "n_no3_numeric",
                "N-NO3 [mg N L-1] (representación numérica)",
                "NUMERIC",
                derived_from=_NITRATE,
            ),
        )

        self.assertEqual(text_measurement_notes(_state(projected), "es"), [])

    def test_a_computable_candidate_means_text_is_not_the_cause(self) -> None:
        numeric = _dataset(_column("nitratos_mg_l", "Nitratos (mg/l )", "NUMERIC"))

        self.assertEqual(text_measurement_notes(_state(_text_dataset(), numeric), "es"), [])

    def test_only_unplanned_numeric_statistics_get_the_note(self) -> None:
        dataset = _text_dataset()

        self.assertEqual(text_measurement_notes(_state(dataset, status="plannable"), "es"), [])
        self.assertEqual(text_measurement_notes(_state(dataset, status="needs_clarification"), "es"), [])
        self.assertEqual(text_measurement_notes(_state(dataset, statistics=("count",)), "es"), [])
        self.assertEqual(text_measurement_notes(_state(dataset, statistics=()), "es"), [])

    def test_the_unanswered_reply_publishes_the_note_after_the_coverage_message(self) -> None:
        response = SimpleNamespace(segments=[], answer="")
        validation = SimpleNamespace(
            status="insufficient_coverage",
            issues=[],
            clarification_question=None,
            missing_information=[],
        )

        GenerateResponseService(None)._publish_unanswered(
            response, _state(_text_dataset()), validation, "es"
        )

        lead, note = response.segments
        self.assertTrue(lead.text.startswith("No encontré datos suficientes"))
        self.assertEqual((note.kind, note.text), ("note", _SPANISH_NOTE))
        self.assertIn(_SPANISH_NOTE, response.answer)


if __name__ == "__main__":
    unittest.main()
