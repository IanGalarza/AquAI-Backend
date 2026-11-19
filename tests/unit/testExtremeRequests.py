import unittest
from datetime import date

from aquai.application.contracts.routing import DataMention
from aquai.application.matching import period_ranges, reference_operator
from aquai.application.planning.context import asks_period_span, extreme_requested, requested_statistics
from aquai.application.Rules.vocabulary import TREND_CUES

_EXTREMES = (
    "¿Cuál es la mayor abundancia absoluta (cél/ml)?",
    "¿Cuál fue el menor pH en Punta Lara?",
    "¿Qué sitio tuvo mayor turbiedad?",
    "¿Cuáles son las mayores concentraciones de nitrato?",
    "¿En qué mes fue menor?",
    "¿Cuál es la mayor de las concentraciones de nitrato?",
    "¿Cuál es la mayor turbiedad en sitios con pH mayor a 8?",
    "¿Cuál es el máximo de pH?",
    "¿Cuál es el máximo de 2021?",
    "Mostrame un máximo de 5 registros con la mayor turbiedad",
    "¿Cuál fue el valor más alto de turbiedad?",
    "What is the largest chlorophyll value?",
    "Which site has the smallest pH?",
)
_NOT_EXTREMES = (
    "¿Qué mediciones tienen turbiedad mayor a 100?",
    "Sitios con pH mayor que 8",
    "Valores mayores de 5",
    "Valores menores de 0,5 mg/L",
    "Mediciones con pH menor o igual a 6",
    "¿Qué registros superan el valor mayor al promedio?",
    "¿La mayor parte de los sitios tiene datos de pH?",
    "¿El pH en Punta Lara es mayor que en Balandra?",
    "¿Cuántos registros de pH hay?",
    "Mostrame el pH de mayor a menor",
    "Listá el pH de menor a mayor en Punta Lara",
    "Show the pH sorted from highest to lowest",
    "Values greater than 5",
    "Mostrame un máximo de 5 registros de turbiedad",
    "Dame como máximo 10 filas de pH",
    "Show a maximum of 5 records",
)


class ExtremeRequestTests(unittest.TestCase):
    def test_superlatives_and_extreme_words_ask_for_the_extreme(self) -> None:
        for question in _EXTREMES:
            with self.subTest(question=question):
                self.assertTrue(extreme_requested(question))

    def test_thresholds_comparisons_and_full_orders_do_not(self) -> None:
        for question in _NOT_EXTREMES:
            with self.subTest(question=question):
                self.assertFalse(extreme_requested(question))


class RequestedWordingTests(unittest.TestCase):
    def test_a_day_written_with_the_month_name_is_that_day(self) -> None:
        day = (date(2021, 9, 22), date(2021, 9, 22))
        for text in ("el 22 de septiembre de 2021", "22 sep. 2021", "September 22nd, 2021", "22/09/2021"):
            with self.subTest(text=text):
                self.assertEqual(period_ranges(text), [day])

        self.assertEqual(period_ranges("entre el 12 y el 28 de enero de 2022"), [(date(2022, 1, 12), date(2022, 1, 28))])
        self.assertEqual(period_ranges("septiembre de 2021"), [(date(2021, 9, 1), date(2021, 9, 30))])

    def test_period_and_trend_requests_are_read_in_different_wordings(self) -> None:
        for text in ("¿Qué fechas abarcan los datos?", "¿Desde cuándo hay registros?", "Indicá promedios y períodos"):
            with self.subTest(text=text):
                self.assertTrue(asks_period_span(text, []))
        self.assertTrue(asks_period_span("promedio", [DataMention(text="qué fechas abarcan", kind="date")]))
        self.assertFalse(asks_period_span("promedio", [DataMention(text="el último dato", kind="date")]))
        self.assertFalse(asks_period_span("¿Cuál fue el promedio en el período 2012-2013?", []))
        for text in ("¿la turbidez fue cambiando con los años?", "¿Cambió el pH en las últimas décadas?", "¿hay tendencia?"):
            with self.subTest(text=text):
                self.assertIsNotNone(TREND_CUES.search(text))
        self.assertIsNone(TREND_CUES.search("¿Cuál fue la turbidez más baja en los años 2012 y 2013?"))

    def test_a_reference_reached_or_exceeded_includes_the_limit(self) -> None:
        self.assertEqual(reference_operator("¿Cuántos alcanzaron o superaron el nivel guía?"), "ge")
        self.assertEqual(reference_operator("¿Cuántos superaron el nivel guía?"), "gt")
        self.assertEqual(reference_operator("How many were at least the guideline or above it?"), "ge")
        self.assertEqual(reference_operator("¿Cuántos quedaron por debajo del límite?"), "lt")

    def test_how_many_per_unit_asks_for_a_concentration_not_a_count(self) -> None:
        self.assertEqual(requested_statistics("¿Cuántas cianobacterias tóxicas por mililitro informa BASE FQ?"), [])
        self.assertEqual(requested_statistics("¿Cuántas mediciones hay por sitio?"), ["count"])


if __name__ == "__main__":
    unittest.main()
