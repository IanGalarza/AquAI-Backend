import unittest

from aquai.application.planning.context import extreme_requested

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


if __name__ == "__main__":
    unittest.main()
