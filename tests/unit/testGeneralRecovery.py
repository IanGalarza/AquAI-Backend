import asyncio
import unittest
from types import SimpleNamespace
from uuid import uuid4

from aquai.application.answering.blocked import request_gaps
from aquai.application.answering.drafting import ReviewedDraft, apply_support, support_phrases
from aquai.application.contracts.answering import DraftSegment
from aquai.application.contracts.planning import RequestedThreshold, ValidationIssue
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import PlannerValidationError
from aquai.application.matching import coordinated_parts, numeric_comparisons, reference_operator
from aquai.application.planning.context import requested_statistics
from aquai.application.planning.validation import _accepted_gaps, _line_axis_problem
from aquai.application.Services.generateResponseService import GenerateResponseService
from aquai.application.Services.planQueryService import PlanQueryService, _organism_counts, _VerifiedPlan
from aquai.application.Services.routeQueryService import RouteQueryService
from tests.unit.testExtremeReads import _NUMERIC, _RAW, _READ, _SITE, _TURBIDITY, _YEAR, _dataset, _plan, _ref

_CHART = ValidationIssue(code="missing_requested_chart", severity="clarification", message="Falta el gráfico.")
_MEAN = ValidationIssue(
    code="missing_requested_statistic", severity="clarification", message="Falta el promedio.", subject="mean"
)
_PERIOD = ValidationIssue(code="missing_requested_period", severity="clarification", message="Falta el período.")


class _Provider:
    context_budget_bytes = 100_000

    async def propose(self, context):
        return SimpleNamespace(decision=None, model="scripted", input_tokens=1, output_tokens=1)


def _proposal(*outcomes):
    calls = []

    async def verify(decision):
        outcome = outcomes[len(calls)]
        calls.append(decision)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    context = SimpleNamespace(evidence={}, sent_to_model=False)
    return asyncio.run(PlanQueryService(_Provider(), None)._propose(context, verify))


def _verified(*issues) -> _VerifiedPlan:
    return _VerifiedPlan(decision=None, selected=[], operations=[], issues=list(issues), resolutions=[])


class PartialPublicationTests(unittest.TestCase):
    def test_a_plan_missing_only_the_requested_chart_publishes_what_it_calculated(self) -> None:
        proposal = _proposal(_verified(_CHART), _verified(_CHART))

        self.assertIsNone(proposal.error)
        self.assertEqual([(item.code, item.severity) for item in proposal.verified.issues], [("missing_requested_chart", "warning")])

    def test_an_answerable_first_attempt_survives_a_rejected_retry(self) -> None:
        proposal = _proposal(_verified(_MEAN), PlannerValidationError("Contrato roto."))

        self.assertIsNone(proposal.error)
        self.assertEqual(proposal.verified.issues[0].severity, "warning")
        self.assertEqual(proposal.attempts, 2)

    def test_another_defect_still_blocks_the_plan(self) -> None:
        verified = _verified(_CHART, _PERIOD)

        self.assertIsNone(PlanQueryService._answerable(verified, verified.issues))
        proposal = _proposal(verified, verified)
        self.assertEqual({item.severity for item in proposal.verified.issues}, {"clarification"})

    def test_the_user_is_told_what_was_left_out(self) -> None:
        threshold = ValidationIssue(
            code="missing_requested_threshold", severity="warning", message="x", subject="mayor o igual que 20000"
        )
        issues = [_CHART.model_copy(update={"severity": "warning"}), _MEAN.model_copy(update={"severity": "warning"}), threshold]
        state = SimpleNamespace(planning=SimpleNamespace(validation_issues=issues))

        notes = request_gaps(state, "es")

        self.assertEqual(len(notes), 3)
        self.assertIn("gráfico", notes[0])
        self.assertIn("No pude calcular el promedio", notes[1])
        self.assertIn("«mayor o igual que 20000»", notes[2])

    def test_execution_does_not_ask_again_for_what_the_planner_accepted_as_missing(self) -> None:
        plan = SimpleNamespace(validation_issues=[_CHART.model_copy(update={"severity": "warning"}), _MEAN, _PERIOD])

        self.assertEqual(_accepted_gaps(plan), {"missing_requested_chart"})

    def test_unmet_parts_still_asking_the_model_are_not_notes(self) -> None:
        state = SimpleNamespace(planning=SimpleNamespace(validation_issues=[_CHART]))

        self.assertEqual(request_gaps(state, "es"), [])


class FailedChartTests(unittest.TestCase):
    def test_a_chart_that_fails_verification_publishes_the_data_it_would_draw(self) -> None:
        dataset = _dataset(_TURBIDITY, "sitio")
        site = {"kind": "result", "step_id": _READ, "field": "sitio"}
        mean = {
            "id": "mean",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "group_by": [{"field": site, "output_name": "sitio"}],
            "calculations": [
                {"function": "mean", "inputs": [{"kind": "result", "step_id": _READ, "field": _TURBIDITY}], "output_name": "promedio"}
            ],
            "purpose": "Promediar por sitio.",
        }
        chart = {
            "id": "chart",
            "tool": "python",
            "operation": "visualize",
            "depends_on": ["mean"],
            "visualization": {
                "title": "Promedio por sitio",
                "x": {"kind": "result", "step_id": "mean", "field": "sitio"},
                "y": {"kind": "result", "step_id": "mean", "field": "promedio"},
                "x_label": "Sitio",
                "y_label": "Turbiedad",
                "chart_type": "line",
            },
            "purpose": "Graficar.",
        }
        plan = _plan(dataset, [_TURBIDITY, "sitio"], [mean, chart], output="chart")

        reduced, issues = PlanQueryService._verifiable_branches(plan, [dataset], [])

        self.assertEqual(reduced.output_steps, ["mean"])
        self.assertEqual([(item.code, item.severity) for item in issues], [("missing_requested_chart", "warning")])
        self.assertIn("chart_type bar", issues[0].message)


class ThresholdTargetTests(unittest.TestCase):
    def _counted(self):
        dataset = _dataset(_RAW, _NUMERIC, _YEAR)
        count = {
            "id": "counted",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "calculations": [{"function": "count", "output_name": "total"}],
            "purpose": "Contar.",
        }
        return dataset, _plan(dataset, [_NUMERIC, _YEAR], [count], output="counted")

    def test_an_omitted_threshold_lands_on_the_requested_variable_not_the_year(self) -> None:
        dataset, plan = self._counted()
        threshold = RequestedThreshold(
            text="mayor o igual que 20000", operator="ge", value=20000, compares_statistic=False
        )

        for variables in ({dataset.dataset_id: {_RAW}}, None):
            with self.subTest(variables=variables):
                repaired = PlanQueryService._omitted_thresholds(plan, [dataset], [threshold], variables)
                filters = [(item.field.column, item.operator, item.values) for item in repaired.steps[0].filters]

                self.assertEqual(filters, [(_NUMERIC, "ge", [20000])])


class FeedbackTests(unittest.TestCase):
    def test_a_line_over_a_category_suggests_bars_or_series(self) -> None:
        self.assertIn("chart_type bar", _line_axis_problem(SimpleNamespace(name="sitio")))
        self.assertIn("fechas como texto", _line_axis_problem(SimpleNamespace(name="fecha")))


class ComparisonWordingTests(unittest.TestCase):
    def test_comparison_verbs_say_their_direction(self) -> None:
        for question, expected in (
            ("¿Cuántos recuentos llegaron o pasaron los niveles guía de la OMS?", "ge"),
            ("¿Cuántos recuentos alcanzaron o superaron los niveles guía?", "ge"),
            ("¿Cuántos sobrepasaron el valor guía?", "gt"),
            ("¿Cuántas muestras rebasaron el límite recomendado?", "gt"),
            ("¿Cuántos valores no alcanzaron el nivel guía?", "lt"),
            ("¿Cuántos quedaron por debajo del valor guía?", "lt"),
            ("¿Qué pasó con los niveles guía de la OMS?", None),
        ):
            with self.subTest(question=question):
                self.assertEqual(reference_operator(question), expected)

    def test_numeric_thresholds_accept_the_same_verbs(self) -> None:
        for text, expected in (
            ("pasan de 100 UNT", ("gt", 100)),
            ("pasaron los 20.000 cél/mL", ("gt", 20000)),
            ("no llegan a 100 UNT", ("lt", 100)),
        ):
            with self.subTest(text=text):
                found = numeric_comparisons(text)
                self.assertEqual((found[0].operator, found[0].value), expected)
        self.assertEqual(numeric_comparisons("pasaron 3 días entre muestreos"), [])


class CountRequestTests(unittest.TestCase):
    def test_asking_how_many_organisms_a_source_reports_is_not_a_record_count(self) -> None:
        question = "¿Qué turbiedad tiene ABSA y cuántas cianobacterias potencialmente tóxicas informa la base FQ?"

        self.assertEqual(requested_statistics(question, ["cianobacterias potencialmente tóxicas"]), [])
        self.assertEqual(requested_statistics(question), ["count"])

    def test_counting_records_of_a_variable_is_still_a_count(self) -> None:
        question = "¿Cuántos recuentos de cianos potencialmente tóxicas llegaron a los niveles guía?"

        self.assertEqual(requested_statistics(question, ["cianos potencialmente tóxicas"]), ["count"])

    def test_variables_measured_as_organisms_per_volume_are_counts(self) -> None:
        dataset = _dataset(_NUMERIC, _TURBIDITY)
        evidence = [
            SimpleNamespace(
                dataset_id=dataset.dataset_id,
                variable_matches={"cianobacterias": [_NUMERIC], "turbiedad": [_TURBIDITY]},
            )
        ]

        self.assertEqual(_organism_counts(evidence, [dataset]), ["cianobacterias"])


class SourceNameTests(unittest.TestCase):
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
            source("Tabla completa ciano RdlP", "tabla datos"),
            source("BaseCiano_PARA BELEN_5MAR2024 (2)", "BASE FQ", "BASE CIANO"),
        ]

    def test_a_file_name_written_without_its_extension_is_reported_missing(self) -> None:
        question = "¿Coinciden los valores entre la planilla Rio 12-1 28-1 2-2 y la base completa fq?"
        result = SimpleNamespace(warnings=[])
        explicit = RouteQueryService._explicit_mentions(
            ["Rio 12-1 28-1 2-2.xlsx", "base completa fq.xlsx"], question, [], result
        )

        self.assertEqual(explicit, ["Rio 12-1 28-1 2-2.xlsx", "base completa fq.xlsx"])
        with self.assertRaises(ValueError) as raised:
            RouteQueryService(None, None)._resolve_sources(explicit, question, self._catalog())
        self.assertIn("Rio 12-1 28-1 2-2.xlsx ni base completa fq.xlsx", str(raised.exception))

    def test_a_value_naming_a_source_acronym_refers_to_the_source(self) -> None:
        router = RouteQueryService(None, None)

        async def stored_anywhere(value, catalog):
            return False

        router._stored_anywhere = stored_anywhere
        question = "comparame la turbidez de punta lara entre el excel de ABSA y la tabla datos de RdlP"
        mentions = [DataMention(text="excel de ABSA", kind="value"), DataMention(text="punta lara", kind="value")]

        found, aliases = asyncio.run(router._catalog_references(question, mentions, self._catalog(), []))

        self.assertIn("datos ABSA PL", found)
        self.assertEqual(aliases, ["excel de ABSA"])

    def test_a_document_question_that_restricts_named_data_is_answered_with_data(self) -> None:
        router = RouteQueryService(None, None)

        async def stored_anywhere(value, catalog):
            return False

        router._stored_anywhere = stored_anywhere
        question = "en 2005, en la base fq, ¿hubo recuentos de 20.000 cél/mL o más con el agua a menos de 25 grados?"
        decision = SimpleNamespace(source_mentions=[])
        restricted = [
            DataMention(text="cianos tóxicas", kind="variable"),
            DataMention(text="2005", kind="date"),
            DataMention(text="menos de 25", kind="threshold"),
        ]

        self.assertTrue(asyncio.run(router._asks_named_data(question, restricted, self._catalog(), decision)))
        self.assertFalse(asyncio.run(router._asks_named_data(question, restricted[:1], self._catalog(), decision)))


class DocumentAuditTests(unittest.TestCase):
    _LINE = (
        "Cyanobacterial scum formation in areas where whole-body contact and/or risk of ingestion/aspiration occur "
        "(potential for acute poisoning; long-term illness)"
    )

    def test_list_items_are_compared_with_the_phrases_they_translate(self) -> None:
        phrases = support_phrases([self._LINE])
        item, sentence = "contacto total del cuerpo", "Una oración que traduce sólo una frase."
        segment = SimpleNamespace(items=[item])
        close, phrase = [1.0, 0.0], [0.8, 0.6]
        vectors = {item: close, sentence: close, self._LINE: [0.0, 1.0], **{text: phrase for text in phrases}}
        claims = [(segment, item), (segment, sentence)]

        request = SimpleNamespace(record=SimpleNamespace(evidence={"documents": []}))
        unsupported = GenerateResponseService._unsupported(
            vectors, claims, {id(segment): [self._LINE]}, request, {id(segment): phrases}
        )

        self.assertIn("potential for acute poisoning", phrases)
        self.assertIn("risk of ingestion/aspiration occur", phrases)
        self.assertEqual(unsupported, [(segment, sentence)])

    def test_a_documentary_block_left_out_is_counted(self) -> None:
        listed = DraftSegment(role="documented", text="Los niveles son:", document_ids=["d1"], fact_ids=[], items=["A", "B"])
        kept = DraftSegment(role="documented", text="Se describe un nivel.", document_ids=["d1"], fact_ids=[], items=[])

        reviewed = apply_support(ReviewedDraft([listed, kept], [], []), [(listed, "A")])

        self.assertEqual(reviewed.dropped, 1)
        self.assertEqual(apply_support(ReviewedDraft([kept], [], []), []).dropped, 0)


class GroupRankingTests(unittest.TestCase):
    def test_a_ranked_mean_by_group_carries_how_many_values_it_has(self) -> None:
        dataset = _dataset(_TURBIDITY, "sitio")
        field = {"kind": "result", "step_id": _READ, "field": _TURBIDITY}
        mean = {
            "id": "mean",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "group_by": [{"field": {"kind": "result", "step_id": _READ, "field": "sitio"}, "output_name": "sitio"}],
            "calculations": [{"function": "mean", "inputs": [field], "output_name": "promedio"}],
            "purpose": "Promediar.",
        }
        top = {
            "id": "top",
            "tool": "sql",
            "operation": "sort",
            "depends_on": ["mean"],
            "order_by": [{"field": {"kind": "result", "step_id": "mean", "field": "promedio"}, "direction": "desc"}],
            "limit": 1,
            "purpose": "Ordenar.",
        }
        plan = _plan(dataset, [_TURBIDITY, "sitio"], [mean, top], output="top")

        repaired = PlanQueryService._ranked_group_counts(plan)
        calculations = [(item.function, item.output_name) for item in repaired.steps[1].calculations]

        self.assertEqual(calculations, [("mean", "promedio"), ("count", "cantidad_de_valores")])
        self.assertIs(PlanQueryService._ranked_group_counts(repaired), repaired)


class DegenerateComparisonTests(unittest.TestCase):
    def test_a_difference_between_two_copies_of_the_same_mean_is_dropped(self) -> None:
        dataset = _dataset(_TURBIDITY, "sitio")
        field = {"kind": "result", "step_id": _READ, "field": _TURBIDITY}
        means = {
            "id": "means",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "group_by": [{"field": {"kind": "result", "step_id": _READ, "field": "sitio"}, "output_name": "sitio"}],
            "calculations": [
                {"function": "mean", "inputs": [field], "output_name": "media_a"},
                {"function": "mean", "inputs": [field], "output_name": "media_b"},
            ],
            "purpose": "Promediar.",
        }
        difference = {
            "id": "difference",
            "tool": "python",
            "operation": "calculate",
            "depends_on": ["means"],
            "calculations": [
                {
                    "function": "difference",
                    "inputs": [
                        {"kind": "result", "step_id": "means", "field": "media_a"},
                        {"kind": "result", "step_id": "means", "field": "media_b"},
                    ],
                    "output_name": "diferencia",
                }
            ],
            "purpose": "Restar.",
        }
        plan = _plan(dataset, [_TURBIDITY, "sitio"], [means, difference], output="difference")

        repaired = PlanQueryService._degenerate_comparisons(plan)

        self.assertEqual(repaired.output_steps, ["means"])
        self.assertEqual([item.output_name for item in repaired.steps[1].calculations], ["media_a"])
        self.assertEqual([step.id for step in repaired.steps], [_READ, "means"])


class ThresholdJustificationTests(unittest.TestCase):
    @staticmethod
    def _filtered(dataset, operator, value, tail=()):
        bound = {"field": _ref(dataset, _TURBIDITY), "operator": operator, "values": [value]}
        return _plan(dataset, [_TURBIDITY, _YEAR], list(tail), filters=[bound], output=tail[-1]["id"] if tail else _READ)

    @staticmethod
    def _requested(reference=None, thresholds=(), numbers=()):
        from aquai.application.Services.planQueryService import _UNRESTRICTED

        return _UNRESTRICTED._replace(
            reference_request=reference, thresholds=list(thresholds), context_numbers=frozenset(numbers)
        )

    def _kept(self, plan, dataset, requested):
        repaired, issues = PlanQueryService._unjustified_reference_thresholds(plan, [dataset], requested)
        return [(item.operator, item.values[0] if item.values else None) for item in repaired.steps[0].filters], issues

    def test_a_number_standing_in_for_an_unverified_reference_is_removed(self) -> None:
        dataset = _dataset(_TURBIDITY, _YEAR)
        plan = self._filtered(dataset, "gt", 40.0)

        filters, issues = self._kept(plan, dataset, self._requested("niveles guía de la OMS"))

        self.assertEqual(filters, [("is_not_null", None)])
        self.assertEqual([item.code for item in issues], ["unverified_reference_threshold"])

    def test_justified_thresholds_are_kept_even_with_an_unverified_reference(self) -> None:
        dataset = _dataset(_TURBIDITY, _YEAR)
        threshold = RequestedThreshold(text="mayor o igual que 20000", operator="ge", value=20000, compares_statistic=False)
        for operator, value, requested in (
            ("gt", 100, self._requested("niveles guía", numbers={100.0})),
            ("ge", 20000, self._requested("niveles guía", thresholds=[threshold])),
            ("gt", 500, self._requested("valor guía", numbers={0.5})),
            ("gt", 0, self._requested("niveles guía")),
        ):
            with self.subTest(operator=operator, value=value):
                filters, issues = self._kept(self._filtered(dataset, operator, value), dataset, requested)
                self.assertEqual((filters, issues), ([(operator, value)], []))

    def test_without_a_documentary_reference_inferred_thresholds_are_not_judged(self) -> None:
        dataset = _dataset(_TURBIDITY, _YEAR)

        filters, issues = self._kept(self._filtered(dataset, "lt", 7), dataset, self._requested())

        self.assertEqual((filters, issues), ([("lt", 7)], []))

    def test_a_bound_on_a_calculated_value_is_not_judged(self) -> None:
        dataset = _dataset(_TURBIDITY, _YEAR)
        mean = {
            "id": "mean",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "calculations": [
                {"function": "mean", "inputs": [{"kind": "result", "step_id": _READ, "field": _TURBIDITY}], "output_name": "promedio"}
            ],
            "purpose": "Promediar.",
        }
        top = {
            "id": "top",
            "tool": "sql",
            "operation": "sort",
            "depends_on": ["mean"],
            "order_by": [{"field": {"kind": "result", "step_id": "mean", "field": "promedio"}, "direction": "desc"}],
            "filters": [{"field": {"kind": "result", "step_id": "mean", "field": "promedio"}, "operator": "gt", "values": [33]}],
            "purpose": "Ordenar.",
        }
        plan = _plan(dataset, [_TURBIDITY], [mean, top], output="top")

        repaired, issues = PlanQueryService._unjustified_reference_thresholds(plan, [dataset], self._requested("niveles"))

        self.assertEqual((len(repaired.steps[2].filters), issues), (1, []))

    def test_numbers_from_the_question_the_history_and_cited_filters_justify_a_bound(self) -> None:
        from aquai.application.Services.planQueryService import _context_numbers

        analysis = SimpleNamespace(operations=[{"filters": [{"values": [150]}, {"values": ["PUNTA LARA"]}]}])

        numbers = _context_numbers(["¿superan 20.000 cél/mL?", "pH de 8,5 o más"], analysis)

        self.assertEqual(numbers, frozenset({20000.0, 8.5, 150.0}))

    def test_an_impossible_branch_is_dropped_when_another_answers(self) -> None:
        dataset = _dataset(_TURBIDITY, _SITE)
        plan = _plan(dataset, [_TURBIDITY, _SITE])
        impossible = plan.steps[0].model_copy(update={"id": "impossible"})
        plan.steps.append(impossible)
        plan.output_steps = [_READ, "impossible"]
        issue = ValidationIssue(code="contradictory_filters", severity="clarification", message="x", step_id="impossible")

        reduced, issues = PlanQueryService._viable_branches(plan, [issue])
        alone, none = PlanQueryService._viable_branches(_plan(dataset, [_TURBIDITY]), [issue.model_copy(update={"step_id": _READ})])

        self.assertEqual(reduced.output_steps, [_READ])
        self.assertEqual([item.code for item in issues], ["impossible_branch"])
        self.assertEqual((alone.output_steps, none), ([_READ], []))


class EnumerationTests(unittest.TestCase):
    def test_codes_listed_after_a_shared_name_are_separate_values(self) -> None:
        self.assertEqual(coordinated_parts("estaciones 1, 2 y 3"), ["estacion 1", "estacion 2", "estacion 3"])
        self.assertEqual(coordinated_parts("Pta Lara y Balandra"), ["Pta Lara", "Balandra"])
        self.assertEqual(coordinated_parts("muestras 2005 y 2021"), [])


class PostfixThresholdTests(unittest.TestCase):
    def test_a_bound_written_after_the_number_is_a_threshold(self) -> None:
        for text, expected in (
            ("recuentos de 20.000 cél/mL o más", ("ge", 20000)),
            ("temperaturas de 25 grados o menos", ("le", 25)),
            ("turbiedad de 100 UNT o superior", ("ge", 100)),
            ("100 como mínimo", ("ge", 100)),
            ("pH 7,5 o más", ("ge", 7.5)),
        ):
            with self.subTest(text=text):
                found = numeric_comparisons(text)
                self.assertEqual([(item.operator, item.value) for item in found], [expected])
        self.assertEqual(numeric_comparisons("entre 2005 y 2021"), [])
        self.assertEqual(len(numeric_comparisons("al menos 20.000 cél/mL")), 1)

    def test_mentions_mislabeled_as_value_or_date_become_thresholds(self) -> None:
        question = "en 2005, ¿hubo recuentos de 20.000 cél/mL o más con el agua a menos de 25 grados?"
        mentions = [
            DataMention(text="recuentos", kind="variable"),
            DataMention(text="20.000 cél/mL", kind="value"),
            DataMention(text="menos de 25 grados", kind="date"),
            DataMention(text="2005", kind="date"),
        ]

        corrected = RouteQueryService._comparison_mentions(mentions, [question])

        self.assertEqual(
            [(item.kind, item.text) for item in corrected],
            [
                ("variable", "recuentos"),
                ("date", "2005"),
                ("threshold", "menos de 25"),
                ("threshold", "20.000 cél/mL o más"),
            ],
        )


class CuedSourceTests(unittest.TestCase):
    @staticmethod
    def _router():
        router = RouteQueryService(None, None)

        async def stored_anywhere(value, catalog):
            return value.casefold() in {"punta lara"}

        router._stored_anywhere = stored_anywhere
        return router

    def test_a_named_file_that_is_not_loaded_is_reported_instead_of_a_loose_word(self) -> None:
        question = "¿Coinciden los valores entre la planilla Rio 12-1 28-1 2-2 y la base completa fq, y por qué?"
        catalog = SourceNameTests._catalog()

        self.assertEqual(
            RouteQueryService._cued_sources(question, catalog),
            [("planilla", "Rio 12-1 28-1 2-2"), ("base", "completa fq")],
        )
        self.assertEqual(
            RouteQueryService._cued_sources("los datos de RdlP pasaron los niveles", catalog), [("datos", "RdlP")]
        )
        unknown = asyncio.run(self._router()._unknown_cued_sources(question, [], catalog, []))
        self.assertEqual(unknown, ["Rio 12-1 28-1 2-2", "base completa fq"])

    def test_names_of_loaded_sources_variables_and_values_are_not_reported(self) -> None:
        catalog = SourceNameTests._catalog()
        router = self._router()
        for question, mentions in (
            ("¿cuál es el pH en los datos de ABSA?", []),
            ("en la base FQ, durante 2005", []),
            ("¿cuáles son los datos de turbiedad media en Punta Lara?", [DataMention(text="turbiedad media", kind="variable")]),
            ("en el archivo de Punta Lara ¿qué turbiedad hay?", []),
            ("en la planilla que me pasaste ¿qué pH hay?", []),
            ("en la tabla datos completa de RdlP", []),
        ):
            with self.subTest(question=question):
                self.assertEqual(asyncio.run(router._unknown_cued_sources(question, mentions, catalog, [])), [])


class AbbreviatedVariableTests(unittest.TestCase):
    def test_a_clipped_variable_finds_the_column_it_abbreviates(self) -> None:
        from aquai.application.Services.retrieveDatasetsService import _variable_columns_named

        dataset = _dataset(_RAW, _NUMERIC, _TURBIDITY)

        self.assertEqual(_variable_columns_named(dataset, "abund absoluta"), [_RAW, _NUMERIC])
        self.assertEqual(_variable_columns_named(dataset, "turbiedad"), [_TURBIDITY])
        self.assertEqual(_variable_columns_named(dataset, "salinidad"), [])


class ReferenceVariableNameTests(unittest.TestCase):
    def test_a_reference_names_the_abbreviated_variable_by_its_full_name(self) -> None:
        from aquai.application.answering.measures import MeasuredResult
        from aquai.application.answering.references import _names_any

        result = MeasuredResult(
            evidence_id="",
            operation_id="",
            field="recuento",
            label="Recuento",
            variable="cianos potencialmente tóxicas",
            column="Recuento total de cianobacterias potencialmente tóxicas (cél/mL)",
            unit="cél/mL",
            unit_status="documented",
            basis=None,
            function=None,
            values=(),
            sources=(),
        )

        self.assertIn("cyanobacterial", result.names)
        self.assertTrue(_names_any("a guideline level of 20000 cyanobacterial cells/ml", result.names))


class DocumentSubjectTests(unittest.TestCase):
    _TEXT = "a guideline level of 20000 cyanobacterial cells/ml under conditions of cyanobacterial dominance"

    def test_an_abbreviated_variable_is_searched_as_its_column_names_it(self) -> None:
        from aquai.application.matching import names_subject, spelled_as, subject_terms

        spelled = spelled_as("cianos potencialmente tóxicas", "Recuento total de cianobacterias potencialmente tóxicas")

        self.assertEqual(spelled, "cianobacterias potencialmente toxicas")
        self.assertTrue(names_subject(self._TEXT, subject_terms(spelled)))
        self.assertFalse(names_subject(self._TEXT, subject_terms("temperatura del agua")))
        self.assertEqual(spelled_as("turbiedad media", "Turbiedad"), "turbiedad media")

    def test_the_reference_search_uses_the_spelled_variable_and_its_known_name(self) -> None:
        from aquai.application.Services.retrieveDocumentsService import _reference_request

        dataset = _dataset(_RAW, _NUMERIC)
        mention = "abund absolutas"
        state = SimpleNamespace(
            routing=SimpleNamespace(
                data_mentions=[DataMention(text=mention, kind="variable")],
                reference_comparison="niveles guía",
                document_ids=[],
            ),
            retrieval=SimpleNamespace(
                candidates=[dataset],
                evidence=[SimpleNamespace(dataset_id=dataset.dataset_id, variable_matches={mention: [_NUMERIC]})],
            ),
        )

        request = _reference_request(state)

        self.assertEqual(request.requests, ["abundancia absolutas: niveles guía"])
        self.assertIn("abundancia absolutas", request.subject)


if __name__ == "__main__":
    unittest.main()
