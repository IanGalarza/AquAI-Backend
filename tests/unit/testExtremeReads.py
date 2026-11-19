import asyncio
import json
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from pydantic import TypeAdapter

from aquai.application.answering.context import _date_pairs, _Writer
from aquai.application.contracts.planning import PlannerOutput, PlannerReply, ValidationIssue
from aquai.application.contracts.retrieval import (
    DatasetEvidence,
    DatasetProvenance,
    DatasetRetrievalResult,
    RetrievedColumn,
    VerifiedDataset,
)
from aquai.application.contracts.routing import DataMention
from aquai.application.errors import PlannerValidationError
from aquai.application.planning.context import asked_extremes, extreme_orders
from aquai.application.Services.planQueryService import PlanQueryService, _Requested
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus

_RAW = "abundancia_absoluta_cel_ml"
_NUMERIC = "abundancia_absoluta_cel_ml_numeric"
_TURBIDITY = "turbiedad_ntu"
_PH = "ph"
_SITE = "sitio"
_SPECIES = "merismopedia_minima"
_YEAR = "ano"
_DATE = "fecha"
_ADDRESS = "direccion"
_READ = "read_rows"
_SORT = f"{_READ}_extreme"
_MISSING = "missing_requested_extreme"
_COLUMNS = {
    _RAW: ("Abundancia absoluta (cél/ml)", "TEXT", "cél/ml", None),
    _NUMERIC: ("Abundancia absoluta (cél/ml) (representación numérica)", "NUMERIC", "cél/ml", _RAW),
    _TURBIDITY: ("Turbiedad (NTU)", "NUMERIC", "NTU", None),
    _PH: ("pH", "NUMERIC", None, None),
    _SITE: ("Sitio", "TEXT", None, None),
    _SPECIES: ("Merismopedia minima", "NUMERIC", None, None),
    _YEAR: ("Año", "BIGINT", None, None),
    _DATE: ("Fecha", "DATE", None, None),
    _ADDRESS: ("Direccion", "TEXT", None, None),
}


class _ScriptedPlanner:
    context_budget_bytes = 100_000

    def __init__(self, decisions) -> None:
        self._decisions = list(decisions)
        self.feedback = []

    async def propose(self, context) -> PlannerReply:
        self.feedback.append(context.evidence.get("previous_attempt_feedback"))
        return PlannerReply(decision=self._decisions.pop(0), model="scripted")


class _Catalog:
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def verified_datasets(self, ids, versions=None) -> list:
        return [self._dataset]

    def find_text_values(self, datasets, terms, *, per_column=50) -> dict:
        return {}

    def frequent_text_values(self, dataset, column, *, measured=None) -> list:
        return []


def _column(name) -> RetrievedColumn:
    original_name, sql_type, unit, derived_from = _COLUMNS[name]
    return RetrievedColumn(
        original_name=original_name,
        sql_name=name,
        sql_type=sql_type,
        unit=unit,
        numeric_eligible=sql_type == "NUMERIC",
        derived_from=derived_from,
    )


def _dataset(*names) -> VerifiedDataset:
    return VerifiedDataset(
        dataset_id=uuid4(),
        source_id=uuid4(),
        semantic_point_id=uuid4(),
        name="Fuente / BASE / A1:C10",
        description=None,
        physical_table_name="dataset_base",
        verified_version=1,
        row_count=10,
        columns_metadata=[_column(name) for name in names],
        provenance=DatasetProvenance(
            source_name="Fuente.xlsx",
            original_filename="Fuente.xlsx",
            file_type="xlsx",
            content_hash="0" * 64,
            source_locator={},
            processing_status=ProcessingStatus.READY,
            semantic_index_status=SemanticIndexStatus.READY,
        ),
    )


def _ref(dataset, name) -> dict:
    return {"kind": "column", "dataset_id": str(dataset.dataset_id), "column": name}


def _plan(dataset, names, tail=(), output=_READ, **read) -> object:
    data = {
        "status": "plannable",
        "reason": "Lee lo pedido.",
        "datasets": [
            {
                "dataset_id": str(dataset.dataset_id),
                "version": 1,
                "columns": list(names),
                "reason": "Tiene lo pedido.",
            }
        ],
        "steps": [
            {
                "id": _READ,
                "tool": "sql",
                "operation": "read",
                "columns": [_ref(dataset, name) for name in names],
                "filters": [],
                "purpose": "Leer los registros.",
                **read,
            },
            *tail,
        ],
        "output_steps": [output],
    }
    return TypeAdapter(PlannerOutput).validate_json(json.dumps(data))


def _maximum(dataset, name, group=None) -> object:
    field = {"kind": "result", "step_id": _READ, "field": group}
    return _plan(
        dataset,
        [name, group] if group else [name],
        [
            {
                "id": "max_rows",
                "tool": "sql",
                "operation": "aggregate",
                "depends_on": [_READ],
                "group_by": [{"field": field, "output_name": group}] if group else [],
                "calculations": [
                    {
                        "function": "max",
                        "inputs": [{"kind": "result", "step_id": _READ, "field": name}],
                        "output_name": "maximum",
                    }
                ],
                "purpose": "Calcular el máximo.",
            }
        ],
        output="max_rows",
    )


def _planned(dataset, question, decisions, *, matched=(), mentions=None):
    planner = _ScriptedPlanner(decisions)
    mentions = mentions or [DataMention(text="variable pedida", kind="variable")]
    retrieval = DatasetRetrievalResult(
        question=question,
        candidates=[dataset],
        evidence=[
            DatasetEvidence(
                dataset_id=dataset.dataset_id,
                variable_matches={mentions[0].text: list(matched)} if matched else {},
            )
        ],
        verified_at=datetime.now(timezone.utc),
        status="candidates_available",
    )
    service = PlanQueryService(planner, _Catalog(dataset))
    result = asyncio.run(service.plan(question, retrieval, data_mentions=mentions))

    return result, planner.feedback


def _step(result, key) -> object:
    return next(step for step in result.operations if step.id == key)


def _codes(result) -> list[str]:
    return [item.code for item in result.validation_issues if item.severity == "clarification"]


class ExtremeOrderTests(unittest.TestCase):
    def test_each_word_says_its_direction(self) -> None:
        for question, expected in (
            ("¿Cuál es la mayor turbiedad?", {"desc"}),
            ("¿Cuáles son las mayores turbiedades?", {"desc"}),
            ("¿Cuál es el máximo de pH?", {"desc"}),
            ("¿Cuál fue el valor más alto de pH?", {"desc"}),
            ("What is the largest pH?", {"desc"}),
            ("¿Cuál es la menor turbiedad?", {"asc"}),
            ("¿Cuál es la turbiedad mínima?", {"asc"}),
            ("¿Cuál fue el valor más bajo de pH?", {"asc"}),
            ("Which site has the smallest pH?", {"asc"}),
            ("¿Cuál es el mayor y el menor pH?", {"asc", "desc"}),
            ("¿Qué mediciones tienen pH mayor a 8?", set()),
            ("Mostrame el pH de mayor a menor", set()),
        ):
            with self.subTest(question=question):
                self.assertEqual(extreme_orders(question), expected)

    def test_a_word_inside_a_name_is_not_an_extreme(self) -> None:
        names = ["Merismopedia minima"]

        self.assertEqual(asked_extremes("Mostrame Merismopedia minima", names), set())
        self.assertEqual(
            asked_extremes("¿Cuál es la mayor abundancia de Merismopedia minima?", names), {"desc"}
        )

    def test_the_unit_of_a_header_does_not_hide_its_name(self) -> None:
        self.assertEqual(
            asked_extremes("¿Cuál es la temperatura máxima?", ["Temperatura máxima (°C)"]), set()
        )


class ExtremeRepairTests(unittest.TestCase):
    def test_a_lone_read_gets_the_ordering_of_the_largest_value(self) -> None:
        dataset = _dataset(_RAW, _NUMERIC)

        result, _ = _planned(
            dataset,
            "¿Cuál es la mayor abundancia absoluta en cél/ml?",
            [_plan(dataset, [_NUMERIC])],
            matched=(_RAW, _NUMERIC),
        )

        sort = _step(result, _SORT)
        self.assertEqual(result.status, "plannable")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output_steps, [_SORT])
        self.assertEqual(sort.order_by[0].direction, "desc")
        self.assertEqual(sort.order_by[0].field.field, _NUMERIC)
        self.assertEqual(sort.limit, 5)

    def test_the_lowest_value_orders_upwards(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, _ = _planned(
            dataset, "¿Cuál es la turbiedad mínima?", [_plan(dataset, [_TURBIDITY])], matched=(_TURBIDITY,)
        )

        self.assertEqual(_step(result, _SORT).order_by[0].direction, "asc")

    def test_records_without_the_variable_stay_out(self) -> None:
        dataset = _dataset(_TURBIDITY, _SITE)

        result, _ = _planned(
            dataset, "¿Cuál es la mayor turbiedad?", [_plan(dataset, [_TURBIDITY, _SITE])], matched=(_TURBIDITY,)
        )

        filters = _step(result, _READ).filters
        self.assertEqual([(item.field.column, item.operator) for item in filters], [(_TURBIDITY, "is_not_null")])

    def test_a_text_measurement_ends_on_its_numeric_projection(self) -> None:
        dataset = _dataset(_RAW, _NUMERIC)

        result, _ = _planned(
            dataset, "¿Cuál es la mayor abundancia?", [_plan(dataset, [_RAW])], matched=(_RAW, _NUMERIC)
        )

        self.assertEqual(result.status, "plannable")
        self.assertEqual([ref.column for ref in _step(result, _READ).columns], [_NUMERIC])
        self.assertEqual(_step(result, _SORT).order_by[0].field.field, _NUMERIC)

    def test_the_numeric_version_wins_over_its_text(self) -> None:
        dataset = _dataset(_RAW, _NUMERIC)

        result, _ = _planned(
            dataset, "¿Cuál es la mayor abundancia?", [_plan(dataset, [_RAW, _NUMERIC])], matched=(_RAW, _NUMERIC)
        )

        self.assertEqual(_step(result, _SORT).order_by[0].field.field, _NUMERIC)

    def test_a_text_measurement_without_projection_is_not_ordered(self) -> None:
        dataset = _dataset(_RAW)

        result, _ = _planned(
            dataset,
            "¿Cuál es la mayor abundancia?",
            [_plan(dataset, [_RAW]), _plan(dataset, [_RAW])],
            matched=(_RAW,),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.operations, [])
        self.assertEqual(_codes(result), ["text_measurement_order"])

    def test_the_requested_count_replaces_the_default_rows(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, _ = _planned(
            dataset,
            "¿Cuáles son las 3 mayores turbiedades?",
            [_plan(dataset, [_TURBIDITY])],
            matched=(_TURBIDITY,),
            mentions=[
                DataMention(text="turbiedades", kind="variable"),
                DataMention(text="3", kind="limit"),
            ],
        )

        self.assertEqual(_step(result, _SORT).limit, 3)

    def test_a_long_read_id_still_fits(self) -> None:
        dataset = _dataset(_TURBIDITY)
        long_id = "r" * 63
        plan = _plan(dataset, [_TURBIDITY], output=long_id, id=long_id)

        result, _ = _planned(dataset, "¿Cuál es la mayor turbiedad?", [plan], matched=(_TURBIDITY,))

        self.assertEqual(result.status, "plannable")
        self.assertLessEqual(len(result.output_steps[0]), 63)


class ExtremeIssueTests(unittest.TestCase):
    def _retried(self, question, names, matched, mentions=None, group=None):
        dataset = _dataset(*dict.fromkeys(names))
        return _planned(
            dataset,
            question,
            [_plan(dataset, names), _maximum(dataset, matched[0], group)],
            matched=matched,
            mentions=mentions,
        )

    def _assert_retried(self, result, feedback) -> None:
        self.assertEqual(result.status, "plannable")
        self.assertEqual(result.attempts, 2)
        self.assertIn("ninguna salida ordena ni calcula", feedback[1][0])
        self.assertNotIn(_SORT, result.output_steps)

    def test_both_directions_ask_the_model_again(self) -> None:
        self._assert_retried(
            *self._retried("¿Cuál es el mayor y el menor pH?", [_PH], (_PH,))
        )

    def test_two_variables_ask_the_model_again(self) -> None:
        self._assert_retried(
            *self._retried("¿Cuál es la mayor turbiedad?", [_TURBIDITY, _PH], (_TURBIDITY, _PH))
        )

    def test_a_requested_dimension_asks_the_model_again(self) -> None:
        self._assert_retried(
            *self._retried(
                "¿Cuál es la mayor turbiedad de cada sitio?",
                [_TURBIDITY, _SITE],
                (_TURBIDITY,),
                [
                    DataMention(text="turbiedad", kind="variable"),
                    DataMention(text="sitio", kind="dimension"),
                ],
                group=_SITE,
            )
        )

    def test_a_variable_the_retriever_did_not_match_asks_the_model_again(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, feedback = _planned(
            dataset,
            "¿Cuál es la mayor turbiedad?",
            [_plan(dataset, [_TURBIDITY]), _maximum(dataset, _TURBIDITY)],
        )

        self._assert_retried(result, feedback)

    def test_a_read_that_stays_unanswered_is_not_validated(self) -> None:
        dataset = _dataset(_PH)
        question = "¿Cuál es el mayor y el menor pH?"

        result, _ = _planned(dataset, question, [_plan(dataset, [_PH]), _plan(dataset, [_PH])], matched=(_PH,))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.operations, [])
        self.assertEqual(_codes(result), [_MISSING])


class UntouchedPlanTests(unittest.TestCase):
    def _assert_untouched(self, result, attempts=1) -> None:
        self.assertEqual(result.status, "plannable")
        self.assertEqual(result.attempts, attempts)
        self.assertNotIn(_MISSING, _codes(result))
        self.assertFalse([step for step in result.operations if step.id == _SORT])

    def test_a_question_without_extreme_keeps_its_read(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, _ = _planned(
            dataset, "Mostrame la turbiedad", [_plan(dataset, [_TURBIDITY])], matched=(_TURBIDITY,)
        )

        self._assert_untouched(result)
        self.assertEqual(result.output_steps, [_READ])

    def test_a_species_named_like_an_extreme_is_not_ordered(self) -> None:
        dataset = _dataset(_SPECIES)

        result, _ = _planned(
            dataset, "Mostrame Merismopedia minima", [_plan(dataset, [_SPECIES])], matched=(_SPECIES,)
        )

        self._assert_untouched(result)

    def test_a_value_named_like_an_extreme_is_not_ordered(self) -> None:
        dataset = _dataset(_TURBIDITY, _SITE)

        result, _ = _planned(
            dataset,
            "¿Cuál es la turbiedad de Laguna Mínima?",
            [_plan(dataset, [_TURBIDITY, _SITE])],
            matched=(_TURBIDITY,),
            mentions=[
                DataMention(text="turbiedad", kind="variable"),
                DataMention(text="Laguna Mínima", kind="dimension"),
            ],
        )

        self._assert_untouched(result)

    def test_a_count_bound_is_not_an_extreme(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, _ = _planned(
            dataset,
            "Mostrame un máximo de 5 registros de turbiedad",
            [_plan(dataset, [_TURBIDITY])],
            matched=(_TURBIDITY,),
        )

        self._assert_untouched(result)

    def test_a_calculated_extreme_is_left_alone(self) -> None:
        dataset = _dataset(_TURBIDITY)

        result, _ = _planned(
            dataset, "¿Cuál es la mayor turbiedad?", [_maximum(dataset, _TURBIDITY)], matched=(_TURBIDITY,)
        )

        self._assert_untouched(result)
        self.assertEqual(result.output_steps, ["max_rows"])

    def test_a_read_that_already_orders_is_left_alone(self) -> None:
        dataset = _dataset(_TURBIDITY)
        ordering = {"field": _ref(dataset, _TURBIDITY), "direction": "desc"}
        plan = _plan(dataset, [_TURBIDITY], order_by=[ordering], limit=1)

        result, _ = _planned(dataset, "¿Cuál es la mayor turbiedad?", [plan], matched=(_TURBIDITY,))

        self._assert_untouched(result)
        self.assertEqual(result.output_steps, [_READ])


class PlanFormRepairTests(unittest.TestCase):
    def test_a_threshold_written_as_text_becomes_a_number(self) -> None:
        dataset = _dataset(_TURBIDITY)
        threshold = {"field": _ref(dataset, _TURBIDITY), "operator": "ge", "values": ["20.000"]}
        mentions = [DataMention(text="turbiedad", kind="variable"), DataMention(text="al menos 20.000", kind="threshold")]

        result, _ = _planned(
            dataset,
            "¿Qué registros tienen una turbiedad de al menos 20.000?",
            [_plan(dataset, [_TURBIDITY], filters=[threshold])],
            matched=(_TURBIDITY,),
            mentions=mentions,
        )

        self.assertEqual(result.status, "plannable")
        self.assertEqual(_step(result, _READ).filters[0].values, [20000])

    def test_a_value_filtered_in_the_wrong_column_moves_where_it_was_found(self) -> None:
        dataset = _dataset(_TURBIDITY, _SITE, _ADDRESS)
        stored = "C: AVENIDA ALTE G BROWN N 330 [Club de Pesca][Toma de Agua]"
        wrong = {"field": _ref(dataset, _SITE), "operator": "eq", "values": ["Club de Pesca"]}
        plan = _plan(dataset, [_TURBIDITY, _SITE], filters=[wrong])
        requested = _Requested(
            thresholds=[], periods=[], statistics=[], dimensions={}, chart=False, chart_required=False,
            values=["Club de Pesca"], categories=False, several_rows=True,
            value_matches={dataset.dataset_id: {"Club de Pesca": [(_ADDRESS, [stored])]}},
        )

        repaired = PlanQueryService._misplaced_values(plan, requested)
        predicate = repaired.steps[0].filters[0]

        self.assertEqual((predicate.field.column, predicate.operator, predicate.values), (_ADDRESS, "in", [stored]))
        self.assertIn(_ADDRESS, repaired.datasets[0].columns)

    def test_an_unrequested_limit_by_year_is_removed(self) -> None:
        dataset = _dataset(_TURBIDITY, _YEAR)
        latest = {
            "id": "latest_year",
            "tool": "sql",
            "operation": "sort",
            "depends_on": [_READ],
            "order_by": [{"field": {"kind": "result", "step_id": _READ, "field": _YEAR}, "direction": "desc"}],
            "limit": 1,
            "purpose": "Quedarse con el último año.",
        }
        plan = _plan(dataset, [_TURBIDITY, _YEAR], [latest], output="latest_year")

        result, _ = _planned(dataset, "Mostrame los registros de turbiedad", [plan], matched=(_TURBIDITY,))

        self.assertEqual(result.output_steps, [_READ])
        self.assertNotIn("latest_year", [step.id for step in result.operations])

    def test_a_series_by_month_gets_a_line_chart(self) -> None:
        dataset = _dataset(_TURBIDITY, _DATE)
        monthly = {
            "id": "monthly_mean",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "group_by": [
                {"field": {"kind": "result", "step_id": _READ, "field": _DATE}, "output_name": "mes", "time_grain": "month"}
            ],
            "calculations": [
                {"function": "mean", "inputs": [{"kind": "result", "step_id": _READ, "field": _TURBIDITY}], "output_name": "promedio"}
            ],
            "purpose": "Promediar por mes.",
        }
        plan = _plan(dataset, [_TURBIDITY, _DATE], [monthly], output="monthly_mean")

        result, _ = _planned(dataset, "¿Cuál fue la turbiedad promedio por mes?", [plan], matched=(_TURBIDITY,))

        charts = [step for step in result.operations if step.operation == "visualize"]
        self.assertEqual(result.status, "plannable")
        self.assertEqual([item.visualization.chart_type for item in charts], ["line"])
        self.assertEqual(result.output_steps, ["monthly_mean", charts[0].id])

    def test_an_unrequested_chart_of_records_that_feed_a_mean_publishes_only_the_mean(self) -> None:
        dataset = _dataset(_TURBIDITY, _DATE)
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
        trend = {
            "id": "trend",
            "tool": "python",
            "operation": "visualize",
            "depends_on": [_READ],
            "visualization": {
                "chart_type": "line",
                "title": "Turbiedad",
                "x": {"kind": "result", "step_id": _READ, "field": _DATE},
                "y": {"kind": "result", "step_id": _READ, "field": _TURBIDITY},
                "x_label": "Fecha",
                "y_label": "Turbiedad",
            },
            "purpose": "Ver la tendencia.",
        }
        plan = _plan(dataset, [_TURBIDITY, _DATE], [mean, trend], output="mean")
        plan.output_steps = ["mean", "trend"]

        repaired = PlanQueryService._unrequested_charts(plan, False)

        self.assertEqual(repaired.output_steps, ["mean"])
        self.assertNotIn("trend", [step.id for step in repaired.steps])

    def test_a_python_only_statistic_left_in_sql_moves_to_python_with_its_dependents(self) -> None:
        dataset = _dataset(_TURBIDITY)
        median = {
            "id": "median",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "calculations": [
                {"function": "median", "inputs": [{"kind": "result", "step_id": _READ, "field": _TURBIDITY}], "output_name": "mediana"}
            ],
            "purpose": "Mediana.",
        }
        ordered = {
            "id": "ordered",
            "tool": "sql",
            "operation": "sort",
            "depends_on": ["median"],
            "order_by": [{"field": {"kind": "result", "step_id": "median", "field": "mediana"}, "direction": "desc"}],
            "purpose": "Ordenar.",
        }
        plan = _plan(dataset, [_TURBIDITY], [median, ordered], output="ordered")

        repaired = PlanQueryService._python_only_steps(plan)

        self.assertEqual([step.tool for step in repaired.steps], ["sql", "python", "python"])

    def test_a_branch_that_fails_verification_is_left_out_and_the_rest_answers(self) -> None:
        dataset = _dataset(_TURBIDITY, _PH)
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
        broken = {
            "id": "broken",
            "tool": "sql",
            "operation": "aggregate",
            "depends_on": [_READ],
            "calculations": [
                {"function": "max", "inputs": [{"kind": "result", "step_id": _READ, "field": "no_existe"}], "output_name": "maximo"}
            ],
            "purpose": "Máximo.",
        }
        plan = _plan(dataset, [_TURBIDITY, _PH], [mean, broken], output="mean")
        plan.output_steps = ["mean", "broken"]

        result, feedback = _planned(dataset, "¿Cuál es la turbiedad promedio?", [plan, plan], matched=(_TURBIDITY,))

        self.assertEqual(result.status, "plannable")
        self.assertEqual(result.output_steps, ["mean"])
        self.assertIn("unverified_branch", [item.code for item in result.validation_issues])
        self.assertTrue(feedback[1])

    def test_a_plan_missing_only_a_named_source_survives_a_failed_retry(self) -> None:
        class Provider:
            context_budget_bytes = 100_000

            async def propose(self, context):
                return SimpleNamespace(decision=None, model="scripted", input_tokens=1, output_tokens=1)

        gap = ValidationIssue(code="missing_named_source", severity="warning", message="Falta BASE FQ.", subject="BASE FQ")
        first, calls = SimpleNamespace(issues=[gap]), []

        async def verify(decision):
            calls.append(decision)
            if len(calls) == 1:
                return first
            raise PlannerValidationError("El modelo seleccionó una columna inexistente: sitio.")

        context = SimpleNamespace(evidence={}, sent_to_model=False)
        proposal = asyncio.run(PlanQueryService(Provider(), None)._propose(context, verify))

        self.assertIs(proposal.verified, first)
        self.assertIsNone(proposal.error)
        self.assertEqual((proposal.attempts, len(proposal.rejections)), (2, 2))

    def test_a_requested_period_adds_the_first_and_last_date_to_the_mean(self) -> None:
        dataset = _dataset(_TURBIDITY, _DATE)
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
        plan = _plan(dataset, [_TURBIDITY], [mean], output="mean")

        result, _ = _planned(
            dataset, "Indicá la turbiedad promedio y el período que cubren los datos", [plan], matched=(_TURBIDITY,)
        )
        calculations = {item.output_name: item for item in _step(result, "mean").calculations}

        self.assertEqual(result.status, "plannable")
        self.assertEqual((calculations["primera_fecha"].function, calculations["ultima_fecha"].function), ("min", "max"))
        self.assertEqual(calculations["primera_fecha"].inputs[0].field, _DATE)
        self.assertIn(_DATE, [ref.column for ref in _step(result, _READ).columns])

    def test_the_first_and_last_date_of_a_mean_are_said_as_one_period(self) -> None:
        writer = _Writer({}, {}, "es", False)
        fields = {
            "promedio": SimpleNamespace(sql_type="NUMERIC"),
            "primera_fecha": SimpleNamespace(sql_type="DATE"),
            "ultima_fecha": SimpleNamespace(sql_type="DATE"),
        }
        field = SimpleNamespace(field=_DATE)
        calculations = {
            "promedio": SimpleNamespace(function="mean", inputs=[SimpleNamespace(field=_TURBIDITY)]),
            "primera_fecha": SimpleNamespace(function="min", inputs=[field]),
            "ultima_fecha": SimpleNamespace(function="max", inputs=[field]),
        }

        self.assertEqual(_date_pairs(list(fields), calculations, fields), {"primera_fecha": "ultima_fecha"})
        self.assertEqual(
            writer.correlation_summary(0.109, 51), "Se calculó con 51 pares de valores: una relación lineal débil y positiva."
        )
        self.assertEqual(
            writer._records_span(date(2021, 9, 1), date(2022, 3, 31)), "los registros van del 01/09/2021 al 31/03/2022"
        )


if __name__ == "__main__":
    unittest.main()
