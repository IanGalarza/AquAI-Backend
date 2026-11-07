import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import NamedTuple
from uuid import uuid4

from pydantic import TypeAdapter

from aquai.application.answering.coverage import text_measurement_notes
from aquai.application.contracts.planning import PlannerOutput, PlannerReply, ResultRef
from aquai.application.contracts.retrieval import (
    DatasetEvidence,
    DatasetProvenance,
    DatasetRetrievalResult,
    RetrievedColumn,
    VerifiedDataset,
)
from aquai.application.contracts.routing import DataMention
from aquai.application.planning.validation import validate_text_orderings
from aquai.application.Services.planQueryService import PlanQueryService
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus

_MENTION = "abundancia absoluta (ind/ml)"
_LARGEST = "¿Cuál es la mayor abundancia absoluta (ind/ml)?"
_MAXIMUM = "¿Cuál es el máximo de abundancia absoluta (ind/ml)?"
_ABOVE = "¿Qué registros tienen abundancia absoluta (ind/ml) mayor a 100?"
_RAW = "abundancia_absoluta_ind_ml"
_PROJECTION = "abundancia_absoluta_ind_ml_numeric"
_SITE = "sitio"
_SPECIES = "microcystis_aeruginosa"
_COLONIES = "abundancia_absoluta_colonias_ml"
_TEXT_MEASUREMENT = {
    "original_name": "Abundancia absoluta (ind/ml)",
    "sql_name": _RAW,
    "sql_type": "TEXT",
    "unit": "ind/ml",
    "numeric_eligible": False,
}
_NUMERIC_PROJECTION = {
    "original_name": "Abundancia absoluta (ind/ml) (representación numérica)",
    "sql_name": _PROJECTION,
    "sql_type": "NUMERIC",
    "unit": "ind/ml",
    "numeric_eligible": True,
    "derived_from": _RAW,
    "excluded_count": 1,
}
_CATEGORY = {
    "original_name": "Sitio",
    "sql_name": _SITE,
    "sql_type": "TEXT",
    "numeric_eligible": False,
}
_UNITLESS_MEASUREMENT = {
    "original_name": "Microcystis aeruginosa",
    "sql_name": _SPECIES,
    "sql_type": "TEXT",
    "numeric_eligible": False,
}
_NUMERIC_MEASUREMENT = {
    "original_name": "Abundancia absoluta (colonias/ml)",
    "sql_name": _COLONIES,
    "sql_type": "NUMERIC",
    "unit": "colonias/ml",
    "numeric_eligible": True,
}
_NOTE = (
    "En BaseCiano (hoja BASE CIANO), Abundancia absoluta (ind/ml) está guardada como texto "
    "y no tiene una versión numérica, así que no pude calcular lo pedido."
)


class _Run(NamedTuple):
    result: object
    state: SimpleNamespace
    feedback: list


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


def _dataset(*columns) -> VerifiedDataset:
    return VerifiedDataset(
        dataset_id=uuid4(),
        source_id=uuid4(),
        semantic_point_id=uuid4(),
        name="BaseCiano / BASE CIANO / A1:AG2228",
        description=None,
        physical_table_name="dataset_base_ciano",
        verified_version=2,
        row_count=2227,
        columns_metadata=[RetrievedColumn(**column) for column in columns],
        provenance=DatasetProvenance(
            source_name="BaseCiano.xlsx",
            original_filename="BaseCiano.xlsx",
            file_type="xlsx",
            content_hash="0" * 64,
            source_locator={},
            processing_status=ProcessingStatus.READY,
            semantic_index_status=SemanticIndexStatus.READY,
        ),
    )


def _decision(data) -> object:
    return TypeAdapter(PlannerOutput).validate_json(json.dumps(data))


def _refusal() -> object:
    return _decision(
        {
            "status": "insufficient_coverage",
            "reason": "La variable pedida está guardada como texto.",
            "missing_information": ["Una versión numérica de la variable."],
        }
    )


def _ordered_plan(dataset, ordered, *others, repeated=False, limit=5) -> object:
    dataset_id = str(dataset.dataset_id)
    columns = [ordered, *others]
    last = "limit_rows" if repeated else "sort_rows"
    sort = {
        "id": "sort_rows",
        "tool": "sql",
        "operation": "sort",
        "depends_on": ["read_rows"],
        "order_by": [
            {
                "field": {"kind": "result", "step_id": "read_rows", "field": ordered},
                "direction": "desc",
            }
        ],
        "limit": None if repeated else limit,
        "purpose": "Ordenar los registros.",
    }
    tail = {
        **sort,
        "id": "limit_rows",
        "depends_on": ["sort_rows"],
        "order_by": [
            {
                "field": {"kind": "result", "step_id": "sort_rows", "field": ordered},
                "direction": "desc",
            }
        ],
        "limit": limit,
    }
    return _decision(
        {
            "status": "plannable",
            "reason": "Ordena lo pedido.",
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "version": 2,
                    "columns": columns,
                    "reason": "Tiene lo pedido.",
                }
            ],
            "steps": [
                {
                    "id": "read_rows",
                    "tool": "sql",
                    "operation": "read",
                    "columns": [
                        {"kind": "column", "dataset_id": dataset_id, "column": name}
                        for name in columns
                    ],
                    "filters": [
                        {
                            "field": {"kind": "column", "dataset_id": dataset_id, "column": ordered},
                            "operator": "is_not_null",
                            "values": [],
                        }
                    ],
                    "purpose": "Leer los registros.",
                },
                sort,
                *([tail] if repeated else []),
            ],
            "output_steps": [last],
        }
    )


def _planned(dataset, *decisions, question=_LARGEST, matched=(_RAW,)) -> _Run:
    planner = _ScriptedPlanner(decisions)
    mentions = [DataMention(text=_MENTION, kind="variable")]
    retrieval = DatasetRetrievalResult(
        question=question,
        candidates=[dataset],
        evidence=[
            DatasetEvidence(
                dataset_id=dataset.dataset_id,
                variable_matches={_MENTION: list(matched)} if matched else {},
            )
        ],
        verified_at=datetime.now(timezone.utc),
        status="candidates_available",
    )
    result = asyncio.run(
        PlanQueryService(planner, _Catalog(dataset)).plan(
            question, retrieval, data_mentions=mentions
        )
    )
    state = SimpleNamespace(
        retrieval=retrieval,
        planning=result,
        routing=SimpleNamespace(data_mentions=mentions),
    )
    return _Run(result, state, planner.feedback)


def _blocking_codes(result) -> list[str]:
    return [item.code for item in result.validation_issues if item.severity == "clarification"]


def _step(result, key) -> object:
    return next(step for step in result.operations if step.id == key)


class TextMeasurementOrderTests(unittest.TestCase):
    def test_a_text_measurement_is_rejected_and_the_retry_keeps_the_concrete_reason(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _CATEGORY)

        run = _planned(dataset, _ordered_plan(dataset, _RAW), _refusal())

        self.assertEqual(run.result.status, "insufficient_coverage")
        self.assertEqual(run.result.attempts, 2)
        self.assertEqual(run.result.operations, [])
        self.assertIn(_RAW, run.feedback[1][0])
        self.assertEqual(text_measurement_notes(run.state, "es"), [_NOTE])

    def test_a_repeated_text_order_never_reaches_execution(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _CATEGORY)

        run = _planned(dataset, _ordered_plan(dataset, _RAW), _ordered_plan(dataset, _RAW))

        self.assertEqual(run.result.status, "failed")
        self.assertEqual(run.result.operations, [])
        self.assertEqual(_blocking_codes(run.result), ["text_measurement_order"])
        self.assertEqual(text_measurement_notes(run.state, "es"), [_NOTE])

    def test_an_extreme_refused_at_once_is_a_coverage_limit_with_its_reason(self) -> None:
        run = _planned(_dataset(_TEXT_MEASUREMENT), _refusal(), question=_MAXIMUM)

        self.assertEqual(run.result.status, "insufficient_coverage")
        self.assertEqual(text_measurement_notes(run.state, "es"), [_NOTE])

    def test_a_unit_marks_a_measurement_that_the_mention_did_not_match(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT)
        plans = (_ordered_plan(dataset, _RAW), _ordered_plan(dataset, _RAW))

        run = _planned(dataset, *plans, matched=())

        self.assertEqual(_blocking_codes(run.result), ["text_measurement_order"])

    def test_a_requested_variable_without_unit_is_a_measurement(self) -> None:
        dataset = _dataset(_UNITLESS_MEASUREMENT)
        plans = (_ordered_plan(dataset, _SPECIES), _ordered_plan(dataset, _SPECIES))

        run = _planned(dataset, *plans, matched=(_SPECIES,))

        self.assertEqual(_blocking_codes(run.result), ["text_measurement_order"])

    def test_a_projection_replaces_the_ordered_text_and_answers(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _NUMERIC_PROJECTION, _CATEGORY)

        run = _planned(
            dataset, _ordered_plan(dataset, _RAW, _SITE), matched=(_RAW, _PROJECTION)
        )

        read = _step(run.result, "read_rows")
        self.assertEqual(run.result.status, "plannable")
        self.assertEqual(run.result.attempts, 1)
        self.assertEqual([ref.column for ref in read.columns], [_PROJECTION, _SITE])
        self.assertEqual(read.filters[0].field.column, _PROJECTION)
        self.assertEqual(_step(run.result, "sort_rows").order_by[0].field.field, _PROJECTION)
        self.assertEqual(_blocking_codes(run.result), [])

    def test_every_step_of_the_branch_cites_the_projection(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _NUMERIC_PROJECTION)

        run = _planned(
            dataset,
            _ordered_plan(dataset, _RAW, repeated=True),
            matched=(_RAW, _PROJECTION),
        )

        self.assertEqual(run.result.status, "plannable")
        self.assertEqual(_step(run.result, "limit_rows").order_by[0].field.field, _PROJECTION)
        self.assertEqual(_step(run.result, "limit_rows").order_by[0].field.step_id, "sort_rows")

    def test_a_text_category_can_still_be_sorted(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _CATEGORY)

        run = _planned(dataset, _ordered_plan(dataset, _SITE, _RAW))

        self.assertEqual(run.result.status, "plannable")
        self.assertEqual(_blocking_codes(run.result), [])

    def test_a_numeric_measurement_can_still_be_sorted(self) -> None:
        dataset = _dataset(_NUMERIC_MEASUREMENT)

        run = _planned(dataset, _ordered_plan(dataset, _COLONIES), matched=(_COLONIES,))

        self.assertEqual(run.result.status, "plannable")
        self.assertEqual(_blocking_codes(run.result), [])

    def test_the_largest_value_does_not_publish_every_record(self) -> None:
        dataset = _dataset(_NUMERIC_MEASUREMENT)

        run = _planned(dataset, _ordered_plan(dataset, _COLONIES, limit=None), matched=(_COLONIES,))

        self.assertEqual(_step(run.result, "sort_rows").limit, 5)

    def test_a_projected_largest_value_is_limited_too(self) -> None:
        dataset = _dataset(_TEXT_MEASUREMENT, _NUMERIC_PROJECTION)
        plan = _ordered_plan(dataset, _RAW, repeated=True, limit=None)

        run = _planned(dataset, plan, matched=(_RAW, _PROJECTION))

        self.assertEqual(run.result.status, "plannable")
        self.assertEqual(_step(run.result, "limit_rows").order_by[0].field.field, _PROJECTION)
        self.assertEqual(_step(run.result, "limit_rows").limit, 5)

    def test_a_threshold_keeps_every_record(self) -> None:
        dataset = _dataset(_NUMERIC_MEASUREMENT)
        plan = _ordered_plan(dataset, _COLONIES, limit=None)

        run = _planned(dataset, plan, question=_ABOVE, matched=(_COLONIES,))

        self.assertIsNone(_step(run.result, "sort_rows").limit)

    def test_latest_keeps_ordering_verified_text_periods(self) -> None:
        field = SimpleNamespace(name="fecha", sql_type="TEXT", unit=None, sources=[])
        period = ResultRef(kind="result", step_id="read_rows", field="fecha")
        steps = [
            SimpleNamespace(id="read_rows", operation="read", order_by=[], output_fields=[field]),
            SimpleNamespace(
                id="latest_rows",
                operation="latest",
                order_by=[SimpleNamespace(field=period)],
                output_fields=[field],
            ),
        ]

        self.assertEqual(validate_text_orderings(steps, {}), [])


if __name__ == "__main__":
    unittest.main()
