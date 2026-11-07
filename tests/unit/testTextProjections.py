import unittest
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from sqlalchemy import BigInteger, Column, MetaData, Numeric, Table, Text, select
from sqlalchemy.dialects import postgresql

from aquai.application.answering.context import _ContextBuilder, _Writer
from aquai.application.answering.measures import _Context
from aquai.application.conversations.presentation import _Presenter
from aquai.infrastructure.processing.tabular.normalization import (
    measurement_column,
    numeric_projection,
)
from aquai.infrastructure.queries.sources import _verified_dataset
from aquai.infrastructure.queries.textProjections import (
    projection_warning,
    text_projections,
    with_text_projections,
)

_NITRATE = {
    "sql_name": "n_no3_mg_n_l_1",
    "original_name": "N-NO3 [mg N L-1]",
    "sql_type": "TEXT",
    "unit": "mg N L-1",
    "numeric_eligible": False,
}
_SITE = {
    "sql_name": "sitios",
    "original_name": "Sitios",
    "sql_type": "TEXT",
    "numeric_eligible": False,
}
_STORED_VALUES = ("1.175345", "0.998", None, "<0,001")


class _Session:
    def __init__(self, *rows) -> None:
        self.rows = list(rows)
        self.statements = []

    def execute(self, statement) -> SimpleNamespace:
        self.statements.append(statement)
        return SimpleNamespace(all=lambda: self.rows)


def _dataset(*columns) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        source_id=uuid4(),
        semantic_point_id=uuid4(),
        name="Tabla completa ciano RdlP / tabla datos / A1:X282",
        semantic_description=None,
        description=None,
        physical_table_name="dataset_618040902949496a9be955c4b7ccfdbd",
        version=2,
        row_count=4,
        columns_metadata=[dict(item) for item in columns],
        source_locator={},
        quality_warnings=[],
    )


def _projections(values, column=_NITRATE) -> list[dict]:
    return text_projections(_Session(*((value,) for value in values)), _dataset(column))


def _verified(dataset, projections):
    return _verified_dataset(
        dataset,
        "Tabla completa ciano RdlP.xlsx",
        "Tabla completa ciano RdlP.xlsx",
        "xlsx",
        "a" * 64,
        "ready",
        "ready",
        {},
        projections,
    )


class TextProjectionTests(unittest.TestCase):
    def test_a_stored_measurement_gets_the_projection_of_the_ingestion_rule(self) -> None:
        (projection,) = _projections(_STORED_VALUES)

        self.assertEqual(projection["sql_name"], "n_no3_mg_n_l_1_numeric")
        self.assertEqual(projection["original_name"], "N-NO3 [mg N L-1] (representación numérica)")
        self.assertEqual(projection["derived_from"], "n_no3_mg_n_l_1")
        self.assertEqual(projection["sql_type"], "NUMERIC")
        self.assertTrue(projection["numeric_eligible"])
        self.assertEqual(projection["unit"], "mg N L-1")
        self.assertEqual(projection["excluded_count"], 1)
        self.assertEqual(projection["nonempty_count"], 3)

    def test_the_verified_catalog_adds_the_projection_without_touching_stored_metadata(self) -> None:
        dataset = _dataset(_SITE, _NITRATE)
        session = _Session(*((value,) for value in _STORED_VALUES))

        verified = _verified(dataset, text_projections(session, dataset))

        self.assertEqual(
            [column.sql_name for column in verified.columns_metadata],
            ["sitios", "n_no3_mg_n_l_1", "n_no3_mg_n_l_1_numeric"],
        )
        self.assertEqual(len(dataset.columns_metadata), 2)
        self.assertEqual(
            verified.warnings,
            [
                "N-NO3 [mg N L-1] (representación numérica): calculada al leer; 1 valores no "
                "convertibles conservados en el texto original y excluidos de cálculos."
            ],
        )

    def test_columns_that_are_not_pending_measurements_are_not_read(self) -> None:
        projected = {
            **_NITRATE,
            "sql_name": "n_no3_numeric",
            "sql_type": "NUMERIC",
            "numeric_eligible": True,
            "derived_from": _NITRATE["sql_name"],
        }
        session = _Session(("1.5",))

        self.assertEqual(text_projections(session, _dataset(_SITE, _NITRATE, projected)), [])
        self.assertEqual(session.statements, [])

    def test_only_columns_named_as_measurements_are_candidates(self) -> None:
        self.assertTrue(measurement_column("N-NO3 [mg N L-1]"))
        self.assertTrue(measurement_column("Abundancia absoluta (cél/ml)"))
        self.assertFalse(measurement_column("Sitios"))
        self.assertFalse(measurement_column("Código Sitio"))

    def test_text_that_the_rule_cannot_read_stays_without_projection(self) -> None:
        self.assertEqual(_projections(("1.5", "2.5", "12 abc", "muestra perdida 3")), [])

    def test_decimal_commas_stay_without_projection_because_sql_would_not_read_them(self) -> None:
        values = ("1,5", "2,25", "0,75", "n/d")

        self.assertIsNotNone(numeric_projection(_NITRATE["original_name"], list(values)))
        self.assertEqual(_projections(values), [])

    def test_physical_tables_are_read_as_they_are(self) -> None:
        table = Table("dataset_a", MetaData(), Column("n_no3_numeric", Numeric))
        dataset = SimpleNamespace(
            columns_metadata=[SimpleNamespace(sql_name="n_no3_numeric", derived_from="n_no3")]
        )

        self.assertIs(with_text_projections(table, dataset), table)

    def test_a_virtual_projection_reads_only_plain_numbers_from_the_text(self) -> None:
        table = Table(
            "dataset_a",
            MetaData(),
            Column("_aquai_row", BigInteger),
            Column("n_no3_mg_n_l_1", Text),
        )
        dataset = SimpleNamespace(
            columns_metadata=[
                SimpleNamespace(sql_name="n_no3_mg_n_l_1", derived_from=None),
                SimpleNamespace(sql_name="n_no3_mg_n_l_1_numeric", derived_from="n_no3_mg_n_l_1"),
            ]
        )

        source = with_text_projections(table, dataset)
        statement = select(source.c["n_no3_mg_n_l_1_numeric"]).compile(dialect=postgresql.dialect())

        self.assertIsInstance(source.c["n_no3_mg_n_l_1_numeric"].type, Numeric)
        self.assertIn("n_no3_mg_n_l_1", source.c)
        self.assertRegex(
            str(statement),
            r"CASE WHEN \(pg_catalog\.btrim\(.+\) ~ .+\) THEN CAST\(pg_catalog\.btrim\(.+\) AS NUMERIC\) END",
        )

    def test_the_rule_keeps_exact_decimals(self) -> None:
        numbers, excluded = numeric_projection(_NITRATE["original_name"], list(_STORED_VALUES))

        self.assertEqual(numbers, [Decimal("1.175345"), Decimal("0.998"), None, None])
        self.assertEqual(excluded, 1)

    def test_the_warning_names_the_projection_and_its_exclusions(self) -> None:
        warning = projection_warning({"original_name": "Nitritos (mg/l)", "excluded_count": 2})

        self.assertEqual(
            warning,
            "Nitritos (mg/l): calculada al leer; 2 valores no convertibles conservados en el texto "
            "original y excluidos de cálculos.",
        )


class ProjectionLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_id = uuid4()
        self.origin = SimpleNamespace(
            sql_name="n_no3_mg_n_l_1",
            original_name="N-NO3 [mg N L-1]",
            unit="mg N L-1",
            derived_from=None,
        )
        self.projection = SimpleNamespace(
            sql_name="n_no3_mg_n_l_1_numeric",
            original_name="N-NO3 [mg N L-1] (representación numérica)",
            unit="mg N L-1",
            derived_from="n_no3_mg_n_l_1",
        )

    def _measures(self, *columns) -> _Context:
        measures = _Context.__new__(_Context)
        measures._columns = {(self.dataset_id, item.sql_name): item for item in columns}
        measures._mentions = []

        return measures

    def test_result_sentences_name_a_projection_like_its_origin(self) -> None:
        dataset = SimpleNamespace(columns_metadata=[self.origin, self.projection])
        writer = _Writer({self.dataset_id: dataset}, {}, "es", False)

        self.assertEqual(writer.column(self.dataset_id, self.projection.sql_name), "N-NO3")
        self.assertEqual(writer.column(self.dataset_id, self.origin.sql_name), "N-NO3")

    def test_a_projection_without_its_origin_keeps_its_own_name(self) -> None:
        dataset = SimpleNamespace(columns_metadata=[self.projection])
        writer = _Writer({self.dataset_id: dataset}, {}, "es", False)

        self.assertEqual(
            writer.column(self.dataset_id, self.projection.sql_name),
            "N-NO3 [mg N L-1] (representación numérica)",
        )

    def test_measured_results_are_named_by_the_origin_column(self) -> None:
        measures = self._measures(self.origin, self.projection)

        self.assertEqual(measures._label(self.dataset_id, self.projection.sql_name), "N-NO3")
        self.assertIs(measures._named(self.dataset_id, self.projection.sql_name), self.origin)
        self.assertIsNone(measures._label(self.dataset_id, "otra_columna"))

    def test_stored_answers_show_the_origin_label_in_tables_and_citations(self) -> None:
        view = _Presenter.__new__(_Presenter)
        view._columns = {
            (self.dataset_id, item.sql_name): vars(item) for item in (self.origin, self.projection)
        }

        self.assertEqual(view._column(self.dataset_id, self.projection.sql_name), "N-NO3")
        self.assertIsNone(view._column(self.dataset_id, "otra_columna"))


class ProjectionNoteTests(unittest.TestCase):
    def _builder(self, projections, exclusions) -> _ContextBuilder:
        builder = _ContextBuilder.__new__(_ContextBuilder)
        builder._catalog = []
        builder._writer = SimpleNamespace(
            projections=lambda record: projections,
            exclusions=lambda record, origin, fields: exclusions,
        )
        builder._add = lambda evidence, text, *, kind: builder._catalog.append(
            {"direct": text, "kind": kind, "evidence_ids": [evidence]}
        )

        return builder

    def test_two_results_over_the_same_projection_share_one_note(self) -> None:
        builder = self._builder(["La columna N-NO3 se usa en su versión numérica."], [])

        builder._add_details("e1", None, None, None)
        builder._add_details("e2", None, None, None)

        self.assertEqual([entry["evidence_ids"] for entry in builder._catalog], [["e1"]])

    def test_each_result_keeps_its_own_exclusions(self) -> None:
        builder = self._builder([], ["Se excluyeron 2 registros sin valor."])

        builder._add_details("e1", None, None, None)
        builder._add_details("e2", None, None, None)

        self.assertEqual([entry["evidence_ids"] for entry in builder._catalog], [["e1"], ["e2"]])


if __name__ == "__main__":
    unittest.main()
