import unittest
from uuid import uuid4

from aquai.application.contracts.retrieval import (
    DatasetProvenance,
    RetrievedColumn,
    VerifiedDataset,
)
from aquai.application.contracts.routing import DataMention
from aquai.application.Services.retrieveDatasetsService import RetrieveDatasetsService
from aquai.domain.enums.sourceStatus import ProcessingStatus, SemanticIndexStatus

_CELLS = "abundancia_absoluta_cel_ml"
_CELLS_NUMERIC = "abundancia_absoluta_cel_ml_numeric"
_INDIVIDUALS = "abundancia_absoluta_ind_ml"
_COLONIES = "abundancia_absoluta_colonias_ml"
_WATER_TEMPERATURE = "temperatura_del_agua_c"
_AMBIENT_TEMPERATURE = "temperatura_ambiente_c"
_TEMPERATURE_SYMBOL = "t_c"
_TURBIDITY = "turbiedad_ntu"
_NITRATE = "nitratos_mg_l"
_NITRATE_AS_NITROGEN = "n_no3_mg_n_l_1"
_RIVER = "rio_plata"
_SITE = "sitio"
_SYMBOL_NOTE = "reconocidas por su símbolo y su unidad"
_UNIT_NOTE = "reconocidas por la unidad pedida"


def _column(original_name, sql_name, sql_type="TEXT", **details) -> dict:
    return {
        "original_name": original_name,
        "sql_name": sql_name,
        "sql_type": sql_type,
        "numeric_eligible": sql_type == "NUMERIC",
        **details,
    }


def _dataset(name, *columns) -> VerifiedDataset:
    return VerifiedDataset(
        dataset_id=uuid4(),
        source_id=uuid4(),
        semantic_point_id=uuid4(),
        name=f"Fuente / {name} / A1:C10",
        description=None,
        physical_table_name=f"dataset_{uuid4().hex}",
        verified_version=1,
        row_count=10,
        columns_metadata=[RetrievedColumn(**column) for column in columns],
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


def _plankton() -> VerifiedDataset:
    return _dataset(
        "BASE CIANO",
        _column("Abundancia absoluta (cél/ml)", _CELLS, unit="cél/ml"),
        _column(
            "Abundancia absoluta (cél/ml) (representación numérica)",
            _CELLS_NUMERIC,
            "NUMERIC",
            unit="cél/ml",
            derived_from=_CELLS,
        ),
        _column("Abundancia absoluta (ind/ml)", _INDIVIDUALS, unit="ind/ml"),
        _column("Abundancia absoluta (colonias/ml)", _COLONIES, unit="colonias/ml"),
    )


def _chemistry() -> VerifiedDataset:
    return _dataset(
        "BASE FQ",
        _column("Temperatura del agua (°C)", _WATER_TEMPERATURE, "NUMERIC", unit="°C"),
        _column("Turbiedad (NTU)", _TURBIDITY, "NUMERIC", unit="NTU"),
    )


def _temperatures() -> VerifiedDataset:
    return _dataset(
        "TEMPERATURAS",
        _column("Temperatura del agua (°C)", _WATER_TEMPERATURE, "NUMERIC", unit="°C"),
        _column("Temperatura ambiente (°C)", _AMBIENT_TEMPERATURE, "NUMERIC", unit="°C"),
        _column("T (°C)", _TEMPERATURE_SYMBOL, "NUMERIC", unit="°C"),
    )


def _nutrients() -> VerifiedDataset:
    return _dataset(
        "NUTRIENTES",
        _column("Nitratos (mg/L)", _NITRATE, "NUMERIC", unit="mg/L"),
        _column("N-NO3 [mg N L-1]", _NITRATE_AS_NITROGEN, "NUMERIC", unit="mg N L-1"),
    )


def _places() -> VerifiedDataset:
    return _dataset(
        "SITIOS",
        _column("Río Plata", _RIVER, "NUMERIC"),
        _column("Turbiedad (NTU)", _TURBIDITY, "NUMERIC", unit="NTU"),
        _column("Sitio", _SITE),
    )


class _Index:
    def __init__(self, datasets) -> None:
        self._datasets = datasets

    async def search(self, query, *, limit):
        return [
            {
                "dataset_id": str(dataset.dataset_id),
                "source_id": str(dataset.source_id),
                "semantic_point_id": str(dataset.semantic_point_id),
                "version": dataset.verified_version,
                "score": 0.5,
                "embedding_model": "embedding",
            }
            for dataset in self._datasets
        ]


class _Queries:
    def __init__(self, datasets, stored) -> None:
        self._datasets = datasets
        self._stored = stored

    def verified_datasets(self, dataset_ids, versions=None):
        return self._datasets

    def find_text_values(self, datasets, terms, *, per_column=50):
        return self._stored


async def _retrieve(mentions, *datasets, stored=None):
    service = RetrieveDatasetsService(_Index(datasets), _Queries(datasets, stored or {}))

    return await service.retrieve("Pregunta", data_mentions=mentions)


async def _columns(mention, *datasets) -> list[str]:
    result = await _retrieve([DataMention(text=mention, kind="variable")], *datasets)

    return [
        column
        for item in result.evidence
        for columns in item.variable_matches.values()
        for column in columns
    ]


class VariableConnectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unit_after_a_connector_keeps_only_the_dataset_with_that_column(self):
        mention = "abundancia absoluta en cél/ml"
        plankton = _plankton()

        result = await _retrieve(
            [DataMention(text=mention, kind="variable")], _chemistry(), plankton, _places()
        )

        self.assertEqual([item.dataset_id for item in result.candidates], [plankton.dataset_id])
        self.assertEqual(
            [item.variable_matches for item in result.evidence],
            [{mention: [_CELLS, _CELLS_NUMERIC]}],
        )

    async def test_each_unit_stays_with_its_own_column(self):
        self.assertEqual(await _columns("abundancia absoluta en ind/ml", _plankton()), [_INDIVIDUALS])

    async def test_connectors_of_the_mention_and_the_header_do_not_have_to_agree(self):
        for mention in ("temperatura del agua en °C", "temperatura de agua", "temperatura en el agua"):
            with self.subTest(mention=mention):
                self.assertEqual(await _columns(mention, _chemistry()), [_WATER_TEMPERATURE])

    async def test_words_stay_in_order_and_together(self):
        for mention in ("abundancia de colonias", "absoluta en abundancia", "ml de abundancia absoluta"):
            with self.subTest(mention=mention):
                self.assertEqual(await _columns(mention, _plankton()), [])

    async def test_a_mention_without_inner_connector_stays_literal(self):
        self.assertEqual(await _columns("temperatura agua", _chemistry()), [])

    async def test_a_value_is_not_read_as_a_column_without_its_connectors(self):
        value = "Río de la Plata"
        places = _places()
        rows = [{"column": _SITE, "value": value, "rows": 3}]

        result = await _retrieve(
            [
                DataMention(text="turbiedad", kind="variable"),
                DataMention(text=value, kind="value"),
            ],
            places,
            stored={places.dataset_id: {value: rows}},
        )

        self.assertEqual(result.dimension_mentions, [])
        self.assertEqual(result.evidence[0].value_matches[value][0].column, _SITE)

    async def test_the_note_does_not_blame_a_symbol(self):
        result = await _retrieve(
            [DataMention(text="abundancia absoluta en cél/ml", kind="variable")], _plankton()
        )

        self.assertFalse([note for note in result.warnings if _SYMBOL_NOTE in note])

    async def test_a_synonym_keeps_working_with_a_connector(self):
        result = await _retrieve([DataMention(text="turbidez en NTU", kind="variable")], _chemistry())

        self.assertEqual(result.evidence[0].variable_matches, {"turbidez en NTU": [_TURBIDITY]})
        self.assertTrue(
            [note for note in result.warnings if "reconocidas por un nombre equivalente" in note]
        )


class VariableUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_unit_finds_the_column_of_a_shorter_name(self):
        for mention in (
            "abundancia en cél/ml",
            "abundancia (cél/ml)",
            "abundancia cél/ml",
            "abundancia máxima en cél/ml",
        ):
            with self.subTest(mention=mention):
                self.assertEqual(await _columns(mention, _plankton()), [_CELLS, _CELLS_NUMERIC])

    async def test_another_spelling_is_the_same_unit(self):
        for mention in (
            "abundancia absoluta en células/ml",
            "abundancia absoluta en cells/mL",
            "abundancia absoluta en cél ml-1",
            "abundancia absoluta en células por ml",
            "abundancia absoluta (células/ml)",
            "abundancia absoluta células/ml",
        ):
            with self.subTest(mention=mention):
                self.assertEqual(await _columns(mention, _plankton()), [_CELLS, _CELLS_NUMERIC])

    async def test_each_unit_keeps_its_own_column_with_a_shorter_name(self):
        self.assertEqual(await _columns("abundancia en ind/ml", _plankton()), [_INDIVIDUALS])

    async def test_a_unit_of_no_column_finds_nothing(self):
        self.assertEqual(await _columns("abundancia en mg/L", _plankton()), [])

    async def test_the_unit_does_not_replace_the_name(self):
        self.assertEqual(await _columns("clorofila en cél/ml", _plankton()), [])

    async def test_a_place_is_not_a_unit(self):
        self.assertEqual(await _columns("abundancia en Punta Lara", _plankton()), [])

    async def test_the_chemical_basis_is_part_of_the_unit(self):
        self.assertEqual(await _columns("nitrato en mg/L", _nutrients()), [_NITRATE])
        self.assertEqual(await _columns("nitrato en mg N/L", _nutrients()), [_NITRATE_AS_NITROGEN])

    async def test_a_symbol_counts_as_the_name(self):
        self.assertEqual(
            await _columns("temperatura en °C", _temperatures()),
            [_WATER_TEMPERATURE, _TEMPERATURE_SYMBOL],
        )

    async def test_the_medium_still_decides_with_a_unit(self):
        self.assertEqual(
            await _columns("temperatura del aire en °C", _temperatures()), [_AMBIENT_TEMPERATURE]
        )

    async def test_the_note_says_the_unit_found_the_column(self):
        result = await _retrieve(
            [DataMention(text="abundancia en cél/ml", kind="variable")], _plankton()
        )

        self.assertTrue([note for note in result.warnings if _UNIT_NOTE in note])
        self.assertFalse([note for note in result.warnings if _SYMBOL_NOTE in note])


if __name__ == "__main__":
    unittest.main()
