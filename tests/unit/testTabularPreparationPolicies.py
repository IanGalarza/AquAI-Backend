import json
import io
import unittest
from copy import deepcopy
from hashlib import sha256

from pydantic import ValidationError
from openpyxl import Workbook

from aquai.application.dtos.tabular import (
    ExecutionResult,
    FormulaSample,
    LLMResponse,
    PreparationRecipe,
    SheetExecution,
    SheetSummary,
    StructuralRegion,
    StructuralSummary,
)
from aquai.application.dtos.tabularPolicies import (
    AssociationDiagnostic,
    AssociationPolicy,
    AuxiliaryFormulaEvidence,
    AuxiliaryRowTreatment,
    DateColumnPolicy,
    ObservationRowPolicy,
    PreparationPolicies,
    SourceHeaderEvidence,
)
from aquai.application.errors import TabularInputError
from aquai.application.services.tabularRules import TabularRuleService
from aquai.application.services.tabularValidation import TabularValidationService
from aquai.application.services.tabularEvaluation import TabularEvaluationService
from aquai.domain.enums.tabular import EvaluationStatus
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog
from aquai.infrastructure.processing.tabular import PandasTabularProcessor


class MemoryCatalog:
    def __init__(self, value):
        self.value = value

    def load(self):
        return deepcopy(self.value)


def source_summary(name="reviewed.csv"):
    return StructuralSummary(
        file_name=name, file_type="csv", file_size_bytes=20,
        sheets=[SheetSummary(
            name="data", rows=3, columns=3, initial_columns=["id", "Fecha", "count"],
            inferred_types=["string", "string", "number"], null_counts=[0, 0, 0], sample=[],
        )],
    )


def recipe_for(role="observations", transformations=None, start_column=1):
    return PreparationRecipe.model_validate({
        "status": "review_required" if role == "unknown" else "ready",
        "warnings": ["El rol del bloque necesita revisión."] if role == "unknown" else [],
        "requires_human_review": role == "unknown",
        "sheets": [{
            "sheet_name": "data", "block_id": "records", "role": role,
            "header_rows": [] if role == "header" else [1], "data_start_row": 2, "data_end_row": 3,
            "data_start_column": start_column, "data_end_column": 3,
            "transformations": transformations or [],
        }],
    })


def complete_headers(summary):
    return [SourceHeaderEvidence(
        sheet_name=sheet.name, row_number=1, inspected_columns=sheet.columns,
        values=dict(enumerate(sheet.initial_columns, 1)),
    ) for sheet in summary.sheets]


def fixture_catalog():
    return {
        "version": "1.0.0", "common": [],
        "sources": {"trusted_user": {"verification": "user_confirmation"}},
        "profiles": {"fixture": {
            "name": "Fixture", "file_names": ["reviewed.csv"], "required_sheets": ["data"],
            "rules": [{
                "id": "fixture_formats", "version": "2.0.0", "origin": "trusted_user",
                "text": "Preparación determinística corroborada.",
                "preparation_policies": {
                    "date_columns": [{
                        "sheet_name": "data", "column_indices": [2], "expected_headers": {"2": "Fecha"},
                        "text_formats": ["%d-%m-%Y", "%Y-%m-%d"],
                    }],
                    "number_columns": [{
                        "sheet_name": "data", "column_indices": [3], "expected_headers": {"3": "count"},
                    }],
                    "observation_rows": [{
                        "sheet_name": "data", "auxiliary_column_indices": [1], "expected_headers": {"1": "id"},
                    }],
                },
            }],
        }},
    }


class PreparationPolicyCompositionTests(unittest.TestCase):
    def test_catalog_policies_are_typed_versioned_and_copied_without_mutating_catalog(self):
        catalog = fixture_catalog()
        before = deepcopy(catalog)
        service = TabularRuleService(MemoryCatalog(catalog))
        composition = service.compose(source_summary(), "fixture")
        policies = composition.preparation_policies
        self.assertTrue(policies.enabled)
        self.assertIsInstance(policies.date_columns[0], DateColumnPolicy)
        self.assertEqual(policies.date_columns[0].version, "2.0.0")
        self.assertEqual(policies.date_columns[0].rule_id, "fixture_formats")
        self.assertEqual(policies.date_columns[0].expected_headers, {2: "Fecha"})
        self.assertEqual(catalog, before)

    def test_unverified_profile_and_known_rule_text_cannot_authorize_system_policies(self):
        service = TabularRuleService(MemoryCatalog(fixture_catalog()))
        additional = json.dumps({"preparation_policies": fixture_catalog()["profiles"]["fixture"]["rules"][0]["preparation_policies"]})
        self.assertIsNone(service.compose(source_summary("other.csv"), "fixture", additional).preparation_policies)
        self.assertIsNone(service.compose(source_summary(), None, additional).preparation_policies)
        missing_sheet = source_summary()
        missing_sheet.sheets[0].name = "Other"
        self.assertIsNone(service.compose(missing_sheet, "fixture", additional).preparation_policies)

    def test_exact_cell_confirmation_requires_matching_hash_and_human_file_rule(self):
        catalog = fixture_catalog()
        catalog["file_specific_rules"] = [{
            "id": "confirmed_cell", "version": "3.0.0", "origin": "trusted_user",
            "profile": "fixture", "file_name": "reviewed.csv", "sha256": "a" * 64,
            "text": "Una celda confirmada.", "preparation_policies": {"confirmed_cells": [{
                "sheet_name": "data", "cell": "C2", "expected_source_value": "2023-05-02T00:00:00",
                "expected_number_format": r"d\.m", "expected_source_kind": "native_datetime",
                "expected_raw_value": "45048", "expected_header": "count", "prepared_value": 2.5,
            }]},
        }]
        service = TabularRuleService(MemoryCatalog(catalog))
        matched = service.compose(source_summary(), "fixture", source_sha256="a" * 64)
        self.assertEqual(matched.preparation_policies.confirmed_cells[0].source_sha256, "a" * 64)
        self.assertEqual(matched.preparation_policies.confirmed_cells[0].version, "3.0.0")
        self.assertEqual(service.compose(source_summary(), "fixture", source_sha256="b" * 64).preparation_policies.confirmed_cells, ())
        catalog["sources"]["trusted_user"]["verification"] = "unverified_text"
        with self.assertRaises(TabularInputError):
            TabularRuleService(MemoryCatalog(catalog)).compose(source_summary(), "fixture", source_sha256="a" * 64)

    def test_profile_level_cell_override_is_rejected_even_if_origin_is_human(self):
        catalog = fixture_catalog()
        catalog["profiles"]["fixture"]["rules"][0]["preparation_policies"]["confirmed_cells"] = [{}]
        with self.assertRaises(TabularInputError):
            TabularRuleService(MemoryCatalog(catalog)).compose(source_summary(), "fixture")

    def test_date_policy_rejects_inferred_american_order_or_unguarded_related_columns(self):
        base = dict(rule_id="dates", version="1", origin="user", sheet_name="data",
                    column_indices=[2], expected_headers={2: "Fecha"}, text_formats=["%d-%m-%Y"])
        for changes in ({"text_formats": ["%m-%d-%Y"]}, {"related_date_columns": {"month": 4}},
                        {"related_date_code_formats": ["%Y%m%d"]}, {"related_date_code_formats": ["%m%Y"]}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                DateColumnPolicy.model_validate({**base, **changes})

    def test_numeric_columns_cannot_be_sent_to_date_conversion_even_after_rename(self):
        service = TabularRuleService(MemoryCatalog(fixture_catalog()))
        source = source_summary()
        recipe = recipe_for(transformations=[
            {"operation": "rename_columns", "mapping": {"count": "abundance"}},
            {"operation": "convert_type", "columns": ["abundance"], "target_type": "datetime"},
        ])
        findings = service.validate_recipe(source, recipe, service.compose(source, "fixture"))
        self.assertEqual([finding["code"] for finding in findings], ["rule_policy_type_conflict"])
        self.assertTrue(findings[0]["requires_review"])

    def test_calculated_record_identifiers_are_not_separated_from_observation_fields(self):
        service = TabularRuleService(MemoryCatalog(fixture_catalog()))
        source = source_summary()
        findings = service.validate_recipe(source, recipe_for(start_column=2), service.compose(source, "fixture"))
        self.assertEqual([finding["code"] for finding in findings], ["rule_policy_identifier_separation"])

    def test_existing_profiles_have_no_implicit_belen_preparation_policies(self):
        service = TabularRuleService(FileTabularRuleCatalog())
        for profile, filename, sheets in (
            ("self_sampling", "Tabla completa ciano RdlP.xlsx", ["data"]),
            ("biological_counts", "datos biologicos tesis.xlsx", ["Recuentos", "Cianobacterias", "Hoja3", "Hoja1"]),
        ):
            with self.subTest(profile=profile):
                source = source_summary(filename)
                source.sheets = [source.sheets[0].model_copy(update={"name": sheet}) for sheet in sheets]
                self.assertIsNone(service.compose(source, profile).preparation_policies)


class AuxiliaryRowValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = source_summary()
        self.source.sheets[0].formula_count = 2
        self.source.sheets[0].missing_formula_cache_count = 1
        self.source.sheets[0].formula_sample = [FormulaSample(cell="A3", formula="=B3&C3")]
        self.source.sheets[0].regions = [StructuralRegion(
            region_id="calculated_id", start_row=2, end_row=3, start_column=1, end_column=1,
            kind="derived", reason="La columna contiene fórmulas de identificación.",
        )]
        self.recipe = recipe_for()
        self.policies = PreparationPolicies(observation_rows=[ObservationRowPolicy(
            sheet_name="data", auxiliary_column_indices=[1], expected_headers={1: "id"},
            rule_id="auxiliary", version="1", origin="user",
        )])
        self.execution = ExecutionResult(
            sheets=[SheetExecution(
                sheet_name="data", block_id="records", role="observations",
                data_start_row=2, data_end_row=3, data_start_column=1, data_end_column=3,
                source_rows=3, source_columns=3, rows_before=2, rows_after=1,
                columns_before=["id", "Fecha", "count"], columns_after=["id", "Fecha", "count"],
                source_nonempty_cells=4, preserved_nonempty_cells=3, auxiliary_rows_removed=1,
            )],
            auxiliary_rows=[AuxiliaryRowTreatment(
                sheet_name="data", block_id="records", row_number=3, rule_id="auxiliary", version="1", origin="user",
                inspected_columns=3, auxiliary_cells=[AuxiliaryFormulaEvidence(cell="A3", formula="=B3&C3")],
            )],
            source_headers=complete_headers(self.source),
        )

    def validate(self, policies=True):
        return TabularValidationService().validate(
            self.source, self.recipe, self.execution,
            preparation_policies=self.policies if policies else None,
        )

    def test_proven_auxiliary_row_is_not_observation_or_global_missing_cache_blocker(self):
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.review_reasons, [])
        self.assertFalse(any(finding.code == "source_missing_formula_cache" for finding in result.findings))
        self.assertTrue(any(finding.code == "auxiliary_row_preserved" for finding in result.findings))
        self.assertEqual(self.execution.auxiliary_rows[0].auxiliary_cells[0].formula, "=B3&C3")
        self.assertEqual(self.source.sheets[0].missing_formula_cache_count, 1)

    def test_auxiliary_row_exemption_requires_policy_complete_scan_and_matching_count(self):
        for defect in ("policy", "scan", "count", "header"):
            with self.subTest(defect=defect):
                self.setUp()
                if defect == "scan":
                    self.execution.auxiliary_rows[0].inspected_columns = 2
                elif defect == "count":
                    self.execution.sheets[0].auxiliary_rows_removed = 2
                elif defect == "header":
                    self.source.sheets[0].initial_columns[0] = "ordinary_measurement"
                result = self.validate(policies=defect != "policy")
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(result.errors)

    def test_another_missing_observational_formula_still_requires_review(self):
        self.source.sheets[0].missing_formula_cache_count = 2
        self.source.sheets[0].formula_sample.append(FormulaSample(cell="C2", formula="=1+1"))
        self.execution.sheets[0].unresolved_formula_count = 1
        self.execution.sheets[0].unresolved_formula_cells = ["C2"]
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        finding = next(finding for finding in result.review_reasons if finding.code == "source_missing_formula_cache")
        self.assertEqual(finding.cell, "C2")
        self.assertIn("1", finding.message)

    def test_auxiliary_evidence_cannot_cover_any_other_record_loss(self):
        self.execution.sheets[0].preserved_nonempty_cells = 2
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any(finding.code == "nonempty_value_loss" for finding in result.review_reasons))

    def test_auxiliary_formula_evidence_cannot_contradict_the_original_formula_sample(self):
        for changes in ({"formula": "=1+1"}, {"cached_value": 42, "cache_available": True}):
            with self.subTest(changes=changes):
                self.setUp()
                self.source.sheets[0].formula_sample[0] = self.source.sheets[0].formula_sample[0].model_copy(update=changes)
                result = self.validate()
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(any(finding.code == "auxiliary_row_evidence_mismatch" for finding in result.findings))

    def test_documentary_auxiliary_formula_row_keeps_its_table_and_values(self):
        for role in ("derived", "legend", "unknown"):
            with self.subTest(role=role):
                self.setUp()
                self.recipe = recipe_for(role=role)
                self.execution.sheets[0].role = role
                self.execution.sheets[0].rows_after = 2
                self.execution.sheets[0].preserved_nonempty_cells = 4
                self.execution.sheets[0].auxiliary_rows_removed = 0
                self.execution.auxiliary_rows[0].removed_from_observations = False
                self.source.sheets[0].regions = []
                result = self.validate()
                self.assertFalse(result.errors)
                self.assertFalse(any(finding.code in {"source_missing_formula_cache", "nonempty_value_loss", "unexplained_row_loss"}
                                     for finding in result.findings))
                self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED if role == "unknown"
                                 else EvaluationStatus.READY_WITH_WARNINGS)
                self.assertEqual(self.execution.sheets[0].rows_after, 2)

    def test_documentary_or_header_blocks_cannot_claim_removed_observation_rows(self):
        for role in ("derived", "legend", "header"):
            with self.subTest(role=role):
                self.setUp()
                self.recipe = recipe_for(role=role)
                self.execution.sheets[0].role = role
                result = self.validate()
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(any(finding.code == "auxiliary_row_evidence_mismatch" for finding in result.findings))

    def test_old_execution_serialization_omits_new_empty_trace_fields(self):
        dumped = ExecutionResult().model_dump()
        for key in ("normalizations", "auxiliary_rows", "associations", "source_headers"):
            self.assertNotIn(key, dumped)
        plain_sheet = self.execution.sheets[0].model_copy(update={"auxiliary_rows_removed": 0})
        self.assertNotIn("auxiliary_rows_removed", plain_sheet.model_dump())


class AssociationValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = source_summary()
        self.source.sheets[0].initial_columns[1] = "link"
        self.source.sheets.append(SheetSummary(
            name="Chemistry", rows=4, columns=2, initial_columns=["id", "value"],
            inferred_types=["string", "number"], null_counts=[0, 0], sample=[],
        ))
        raw_recipe = recipe_for().model_dump()
        raw_recipe["sheets"].append({
            **raw_recipe["sheets"][0], "sheet_name": "Chemistry", "block_id": "measurements",
            "data_end_row": 4, "data_end_column": 2,
        })
        self.recipe = PreparationRecipe.model_validate(raw_recipe)
        self.policies = PreparationPolicies(associations=[AssociationPolicy(
            rule_id="optional_measurement", version="1", origin="user",
            source_sheet="data", source_column=2, expected_source_header="link",
            target_sheet="Chemistry", target_column=1, expected_target_header="id",
        )])
        self.execution = ExecutionResult(sheets=[SheetExecution(
            sheet_name=block.sheet_name, block_id=block.block_id, role="observations",
            data_start_row=block.data_start_row, data_end_row=block.data_end_row,
            data_start_column=block.data_start_column, data_end_column=block.data_end_column,
            source_rows=sheet.rows, source_columns=sheet.columns,
            rows_before=sheet.rows - 1, rows_after=sheet.rows - 1,
            columns_before=sheet.initial_columns, columns_after=sheet.initial_columns,
        ) for sheet, block in zip(self.source.sheets, self.recipe.sheets, strict=True)],
            source_headers=complete_headers(self.source))

    def add_diagnostic(self, status, target_rows=(), requires_review=False):
        self.execution.associations.append(AssociationDiagnostic(
            source_sheet="data", source_row=2, source_cell="B2", source_value="measurement",
            target_sheet="Chemistry", target_rows=target_rows, status=status,
            requires_review=requires_review, rule_id="optional_measurement", version="1", origin="user",
        ))

    def validate(self, policies=True):
        return TabularValidationService().validate(
            self.source, self.recipe, self.execution,
            preparation_policies=self.policies if policies else None,
        )

    def test_missing_optional_link_keeps_the_observation_and_does_not_require_review(self):
        self.add_diagnostic("missing")
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.READY)
        self.assertEqual(result.review_reasons, [])
        self.assertEqual(self.execution.sheets[0].rows_after, 2)
        self.assertEqual(self.execution.associations[0].source_value, "measurement")

    def test_multiple_destinations_require_concrete_review_even_if_trace_flag_is_false(self):
        self.add_diagnostic("ambiguous", (2, 3))
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        finding = next(finding for finding in result.review_reasons if finding.code == "ambiguous_optional_association")
        self.assertEqual((finding.sheet_name, finding.cell), ("data", "B2"))
        self.assertIn("2, 3", finding.message)
        self.assertEqual(self.execution.sheets[0].rows_after, 2)

    def test_diagnostics_cannot_bypass_policy_or_source_coordinates(self):
        for defect in ("policy", "coordinate", "header", "status", "duplicate_destinations"):
            with self.subTest(defect=defect):
                self.setUp()
                self.add_diagnostic("matched", (2,))
                if defect == "coordinate":
                    self.execution.associations[0].source_cell = "C2"
                elif defect == "header":
                    self.source.sheets[0].initial_columns[1] = "ordinary_code"
                elif defect == "status":
                    self.execution.associations[0].status = "missing"
                elif defect == "duplicate_destinations":
                    self.execution.associations[0].status = "ambiguous"
                    self.execution.associations[0].target_rows = (2, 2)
                result = self.validate(policies=defect != "policy")
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(any(finding.code == "association_evidence_mismatch" for finding in result.findings))


class AuxiliaryLegendIntegrationTests(unittest.TestCase):
    def execute_fixture(self, *, role="legend", other_value=None):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Samples"
        sheet.append(["id", "abundance"])
        sheet.append(["=B2"])
        sheet.append(["=B3", other_value])
        stream = io.BytesIO()
        workbook.save(stream)
        workbook.close()
        content = stream.getvalue()
        recipe = PreparationRecipe.model_validate({
            "status": "review_required" if role == "unknown" else "ready",
            "requires_human_review": role == "unknown",
            "warnings": ["El rol requiere revisión."] if role == "unknown" else [],
            "sheets": [{
                "sheet_name": "Samples", "block_id": "auxiliary", "role": role, "header_rows": [1],
                "data_start_row": 2, "data_end_row": 3, "data_start_column": 1, "data_end_column": 2,
                "transformations": [],
            }],
        })
        policies = PreparationPolicies(observation_rows=[ObservationRowPolicy(
            sheet_name="Samples", auxiliary_column_indices=[1], expected_headers={1: "id"},
            rule_id="auxiliary", version="1", origin="user",
        )])
        processor = PandasTabularProcessor()
        summary = processor.inspect("generic.xlsx", content)
        execution = processor.execute("generic.xlsx", content, recipe, summary, policies=policies)
        self.assertTrue(any(region.kind == "derived" for region in summary.sheets[0].regions))
        return summary, recipe, execution, policies

    def test_real_derived_candidate_does_not_block_a_fully_verified_auxiliary_legend(self):
        summary, recipe, execution, policies = self.execute_fixture()
        original_recipe = recipe.model_dump_json()
        result = TabularValidationService().validate(summary, recipe, execution, preparation_policies=policies)
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.review_reasons, [])
        self.assertEqual((execution.sheets[0].rows_after, execution.sheets[0].role), (2, "legend"))
        self.assertEqual(execution.sheets[0].source_nonempty_cells, execution.sheets[0].preserved_nonempty_cells)
        self.assertEqual(len(execution.auxiliary_rows), 2)
        self.assertTrue(all(not item.removed_from_observations for item in execution.auxiliary_rows))
        self.assertEqual(recipe.model_dump_json(), original_recipe)

    def test_unknown_role_still_requires_review_with_the_same_complete_auxiliary_proof(self):
        summary, recipe, execution, policies = self.execute_fixture(role="unknown")
        result = TabularValidationService().validate(summary, recipe, execution, preparation_policies=policies)
        self.assertEqual(result.status, EvaluationStatus.REVIEW_REQUIRED)
        self.assertTrue(any(finding.code == "unresolved_block_role" for finding in result.review_reasons))
        self.assertEqual(execution.sheets[0].rows_after, 2)

    def test_mixed_or_incompletely_proven_derived_area_keeps_the_role_conflict(self):
        for defect in ("mixed", "incomplete"):
            with self.subTest(defect=defect):
                summary, recipe, execution, policies = self.execute_fixture(other_value=0 if defect == "mixed" else None)
                if defect == "incomplete":
                    execution.auxiliary_rows[0].inspected_columns = 1
                result = TabularValidationService().validate(summary, recipe, execution, preparation_policies=policies)
                self.assertIn(result.status, {EvaluationStatus.REVIEW_REQUIRED, EvaluationStatus.FAILED})
                self.assertTrue(any(finding.code == "structural_role_conflict" for finding in result.review_reasons))
                self.assertEqual(execution.sheets[0].rows_after, 2)


class CompleteSourceHeaderTests(unittest.TestCase):
    def setUp(self):
        workbook = Workbook()
        source = workbook.active
        source.title = "Wide"
        names = [f"field_{index}" for index in range(1, 35)]
        names[0], names[27] = "id", "measurement_link"
        source.append(names)
        values = [None] * 34
        values[0], values[27] = "observation", "measurement"
        source.append(values)
        target = workbook.create_sheet("Measurements")
        target.append(["id", "value"])
        target.append(["measurement", 0])
        stream = io.BytesIO()
        workbook.save(stream)
        workbook.close()
        content = stream.getvalue()
        self.recipe = PreparationRecipe.model_validate({
            "status": "ready", "requires_human_review": False, "warnings": [],
            "sheets": [{
                "sheet_name": name, "block_id": f"{name}_records", "role": "observations", "header_rows": [1],
                "data_start_row": 2, "data_end_row": 2, "data_start_column": 1,
                "data_end_column": width, "transformations": [],
            } for name, width in (("Wide", 34), ("Measurements", 2))],
        })
        self.policies = PreparationPolicies(associations=[AssociationPolicy(
            rule_id="documented_link", version="1", origin="user",
            source_sheet="Wide", source_column=28, expected_source_header="measurement_link",
            target_sheet="Measurements", target_column=1, expected_target_header="id",
        )])
        processor = PandasTabularProcessor()
        self.summary = processor.inspect("generic.xlsx", content)
        self.execution = processor.execute("generic.xlsx", content, self.recipe, self.summary, policies=self.policies)
        self.assertNotIn(28, self.summary.sheets[0].column_indices)

    def validate(self):
        return TabularValidationService().validate(
            self.summary, self.recipe, self.execution, preparation_policies=self.policies,
        )

    def test_header_outside_compact_summary_is_verified_by_complete_original_evidence(self):
        result = self.validate()
        self.assertEqual(result.status, EvaluationStatus.READY_WITH_WARNINGS)
        self.assertEqual(result.review_reasons, [])
        proof = next(proof for proof in self.execution.source_headers if proof.sheet_name == "Wide")
        self.assertEqual(proof.values[28], "measurement_link")
        self.assertEqual(set(proof.values), set(range(1, 35)))
        self.assertEqual(self.execution.associations[0].target_rows, (2,))

    def test_missing_incomplete_or_contradictory_header_proof_cannot_authorize_the_link(self):
        for defect in ("missing", "incomplete", "dimensions", "expected_label", "compact_label", "formula", "duplicate"):
            with self.subTest(defect=defect):
                self.setUp()
                proof = next(proof for proof in self.execution.source_headers if proof.sheet_name == "Wide")
                if defect == "missing":
                    self.execution.source_headers.clear()
                elif defect == "incomplete":
                    proof.values.pop(28)
                elif defect == "dimensions":
                    proof.inspected_columns = 33
                elif defect == "expected_label":
                    proof.values[28] = "another_link"
                elif defect == "compact_label":
                    proof.values[1] = "ordinary_name"
                elif defect == "formula":
                    proof.formula_columns = (28,)
                elif defect == "duplicate":
                    self.execution.source_headers.append(proof.model_copy(deep=True))
                result = self.validate()
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertTrue(result.errors)
                self.assertTrue(any(finding.code == "association_evidence_mismatch" for finding in result.findings))


class FixedProvider:
    async def generate_recipe(self, summary, model, known_rules):
        return LLMResponse(model=model, content=recipe_for().model_dump_json())


class LegacyProcessor:
    """Su firma anterior verifica que no se envían nuevos kwargs sin políticas."""

    def inspect(self, name, content):
        return source_summary(name)

    def execute(self, name, content, recipe, summary):
        return ExecutionResult(sheets=[SheetExecution(
            sheet_name="data", block_id="records", role="observations",
            data_start_row=2, data_end_row=3, data_start_column=1, data_end_column=3,
            source_rows=3, source_columns=3, rows_before=2, rows_after=2,
            columns_before=["id", "Fecha", "count"], columns_after=["id", "Fecha", "count"],
        )])


class PolicyUseCaseCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_processor_and_validator_legacy_signatures_are_preserved_without_policies(self):
        result = await TabularEvaluationService(
            LegacyProcessor(), FixedProvider(), default_model="offline", max_upload_bytes=1000,
        ).execute("reviewed.csv", b"source")
        self.assertEqual(result.validation.status, EvaluationStatus.READY)

    async def test_catalog_policies_reach_processor_but_additional_text_does_not_authorize_them(self):
        class RecordingProcessor(LegacyProcessor):
            def execute(self, name, content, recipe, summary, *, policies=None):
                self.policies = policies
                execution = super().execute(name, content, recipe, summary)
                if policies:
                    execution.source_headers = complete_headers(summary)
                return execution
        processor = RecordingProcessor()
        service = TabularRuleService(MemoryCatalog(fixture_catalog()))
        evaluator = TabularEvaluationService(processor, FixedProvider(), default_model="offline",
                                       max_upload_bytes=1000, rule_service=service)
        result = await evaluator.execute("reviewed.csv", b"source", rule_profile="fixture")
        self.assertTrue(processor.policies.enabled)
        self.assertEqual(result.validation.status, EvaluationStatus.READY)
        await evaluator.execute("other.csv", b"source", rule_profile="fixture",
                                known_rules=json.dumps({"source_sha256": sha256(b"source").hexdigest()}))
        self.assertIsNone(processor.policies)
