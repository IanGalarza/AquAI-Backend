import unittest

from aquai.application.dtos.tabular import SheetSummary, StructuralSummary
from aquai.application.services.tabularRules import TabularRuleService
from aquai.infrastructure.providers.tabularRules import FileTabularRuleCatalog


class VerifiedLayoutScopeTests(unittest.TestCase):
    source_hash = "5bbf70623f4cbc0e2e8d82c22ab097c0144f1a3d39e1182c64b622829ee0bd2f"

    def source(self, file_name="BaseCiano_PARA BELEN_5MAR2024 (2).XLSX"):
        return StructuralSummary(
            file_name=file_name, file_type="xlsx", file_size_bytes=100,
            sheets=[SheetSummary(name=name, rows=3, columns=2, initial_columns=["id", "value"],
                                 inferred_types=["text", "text"], null_counts=[0, 0], sample=[])
                    for name in ("BASE CIANO", "BASE FQ", "REFERENCIAS")],
        )

    def test_documented_structure_keeps_its_verified_file_hash_and_inspection_origin(self):
        service = TabularRuleService(FileTabularRuleCatalog())
        composition = service.compose(self.source(), "belen_cyanobacteria", source_sha256=self.source_hash)
        rule = next(rule for rule in composition.prompt_rules["file_specific_rules"]
                    if rule["id"] == "belen_verified_structural_limits")
        self.assertEqual(rule["sha256"], self.source_hash)
        self.assertEqual(rule["version"], "1.0.0")
        self.assertEqual(rule["scope"], "file_specific")
        self.assertEqual(composition.metadata["origins"][rule["origin"]]["verification"],
                         "full_sheet_values_and_formulas")
        # La evidencia de estructura no constituye una corrección de celdas.
        self.assertNotIn("preparation_policies", rule)
        self.assertIn("documented_structure", rule)

    def test_documented_structure_cannot_spread_to_other_file_revision_or_profile(self):
        service = TabularRuleService(FileTabularRuleCatalog())
        for file_name, profile, source_hash in (
            ("BaseCiano_PARA BELEN_5MAR2024 (2).XLSX", "belen_cyanobacteria", "0" * 64),
            ("other.xlsx", "belen_cyanobacteria", self.source_hash),
            ("Tabla completa ciano RdlP.xlsx", "self_sampling", self.source_hash),
            ("datos biologicos tesis.xlsx", "biological_counts", self.source_hash),
        ):
            with self.subTest(file_name=file_name, profile=profile):
                composition = service.compose(self.source(file_name), profile, source_sha256=source_hash)
                ids = {rule["id"] for rule in composition.prompt_rules.get("file_specific_rules", [])}
                self.assertNotIn("belen_verified_structural_limits", ids)
