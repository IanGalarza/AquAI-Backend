import unittest

from aquai.application.dtos.tabular import SheetSummary, StructuralSummary
from aquai.application.services.tabular_rules import TabularRuleService
from aquai.infrastructure.providers.tabular_rules import FileTabularRuleCatalog


class TemporalPriorityProfileTests(unittest.TestCase):
    def source(self, file_name, sheet_names):
        return StructuralSummary(file_name=file_name, file_type="xlsx", file_size_bytes=100,
            sheets=[SheetSummary(name=name, rows=2, columns=2, initial_columns=["id", "value"],
                                 inferred_types=["text", "text"], null_counts=[0, 0], sample=[])
                    for name in sheet_names])

    def test_belen_priority_has_versioned_user_origin(self):
        source = self.source("BaseCiano_PARA BELEN_5MAR2024 (2).XLSX", ("BASE CIANO", "BASE FQ", "REFERENCIAS"))
        composed = TabularRuleService(FileTabularRuleCatalog()).compose(source, "belen_cyanobacteria")
        policies = composed.preparation_policies.date_columns
        self.assertEqual(len(policies), 2)
        for policy in policies:
            self.assertEqual(policy.temporal_priority, "date_code_then_date")
            self.assertEqual(policy.version, "1.1.0")
            self.assertEqual(composed.metadata["origins"][policy.origin]["verification"], "user_confirmation")

    def test_temporal_priority_does_not_spread_to_other_workbook_profiles(self):
        service = TabularRuleService(FileTabularRuleCatalog())
        for profile, file_name, sheets in (
            ("self_sampling", "Tabla completa ciano RdlP.xlsx", ("tabla datos", "Datos ordenados por fecha", "Hoja1")),
            ("biological_counts", "datos biologicos tesis.xlsx", ("Recuentos", "Cianobacterias", "Hoja3", "Hoja1")),
        ):
            with self.subTest(profile=profile):
                composed = service.compose(self.source(file_name, sheets), profile)
                self.assertIsNone(composed.preparation_policies)
                self.assertNotIn("belen_temporal_priority_user_20261011", composed.metadata["origins"])
