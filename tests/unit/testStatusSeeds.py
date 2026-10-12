"""Verifica los IDs y nombres acordados para los catálogos iniciales."""

import unittest

from aquai.infrastructure.persistence.configurations.statusSeeds import (
    MESSAGE_STATUS_SEEDS,
    PROCESSING_STATUS_SEEDS,
    message_status_seed_rows,
    processing_status_seed_rows,
)


class StatusSeedTests(unittest.TestCase):
    def test_processing_statuses_match_the_agreed_catalog(self) -> None:
        self.assertEqual(processing_status_seed_rows(), [
            {"id": 1, "name": "uploaded"},
            {"id": 2, "name": "processing"},
            {"id": 3, "name": "ready"},
            {"id": 4, "name": "ready_with_warnings"},
            {"id": 5, "name": "review_required"},
            {"id": 6, "name": "failed"},
        ])

    def test_message_statuses_match_the_agreed_catalog(self) -> None:
        self.assertEqual(message_status_seed_rows(), [
            {"id": 1, "name": "pending"},
            {"id": 2, "name": "processing"},
            {"id": 3, "name": "completed"},
            {"id": 4, "name": "failed"},
            {"id": 5, "name": "cancelled"},
        ])

    def test_migration_rows_are_independent_from_the_seed_definitions(self) -> None:
        for rows in (processing_status_seed_rows, message_status_seed_rows):
            with self.subTest(catalog=rows.__name__):
                original = rows()
                modified = rows()
                modified[0]["name"] = "modified"
                modified.append({"id": 99, "name": "extra"})
                self.assertEqual(rows(), original)

    def test_seed_ids_and_names_are_unique_and_rows_are_immutable(self) -> None:
        for seeds in (PROCESSING_STATUS_SEEDS, MESSAGE_STATUS_SEEDS):
            with self.subTest(catalog=seeds):
                self.assertEqual(len({seed.id for seed in seeds}), len(seeds))
                self.assertEqual(len({seed.name for seed in seeds}), len(seeds))
                with self.assertRaises(AttributeError):
                    setattr(seeds[0], "name", "modified")


if __name__ == "__main__":
    unittest.main()
