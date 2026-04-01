from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from bench_pages.errors import ValidationError
from bench_pages.loader import load_sweep


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestLoadSweep(unittest.TestCase):
    def test_load_sweep_normalizes_native_execution_mode(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "native_minimal")

        self.assertEqual(sweep.execution_mode, "native")
        self.assertEqual(len(sweep.cases), 1)
        self.assertEqual(sweep.cases[0].execution_mode, "native")

    def test_load_sweep_normalizes_docker_execution_mode(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "docker_minimal")

        self.assertEqual(sweep.execution_mode, "docker")
        self.assertEqual(len(sweep.cases), 1)
        self.assertEqual(sweep.cases[0].execution_mode, "docker")

    def test_load_sweep_records_every_artifact_in_tree_walk_inventory(self) -> None:
        root = FIXTURE_DIR / "docker_minimal"
        sweep = load_sweep(root)

        expected = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
        )
        actual = sorted(record.relative_path for record in sweep.artifact_inventory)
        self.assertEqual(actual, expected)

    def test_load_sweep_reads_case_cpu_profile_metadata_when_present(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "docker_minimal")

        cpu_profile = sweep.cases[0].cpu_profile
        self.assertIsNotNone(cpu_profile)
        assert cpu_profile is not None
        self.assertEqual(cpu_profile["profiler_kind"], "samply")
        self.assertEqual(
            cpu_profile["output_relative_path"],
            "profiles/samply-profile.json.gz",
        )
        self.assertEqual(
            cpu_profile["symbol_dir_relative_path"],
            "symbols",
        )
        self.assertEqual(
            cpu_profile["symbol_binary_relative_path"],
            "symbols/nockchain-bench",
        )

    def test_load_sweep_rejects_missing_required_top_level_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied_root = Path(temp_dir) / "docker_minimal"
            shutil.copytree(FIXTURE_DIR / "docker_minimal", copied_root)
            (copied_root / "comparison.json").unlink()

            with self.assertRaises(ValidationError):
                load_sweep(copied_root)

    def test_load_sweep_accepts_native_pma_fixture(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "native_pma_minimal")

        self.assertEqual(sweep.execution_mode, "native")
        self.assertEqual(len(sweep.cases), 1)
        self.assertEqual(sweep.cases[0].execution_mode, "native")
        self.assertEqual(sweep.cases[0].provenance["runtime_flavor"], "pma")
        self.assertEqual(sweep.cases[0].provenance["boot_source"], "checkpoint")
        self.assertEqual(sweep.cases[0].provenance["boot_event_num"], 42)

    def test_load_sweep_accepts_fixture_axis_pma_fixture(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "native_fixture_axis_pma")

        self.assertEqual(sweep.execution_mode, "native")
        self.assertEqual(len(sweep.cases), 2)
        fixture_identities = {
            case.resolved_case["fixture_sha256_hex"]
            for case in sweep.cases
        }
        boot_events = {case.provenance["boot_event_num"] for case in sweep.cases}

        self.assertEqual(len(fixture_identities), 2)
        self.assertEqual(boot_events, {42, 84})


if __name__ == "__main__":
    unittest.main()
