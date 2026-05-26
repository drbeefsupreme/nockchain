from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from bench_pages.artifacts import is_publish_artifact_path
from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest, build_sweep_id
try:
    from .support import create_partial_sweep_fixture
except ImportError:  # pragma: no cover - unittest discover imports as top-level modules.
    from support import create_partial_sweep_fixture


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestManifest(unittest.TestCase):
    def test_build_sweep_id_is_stable_for_same_fixture(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "docker_minimal")

        first_id = build_sweep_id(sweep)
        second_id = build_sweep_id(sweep)

        self.assertEqual(first_id, second_id)

    def test_build_manifest_preserves_all_source_fields_and_artifacts(self) -> None:
        root = FIXTURE_DIR / "docker_minimal"
        sweep = load_sweep(root)
        manifest = build_manifest(sweep)

        case_manifest = manifest["cases"][0]
        summary_keys = set(
            json.loads(
                (root / "cases/case-000-memory_limit_8g/summary.json").read_text()
            ).keys()
        )
        provenance_keys = set(
            json.loads(
                (root / "cases/case-000-memory_limit_8g/provenance.json").read_text()
            ).keys()
        )
        result_keys = set(
            json.loads(
                (root / "cases/case-000-memory_limit_8g/runs/run-0/result.json").read_text()
            ).keys()
        )
        inventory_paths = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file()
            and is_publish_artifact_path(path.relative_to(root))
        }

        self.assertTrue(summary_keys.issubset(case_manifest["summary"].keys()))
        self.assertTrue(provenance_keys.issubset(case_manifest["provenance"].keys()))
        self.assertTrue(result_keys.issubset(case_manifest["runs"][0]["result"].keys()))
        self.assertEqual(
            case_manifest["cpu_profile"]["profile_artifact"]["relative_path"],
            "cases/case-000-memory_limit_8g/profiles/samply-profile.json.gz",
        )
        self.assertEqual(
            case_manifest["cpu_profile"]["symbol_dir"]["relative_path"],
            "cases/case-000-memory_limit_8g/symbols",
        )
        self.assertEqual(
            case_manifest["cpu_profile"]["symbol_binary"]["relative_path"],
            "cases/case-000-memory_limit_8g/symbols/nockchain-bench",
        )
        self.assertEqual(
            case_manifest["cpu_profile"]["load_command"],
            "samply load --symbol-dir artifacts/cases/case-000-memory_limit_8g/symbols artifacts/cases/case-000-memory_limit_8g/profiles/samply-profile.json.gz",
        )
        self.assertEqual(
            {entry["relative_path"] for entry in manifest["artifact_inventory"]},
            inventory_paths,
        )

    def test_build_manifest_excludes_run_work_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied_root = Path(temp_dir) / "docker_pma_minimal"
            shutil.copytree(FIXTURE_DIR / "docker_pma_minimal", copied_root)
            work_file = (
                copied_root
                / "cases/case-000-memory_limit_8g/runs/run-0/work/replay-pma/0.pma"
            )
            work_file.parent.mkdir(parents=True)
            work_file.write_text("transient pma work file")

            manifest = build_manifest(load_sweep(copied_root))

        self.assertNotIn(
            "cases/case-000-memory_limit_8g/runs/run-0/work/replay-pma/0.pma",
            {entry["relative_path"] for entry in manifest["artifact_inventory"]},
        )

    def test_build_manifest_uses_provenance_digest_as_primary_docker_identity(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "docker_minimal")

        manifest = build_manifest(sweep)

        docker_image = manifest["docker_images"][0]
        self.assertEqual(
            docker_image["canonical_identity"],
            docker_image["provenance_image_digest"],
        )
        self.assertIsNone(docker_image["local_image_id"])

    def test_build_manifest_surfaces_generic_pma_context(self) -> None:
        pma_manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_pma_minimal"))
        mixed_manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_fixture_axis_pma"))

        self.assertEqual(pma_manifest["cases"][0]["runtime_flavor"], "pma")
        self.assertEqual(pma_manifest["cases"][0]["boot_source"], "checkpoint")
        self.assertEqual(pma_manifest["cases"][0]["boot_event_num"], 42)
        self.assertEqual(pma_manifest["cases"][0]["pma_work_dir_mode"], "docker_tmpfs")
        self.assertEqual(pma_manifest["sweep"]["runtime_summary"], "pma")
        self.assertEqual(pma_manifest["sweep"]["boot_source_summary"], "checkpoint")
        self.assertEqual(pma_manifest["sweep"]["pma_work_dir_summary"], "docker_tmpfs")
        self.assertEqual(mixed_manifest["sweep"]["runtime_summary"], "pma")
        self.assertEqual(mixed_manifest["sweep"]["fixture_summary"], "2 fixtures")

    def test_unknown_additive_fields_survive_manifest_roundtrip(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_pma_minimal"))

        self.assertIn("new_metric_total", manifest["cases"][0]["summary"])
        self.assertIn("future_probe_status", manifest["cases"][0]["provenance"])

    def test_build_sweep_id_distinguishes_fixture_identity(self) -> None:
        mixed = load_sweep(FIXTURE_DIR / "native_fixture_axis_pma")
        mixed_single_fixture = copy.deepcopy(mixed)

        mixed_single_fixture.cases[1].resolved_case["fixture_sha256_hex"] = (
            mixed_single_fixture.cases[0].resolved_case["fixture_sha256_hex"]
        )
        mixed_single_fixture.cases[1].provenance["fixture_sha256_hex"] = (
            mixed_single_fixture.cases[0].provenance["fixture_sha256_hex"]
        )

        self.assertNotEqual(build_sweep_id(mixed), build_sweep_id(mixed_single_fixture))

    def test_build_manifest_preserves_partial_sweep_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            partial_root = create_partial_sweep_fixture(Path(temp_dir))
            manifest = build_manifest(load_sweep(partial_root))

        self.assertEqual(manifest["sweep"]["completion_state"], "incomplete")
        self.assertEqual(
            manifest["sweep"]["missing_top_level_artifacts"],
            ["comparison.json", "verdict.json"],
        )
        self.assertEqual(manifest["sweep"]["scheduled_case_count"], 3)
        self.assertEqual(manifest["sweep"]["materialized_case_count"], 2)
        self.assertEqual(manifest["sweep"]["complete_case_count"], 1)
        self.assertEqual(manifest["sweep"]["partial_case_count"], 1)
        self.assertEqual(manifest["sweep"]["missing_case_count"], 1)
        self.assertIsNone(manifest["source_artifacts"]["comparison"])
        self.assertIsNone(manifest["source_artifacts"]["verdict"])
        self.assertEqual(manifest["cases"][1]["completion_state"], "partial")
        self.assertEqual(
            manifest["cases"][1]["missing_artifacts"],
            ["summary.json", "verdict.json"],
        )
        self.assertEqual(manifest["cases"][2]["completion_state"], "missing")
        self.assertEqual(
            manifest["cases"][2]["requested_case"]["execution"]["Docker"]["memory_limit"],
            "2g",
        )


if __name__ == "__main__":
    unittest.main()
