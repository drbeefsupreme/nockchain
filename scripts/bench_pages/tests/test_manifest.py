from __future__ import annotations

import json
import unittest
from pathlib import Path

from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest, build_sweep_id


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

    def test_build_manifest_uses_provenance_digest_as_primary_docker_identity(self) -> None:
        sweep = load_sweep(FIXTURE_DIR / "docker_minimal")

        manifest = build_manifest(sweep)

        docker_image = manifest["docker_images"][0]
        self.assertEqual(
            docker_image["canonical_identity"],
            docker_image["provenance_image_digest"],
        )
        self.assertIsNone(docker_image["local_image_id"])


if __name__ == "__main__":
    unittest.main()
