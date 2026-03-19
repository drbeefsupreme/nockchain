from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from bench_pages.loader import load_sweep
from bench_pages.manifest import build_sweep_id


FIXTURE_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[3]


class TestCli(unittest.TestCase):
    def test_publish_sweep_dry_run_cli_smoke(self) -> None:
        sweep_root = FIXTURE_DIR / "native_minimal"
        expected_id = build_sweep_id(load_sweep(sweep_root))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "native-pages"
            env = os.environ.copy()
            env["UV_CACHE_DIR"] = str(Path(temp_dir) / "uv-cache")
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--project",
                    "scripts/bench_pages",
                    "publish-sweep",
                    "--sweep-root",
                    "scripts/bench_pages/tests/fixtures/native_minimal",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn(expected_id, result.stdout)
            self.assertIn(str(output_dir), result.stdout)


if __name__ == "__main__":
    unittest.main()
