from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from bench_pages.gh_pages import bootstrap_pages_checkout, publish_sweep_to_pages
from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestPages(unittest.TestCase):
    def test_bootstrap_pages_checkout_creates_fresh_orphan_layout(self) -> None:
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            pages_root = Path(temp_dir) / "pages"
            repo_root.mkdir()

            def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                if command[:6] == [
                    "git",
                    "-C",
                    str(repo_root),
                    "show-ref",
                    "--verify",
                    "refs/heads/gh-pages",
                ]:
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            bootstrap_pages_checkout(
                repo_root=repo_root,
                pages_root=pages_root,
                branch="gh-pages",
                runner=runner,
            )

            self.assertTrue((pages_root / ".nojekyll").exists())
            self.assertTrue((pages_root / "index.json").exists())
            self.assertTrue((pages_root / "index.html").exists())
            self.assertIn(
                ["git", "-C", str(repo_root), "show-ref", "--verify", "refs/heads/gh-pages"],
                commands,
            )
            self.assertIn(
                ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(pages_root)],
                commands,
            )
            self.assertIn(
                ["git", "-C", str(pages_root), "checkout", "--orphan", "gh-pages"],
                commands,
            )

    def test_publish_sweep_to_pages_writes_expected_layout(self) -> None:
        sweep_root = FIXTURE_DIR / "native_minimal"
        sweep = load_sweep(sweep_root)
        manifest = build_manifest(sweep)

        with tempfile.TemporaryDirectory() as temp_dir:
            pages_root = Path(temp_dir) / "pages"
            assets_root = Path(temp_dir) / "assets"
            assets_root.mkdir(parents=True)
            (assets_root / "site.css").write_text("body {}")
            (assets_root / "chart.umd.js").write_text("window.Chart = {};")

            publish_sweep_to_pages(
                pages_root=pages_root,
                sweep_root=sweep_root,
                manifest=manifest,
                sweep_html="<html><body>sweep</body></html>",
                index_html="<html><body>index</body></html>",
                assets_dir=assets_root,
            )

            sweep_id = manifest["sweep"]["id"]
            self.assertTrue((pages_root / "index.html").exists())
            self.assertTrue((pages_root / "index.json").exists())
            self.assertTrue((pages_root / "assets/site.css").exists())
            self.assertTrue((pages_root / "assets/chart.umd.js").exists())
            self.assertTrue((pages_root / f"sweeps/{sweep_id}/index.html").exists())
            self.assertTrue((pages_root / f"sweeps/{sweep_id}/manifest.json").exists())
            self.assertTrue(
                (
                    pages_root
                    / f"sweeps/{sweep_id}/artifacts/cases/case-000-threads_1/summary.json"
                ).exists()
            )

    def test_publish_sweep_to_pages_copies_profile_symbol_bundle(self) -> None:
        sweep_root = FIXTURE_DIR / "docker_minimal"
        sweep = load_sweep(sweep_root)
        manifest = build_manifest(sweep)

        with tempfile.TemporaryDirectory() as temp_dir:
            pages_root = Path(temp_dir) / "pages"
            assets_root = Path(temp_dir) / "assets"
            assets_root.mkdir(parents=True)
            (assets_root / "site.css").write_text("body {}")
            (assets_root / "chart.umd.js").write_text("window.Chart = {};")

            publish_sweep_to_pages(
                pages_root=pages_root,
                sweep_root=sweep_root,
                manifest=manifest,
                sweep_html="<html><body>sweep</body></html>",
                index_html="<html><body>index</body></html>",
                assets_dir=assets_root,
            )

            sweep_id = manifest["sweep"]["id"]
            self.assertTrue(
                (
                    pages_root
                    / f"sweeps/{sweep_id}/artifacts/cases/case-000-memory_limit_8g/symbols/nockchain-bench"
                ).exists()
            )

    def test_publish_sweep_to_pages_keeps_single_index_entry_without_replace(self) -> None:
        sweep_root = FIXTURE_DIR / "native_minimal"
        sweep = load_sweep(sweep_root)
        manifest = build_manifest(sweep)

        with tempfile.TemporaryDirectory() as temp_dir:
            pages_root = Path(temp_dir) / "pages"
            assets_root = Path(temp_dir) / "assets"
            assets_root.mkdir(parents=True)
            (assets_root / "site.css").write_text("body {}")
            (assets_root / "chart.umd.js").write_text("window.Chart = {};")

            publish_sweep_to_pages(
                pages_root=pages_root,
                sweep_root=sweep_root,
                manifest=manifest,
                sweep_html="<html><body>first</body></html>",
                index_html="<html><body>index</body></html>",
                assets_dir=assets_root,
            )
            publish_sweep_to_pages(
                pages_root=pages_root,
                sweep_root=sweep_root,
                manifest=manifest,
                sweep_html="<html><body>second</body></html>",
                index_html="<html><body>index</body></html>",
                assets_dir=assets_root,
            )

            index_entries = json.loads((pages_root / "index.json").read_text())
            self.assertEqual(len(index_entries), 1)

            replaced_manifest = copy.deepcopy(manifest)
            replaced_manifest["sweep"]["fixture_identity"] = "replacement-fixture"
            publish_sweep_to_pages(
                pages_root=pages_root,
                sweep_root=sweep_root,
                manifest=replaced_manifest,
                sweep_html="<html><body>third</body></html>",
                index_html="<html><body>index</body></html>",
                assets_dir=assets_root,
                replace=True,
            )
            replaced_entries = json.loads((pages_root / "index.json").read_text())
            self.assertEqual(len(replaced_entries), 1)
            self.assertEqual(replaced_entries[0]["fixture_identity"], "replacement-fixture")


if __name__ == "__main__":
    unittest.main()
