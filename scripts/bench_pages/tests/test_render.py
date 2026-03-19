from __future__ import annotations

import unittest
from pathlib import Path

from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest
from bench_pages.render import render_index_page, render_sweep_page


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestRender(unittest.TestCase):
    def test_render_sweep_page_includes_exhaustive_sections(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))

        html = render_sweep_page(manifest)

        self.assertIn(manifest["sweep"]["id"], html)
        self.assertIn("Top-level artifacts", html)
        self.assertIn("cases/case-000-memory_limit_8g/summary.json", html)
        self.assertIn("case-000-memory_limit_8g", html)
        self.assertIn("throughput_blocks_per_second", html)
        self.assertIn("fixture_sha256_hex", html)
        self.assertIn("Docker images", html)
        self.assertIn("Artifact browser", html)

    def test_render_sweep_page_renders_null_metrics_without_charting_zero(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))

        html = render_sweep_page(manifest)

        self.assertIn("peak_process_rss_bytes", html)
        self.assertIn("n/a", html)
        self.assertNotIn('"key":"peak_process_rss_bytes","labels":["case-000-memory_limit_8g"],"values":[0]', html)

    def test_render_sweep_page_displays_valuestats_and_chart_payloads(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))

        html = render_sweep_page(manifest)

        self.assertIn("median", html)
        self.assertIn("min", html)
        self.assertIn("max", html)
        self.assertIn("mad", html)
        self.assertIn("stddev", html)
        self.assertIn("cv", html)
        self.assertIn('"kind":"case-summary"', html)
        self.assertIn('"stat":"median"', html)
        self.assertIn('"kind":"case-values"', html)
        self.assertIn('"stat":"values"', html)

    def test_render_index_page_links_multiple_sweeps(self) -> None:
        native_manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        docker_manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        entries = [
            {
                "id": native_manifest["sweep"]["id"],
                "path": f"sweeps/{native_manifest['sweep']['id']}/index.html",
                "execution_mode": native_manifest["sweep"]["execution_mode"],
                "verdict": "Valid",
                "fixture_identity": native_manifest["sweep"]["fixture_identity"],
            },
            {
                "id": docker_manifest["sweep"]["id"],
                "path": f"sweeps/{docker_manifest['sweep']['id']}/index.html",
                "execution_mode": docker_manifest["sweep"]["execution_mode"],
                "verdict": "Valid",
                "fixture_identity": docker_manifest["sweep"]["fixture_identity"],
            },
        ]

        html = render_index_page(entries)

        self.assertIn(entries[0]["path"], html)
        self.assertIn(entries[1]["path"], html)
        self.assertIn("native", html)
        self.assertIn("docker", html)
        self.assertIn("Valid", html)


if __name__ == "__main__":
    unittest.main()
