from __future__ import annotations

import unittest
from pathlib import Path

from bench_pages.loader import load_sweep
from bench_pages.manifest import build_manifest
from bench_pages.render import render_index_page, render_sweep_page


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestRenderSweepPage(unittest.TestCase):
    """Tests for the table-first sweep page redesign."""

    def test_comparison_table_leads_page(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Cross-Case Comparison", page)
        self.assertIn("comparison-table", page)
        self.assertIn(manifest["sweep"]["id"], page)

    def test_comparison_table_contains_case_rows(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("case-000-memory_limit_8g", page)

    def test_compact_valuestats_in_comparison(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("vs-primary", page)
        self.assertIn("vs-detail", page)

    def test_null_metrics_render_as_na(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # native_minimal has null peak_process_rss_bytes in runs
        self.assertIn("n/a", page)

    def test_no_chart_payloads(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertNotIn("chart-payloads", page)
        self.assertNotIn("chart.umd.js", page)
        self.assertNotIn("data-chart-id", page)
        self.assertNotIn('<canvas', page)

    def test_per_case_run_tables(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("run-table", page)
        self.assertIn("run-0", page)

    def test_evidence_drawers_present(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        for label in ("Provenance", "Requested Case", "Resolved Case",
                       "Verdict Detail", "Raw JSON"):
            with self.subTest(drawer=label):
                self.assertIn(label, page)

    def test_full_fidelity_in_drawers(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        for field in ("median", "stddev", "mad", "cv"):
            with self.subTest(field=field):
                self.assertIn(field, page)
        # Provenance data reachable
        self.assertIn("fixture_sha256_hex", page)

    def test_artifact_browser_present(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Artifact Browser", page)

    def test_docker_images_section(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Docker Images", page)

    def test_docker_images_absent_for_native(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # Docker section is conditional — should not appear for native sweeps
        self.assertNotIn("Docker Images", page)

    def test_humanized_bytes_in_comparison(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        # docker_minimal has peak_process_rss_bytes with large values
        self.assertIn("GiB", page)

    def test_validation_drawer_present_when_data_exists(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Validation", page)


class TestRenderIndexPage(unittest.TestCase):
    def test_links_multiple_sweeps(self) -> None:
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

        page = render_index_page(entries)

        self.assertIn(entries[0]["path"], page)
        self.assertIn(entries[1]["path"], page)
        self.assertIn("native", page)
        self.assertIn("docker", page)
        self.assertIn("Valid", page)


if __name__ == "__main__":
    unittest.main()
