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

    def test_compact_number_formatting(self) -> None:
        """Numbers use compact formatting, not full 6-decimal precision."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        # Full precision numbers should NOT appear in comparison/run tables.
        # The compact format for 24.0 is "24", not "24.000000".
        self.assertNotIn("24.000000", page)

    def test_zero_columns_filtered_from_comparison(self) -> None:
        """Columns where all cases have zero/null values are omitted."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # native_minimal has checkpoint_count with all-zero ValueStats.
        # It should be filtered from the comparison table header row.
        # But the key still appears in the evidence browser (full fidelity).
        # Count header occurrences: "Ckpts" should not appear as a <th>.
        self.assertNotIn(">Ckpts<", page)

    def test_zero_columns_filtered_from_run_table(self) -> None:
        """Run table omits columns where all runs have trivial values."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # native run has checkpoint_count=0, checkpoint_total_time_secs=0, etc.
        self.assertNotIn(">Ckpt Tot<", page)

    def test_null_metrics_render_as_na(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # native_minimal has null peak_rss in runs and summary.
        # n/a appears in the evidence browser's full summary view.
        self.assertIn("n/a", page)

    def test_no_chart_payloads(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertNotIn("chart-payloads", page)
        self.assertNotIn("chart.umd.js", page)
        self.assertNotIn("data-chart-id", page)
        self.assertNotIn("<canvas", page)

    def test_per_case_run_tables(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("run-table", page)
        self.assertIn("run-0", page)

    def test_evidence_browser_present(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Case Evidence Browser", page)
        self.assertIn("browser-layout", page)
        for label in ("Provenance", "Requested Case", "Resolved Case",
                       "Verdict Detail", "Raw JSON"):
            with self.subTest(panel=label):
                self.assertIn(label, page)

    def test_full_fidelity_in_evidence_browser(self) -> None:
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

        self.assertNotIn("Docker Images", page)

    def test_humanized_bytes_in_comparison(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("GiB", page)

    def test_validation_drawer_present_when_data_exists(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Validation", page)

    def test_detail_line_uses_range_format(self) -> None:
        """ValueStats detail shows compact range (min–max) not verbose labels."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # Should use en-dash range format, not "min X · max Y" format.
        self.assertNotIn("min 20", page)
        self.assertNotIn("max 22", page)
        # The range should contain an en-dash (U+2013).
        self.assertIn("\u2013", page)


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
