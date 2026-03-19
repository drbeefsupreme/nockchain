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

        self.assertNotIn("24.000000", page)

    def test_column_headers_include_units(self) -> None:
        """Metric column headers show units where applicable."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("(blk/s)", page)
        self.assertIn("(s)", page)
        self.assertIn("(ms)", page)

    def test_column_headers_have_tooltips(self) -> None:
        """Column headers have title attributes for hover tooltips."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn('title="Blocks replayed per second', page)

    def test_metric_cells_have_tooltips(self) -> None:
        """ValueStats cells have tooltips showing full breakdown."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # ValueStats tooltip includes median, min, max, and samples count
        self.assertIn("median:", page)
        self.assertIn("samples:", page)

    def test_verdict_cells_have_tooltips(self) -> None:
        """Verdict cells explain what Valid/Invalid means."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("All measured runs completed within acceptable parameters", page)

    def test_all_columns_shown_including_zeros(self) -> None:
        """All metric columns appear even when values are zero."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # Zero-valued columns like checkpoints should still appear.
        self.assertIn("Ckpts", page)

    def test_minor_major_faults_always_shown(self) -> None:
        """Minor/Major Fault columns appear even when data is null."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # native_minimal has null faults — should show as n/a, not be hidden.
        self.assertIn("Minor Fault", page)
        self.assertIn("Major Fault", page)

    def test_null_metrics_render_as_na(self) -> None:
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

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

    def test_strip_charts_present(self) -> None:
        """Strip charts render for metrics with ValueStats run data."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("Run Spread", page)
        self.assertIn("strip-chart", page)
        self.assertIn("strip-median", page)
        self.assertIn("strip-dot", page)
        # Case ID appears as a label in the chart
        self.assertIn("case-000-threads_1", page)

    def test_strip_charts_have_tooltips(self) -> None:
        """Strip chart dots have tooltips with run values."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertIn("run-0:", page)
        self.assertIn("median:", page)

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
        self.assertIn("fixture_sha256_hex", page)

    def test_evidence_browser_has_field_tooltips(self) -> None:
        """KV tables in evidence browser have tooltips on known field keys."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        # The summary (full) card should have tooltips on known metric keys
        self.assertIn('title="Blocks replayed per second', page)

    def test_byte_fields_humanized_in_evidence(self) -> None:
        """Large byte values in evidence tables show humanized form."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        page = render_sweep_page(manifest)

        # docker_minimal provenance has total_memory_bytes: 68719476736
        # Should show humanized + raw
        self.assertIn("GiB", page)
        self.assertIn("raw-bytes", page)

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
        """ValueStats detail shows compact range (min-max) not verbose labels."""
        manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        page = render_sweep_page(manifest)

        self.assertNotIn("min 20", page)
        self.assertNotIn("max 22", page)
        self.assertIn("\u2013", page)


class TestRenderIndexPage(unittest.TestCase):

    def _make_entries(self) -> list:
        native_manifest = build_manifest(load_sweep(FIXTURE_DIR / "native_minimal"))
        docker_manifest = build_manifest(load_sweep(FIXTURE_DIR / "docker_minimal"))
        return [
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

    def test_links_multiple_sweeps(self) -> None:
        entries = self._make_entries()
        page = render_index_page(entries)

        self.assertIn(entries[0]["path"], page)
        self.assertIn(entries[1]["path"], page)
        self.assertIn("native", page)
        self.assertIn("docker", page)
        self.assertIn("Valid", page)

    def test_sweep_checkboxes_present(self) -> None:
        """Each sweep row has a checkbox for cross-sweep comparison."""
        entries = self._make_entries()
        page = render_index_page(entries)

        self.assertIn("sweep-check", page)
        self.assertIn('data-id="', page)
        self.assertIn('data-path="', page)

    def test_compare_button_present(self) -> None:
        """Index page has a Compare Selected button."""
        entries = self._make_entries()
        page = render_index_page(entries)

        self.assertIn("compare-btn", page)
        self.assertIn("Compare Selected", page)

    def test_comparison_results_container_present(self) -> None:
        """Index page has a container for comparison results."""
        entries = self._make_entries()
        page = render_index_page(entries)

        self.assertIn("comparison-results", page)

    def test_comparison_js_contains_delta_logic(self) -> None:
        """Index page JS includes delta computation and metric config."""
        entries = self._make_entries()
        page = render_index_page(entries)

        self.assertIn("Cross-Sweep Delta", page)
        self.assertIn("throughput_blocks_per_second", page)
        self.assertIn("manifest.json", page)


if __name__ == "__main__":
    unittest.main()
